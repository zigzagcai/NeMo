import gc
import json
import os
import shutil
import torch

from argparse import ArgumentParser
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from transformers import LlamaConfig, LlamaForCausalLM
from lightning_fabric.utilities.cloud_io import get_filesystem

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default=None, required=True, help="Path to NeMo checkpoint about 1B LLaMA-2 weights",
    )
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Path to write HF model")
    args = parser.parse_args()
    return args

def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

# todo: code refine
# todo: make checkpoint decoupled from ptl
def load_checkpoint():
    pass

def convert(nemo_path, model_path, safe_serialization=True):

    checkpoint = load_checkpoint(nemo_path)

    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    num_layers = 20
    num_attention_heads = 16
    hidden_size = 2048
    head_size = hidden_size // num_attention_heads

    index_dict = {"weight_map": {}}
    for l in range(int(num_layers)):
        state_dict = {}
        filename = f"pytorch_model-{l + 1}-of-{num_layers + 1}.bin"
        # qkv
        qkv_weights = checkpoint['state_dict'][f'model.decoder.layers.{l}.self_attention.linear_qkv.weight'].reshape(3 * num_attention_heads, head_size, hidden_size)
        state_dict[f'model.layers.{l}.self_attn.q_proj.weight'] = qkv_weights[0::3, :, :].reshape(hidden_size, hidden_size)
        state_dict[f'model.layers.{l}.self_attn.k_proj.weight'] = qkv_weights[1::3, :, :].reshape(hidden_size, hidden_size)
        state_dict[f'model.layers.{l}.self_attn.v_proj.weight'] = qkv_weights[2::3, :, :].reshape(hidden_size, hidden_size)

        # attention dense
        state_dict[f'model.layers.{l}.self_attn.o_proj.weight'] = checkpoint['state_dict'][f'model.decoder.layers.{l}.self_attention.linear_proj.weight']

        # MLP      
        mlp_cat_down_gate_weight = checkpoint['state_dict'][f'model.decoder.layers.{l}.mlp.linear_fc1.weight']
        mlp_cat_down_gate_weight_list = torch.split(mlp_cat_down_gate_weight, mlp_cat_down_gate_weight.shape[0]//2)
        state_dict[f'model.layers.{l}.mlp.gate_proj.weight'] = mlp_cat_down_gate_weight_list[0]
        state_dict[f'model.layers.{l}.mlp.up_proj.weight'] = mlp_cat_down_gate_weight_list[1]
        state_dict[f'model.layers.{l}.mlp.down_proj.weight'] = checkpoint['state_dict'][f'model.decoder.layers.{l}.mlp.linear_fc2.weight']

        # LayerNorm
        state_dict[f'model.layers.{l}.input_layernorm.weight'] = checkpoint['state_dict'][f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight']
        state_dict[f'model.layers.{l}.post_attention_layernorm.weight'] = checkpoint['state_dict'][f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight']

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
    
    filename = f"pytorch_model-{num_layers + 1}-of-{num_layers + 1}.bin"
    state_dict = {
        "model.embed_tokens.weight": checkpoint['state_dict'][f'model.embedding.word_embeddings.weight'],
        "model.norm.weight": checkpoint['state_dict'][f'model.decoder.final_layernorm.weight'],
        "lm_head.weight": checkpoint['state_dict'][f'model.output_layer.weight'],
    }
    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
    
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=5632,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_layers,
        rms_norm_eps=0.00001,
        num_key_value_heads=num_attention_heads,
        vocab_size=32000,
        rope_theta=10000.0,
        max_position_embeddings=4096,
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del checkpoint
    gc.collect()

    print("Loading the checkpoint in a Llama model.")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path
    model.config.torch_dtype = torch.bfloat16
    print("Saving in the Transformers format.")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    # shutil.rmtree(tmp_model_path)

if __name__ == '__main__':
    args = get_args()
    convert(
        nemo_path=args.input_dir,
        model_path=args.output_dir,
        safe_serialization=args.safe_serialization,
        )
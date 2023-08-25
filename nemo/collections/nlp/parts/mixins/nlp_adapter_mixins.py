# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import AdapterName
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.classes.mixins.adapter_mixins import (
    AdapterConfig,
    AdapterModelPTMixin,
    AdapterModuleMixin,
    MultiAdaterConfig,
    _prepare_default_adapter_config,
)
from nemo.utils import logging, logging_mode, model_utils


class NLPAdapterModelMixin(AdapterModelPTMixin):
    """ NLP Adapter Mixin that can augment any Encoder module with Adapter module support.
    # Todo rewrite doc string
    This mixin class should be used only with a top level ModelPT subclass, that includes an `encoder` submodule.
    This mixin class adds several utility methods which are propagated to the `encoder`.

    An Adapter module is any Pytorch nn.Module that possess a few properties :

    - It's input and output dimension are the same, while the hidden dimension need not be the same.
    - The final layer of the Adapter module is zero-initialized, so that the residual connection to the adapter
        yields the original output.

    This mixin adds the following instance variables to the class this inherits it:

        -   `adapter_layer`: A torch.nn.ModuleDict(), whose keys are the names of the adapter (globally unique),
                and values are the Adapter nn.Module().
        -   `adapter_cfg`: A OmegaConf DictConfig object that holds the config of the adapters that are initialized.
        -   `adapter_global_cfg_key`: A str representing a key in the model config that can be provided by the user.
                The value resolves to `global_cfg`, and can be overridden via `model.cfg.adapters.global_cfg.*`.

    **Note**: This module **is** responsible for maintaining its config. At the ModelPT level, it will access and
        write Adapter config information to `self.cfg.adapters`.
    """

    def _get_all_keys(self,):
        """
        Returns all the keys in the model
        """
        k = [n for n, p in self.named_parameters()]
        return set(k)

    def add_adapters(
        self, names: Union[AdapterName, List[AdapterName]], cfgs: Union[AdapterConfig, List[AdapterConfig]]
    ):
        """
        High level API to add one or more adapter modules to the model, and freeze the base weights

        Args:
            names: One or more globally unique names for the adapter. Will be used to access, enable and disable adapters.
            cfgs: One or more DictConfigs that contains at the bare minimum `__target__` to instantiate a new Adapter module.
        """
        if not isinstance(names, List):
            names = [names]
        if isinstance(cfgs, MultiAdaterConfig):
            cfgs = cfgs.list()
        elif not isinstance(cfgs, List):
            cfgs = [cfgs]
        assert len(names) == len(cfgs), f"Lengths of `names` ({len(names)}) and `cfgs` ({len(cfgs)}) do not match."

        self.base_keys = self._get_all_keys()
        self.freeze()
        logging.info(f"Before adding PEFT params:\n{self.summarize()}")

        for peft_name, peft_cfg in zip(names, cfgs):
            for _, module in self.named_modules():
                if isinstance(module, AdapterModuleMixin):
                    if model_utils.import_class_by_path(peft_cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(name=peft_name, cfg=peft_cfg)

            # Update the model.cfg with information about the new adapter from cfg
            module_name, adapter_name = self.resolve_adapter_module_name_(peft_name)
            with open_dict(self.cfg):
                # Construct the minimum config required to be updated by adapter implementations
                if 'adapters' not in self.cfg:
                    self.cfg.adapters = OmegaConf.create({})

                self.cfg.adapters = _prepare_default_adapter_config(
                    global_key=self.adapter_global_cfg_key,
                    meta_key=self.adapter_metadata_cfg_key,
                    cfg=self.cfg.adapters,
                )

                # Inject the module name in the adapter metadata cfg
                gcfg = self.adapter_global_cfg_key
                mcfg = self.adapter_metadata_cfg_key
                self.cfg.adapters[gcfg][mcfg]['modules'][adapter_name] = module_name

                self.cfg.adapters[adapter_name] = OmegaConf.create(peft_cfg)

        logging.info(f"After adding PEFT params:\n{self.summarize()}")
        self.adapter_keys = self._get_all_keys() - self.base_keys

    @classmethod
    def restore_from_nemo_with_adapter(
        cls,
        restore_path: str,
        adapter_names: Union[AdapterName, List[AdapterName]],
        adapter_cfgs: Union[AdapterConfig, List[AdapterConfig]],
        strict: bool = True,
        map_location: Optional[torch.device] = None,
        trainer: Optional[Trainer] = None,
    ):
        if not isinstance(adapter_names, List):
            adapter_names = [adapter_names]
        if isinstance(adapter_cfgs, MultiAdaterConfig):
            adapter_cfgs = adapter_cfgs.list()
        elif not isinstance(adapter_cfgs, List):
            adapter_cfgs = [adapter_cfgs]
        assert len(adapter_names) == len(
            adapter_cfgs
        ), f"Lengths of `names` ({len(adapter_names)}) and `cfgs` ({len(adapter_cfgs)}) do not match."

        save_restore_connector = NLPSaveRestoreConnector()

        cls.update_save_restore_connector(save_restore_connector)
        conf, instance, state_dict = save_restore_connector.load_config_and_state_dict(
            cls, restore_path=restore_path, map_location=map_location, trainer=trainer
        )

        instance.add_adapters(adapter_names, adapter_cfgs)

        state_dict = save_restore_connector.modify_state_dict(conf, state_dict)
        save_restore_connector.load_instance_with_state_dict(instance, state_dict, strict)
        logging.info(f'Model {instance.__class__.__name__} was successfully restored from {restore_path}.')

        return instance

from abc import ABC, abstractmethod
import numpy as np
from .hook_fn import activate_on_registers
from functools import partial

class HookManager(ABC):
  def __init__(self, model, debug = False):
    self.model = model
    self.debug = debug

    # Hooks
    self.hooks = {
      "log_layer_outputs": [],
      "log_neuron_activations": [],
      "log_attention_maps": [],
      "intervene_register_neurons": [],
    }

    self.register_neurons_intervention = None

    # Logs of internal model data
    self.logs = {
      "attention_maps": [],
      "layer_outputs": [],
      "neuron_activations": [],
    }

  @abstractmethod
  def reinit(self):
    pass

  @abstractmethod
  def finalize(self):
    pass

  def unregister_hook(self, hook):
    hook.remove()

  def unregister_all_hooks(self):
    for hook_name in self.hooks:
      if isinstance(self.hooks[hook_name], list):
        for hook in self.hooks[hook_name]:
          self.unregister_hook(hook)
      elif isinstance(self.hooks[hook_name], dict):
        for key in self.hooks[hook_name]:
          self.unregister_hook(self.hooks[hook_name][key])
      else:
        raise ValueError(f"Invalid type for hook name: {type(self.hooks[hook_name])}")

  def get_attention_maps(self):
    assert self.debug, "Debug mode must be enabled to get attention maps"
    if not self.logs["attention_maps"]:
      return None
    return np.concatenate(self.logs["attention_maps"], axis=0)

  def get_layer_outputs(self):
    assert self.debug, "Debug mode must be enabled to get layer outputs"
    if not self.logs["layer_outputs"]:
      return None
    return np.concatenate(self.logs["layer_outputs"], axis=0)

  def get_neuron_activations(self):
    assert self.debug, "Debug mode must be enabled to get neuron activations"
    if not self.logs["neuron_activations"]:
      return None
    return np.concatenate(self.logs["neuron_activations"], axis=0)

  def set_debug(self, debug):
    self.debug = debug

  def intervene_register_neurons(self, num_registers, neurons_to_ablate, scale = 1.0, normal_values = "zero", bottom_frac = 0.75):
    self.register_neurons_intervention = {
      "neurons_to_ablate": neurons_to_ablate,
      "num_registers": num_registers,
      "scale": scale,
      "normal_values": normal_values,
      "bottom_frac": bottom_frac,
    }


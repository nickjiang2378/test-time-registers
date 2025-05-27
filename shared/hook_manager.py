from abc import ABC, abstractmethod
import numpy as np
from .hook_fn import activate_on_registers, log_internal
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
  def attn_post_softmax_component(self, layer):
    pass

  @abstractmethod
  def layer_output_component(self, layer):
    pass

  @abstractmethod
  def neuron_activation_component(self, layer):
    pass

  @abstractmethod
  def num_layers(self):
    pass

  def initialize_log_hooks(self):
    for layer in range(self.num_layers()):
      # Attention maps
      log_attention_hook = self.attn_post_softmax_component(layer).register_forward_hook(partial(log_internal, store = self.logs["attention_maps"]))
      self.hooks["log_attention_maps"].append(log_attention_hook)

      # Layer outputs
      log_layer_output_hook = self.layer_output_component(layer).register_forward_hook(partial(log_internal, store = self.logs["layer_outputs"]))
      self.hooks["log_layer_outputs"].append(log_layer_output_hook)

      # Neuron activations
      log_neuron_activations_hook = self.neuron_activation_component(layer).register_forward_hook(partial(log_internal, store = self.logs["neuron_activations"]))
      self.hooks["log_neuron_activations"].append(log_neuron_activations_hook)

  def reinit(self):
    # Remove all hooks
    self.unregister_all_hooks()

    # Clear all logs
    for log_name in self.logs:
      self.logs[log_name].clear()

    # Clear all intervention data
    self.register_neurons_intervention = None

  def finalize(self):
    # Initialize intervention hooks
    if self.register_neurons_intervention is not None:
      register_neurons = self.register_neurons_intervention["neurons_to_ablate"]
      for layer in register_neurons:
        intervene_register_neurons_fn = partial(
            activate_on_registers,
            num_registers=self.register_neurons_intervention["num_registers"],
            neuron_indices=register_neurons[layer],
            scale=self.register_neurons_intervention["scale"],
            normal_values=self.register_neurons_intervention["normal_values"],
            bottom_frac=self.register_neurons_intervention["bottom_frac"]
        )

        intervene_register_neurons_hook = self.neuron_activation_component(layer).register_forward_hook(intervene_register_neurons_fn)
        self.hooks["intervene_register_neurons"].append(intervene_register_neurons_hook)

    # Initialize log hooks
    if self.debug:
      self.initialize_log_hooks()

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


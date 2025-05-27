from abc import ABC, abstractmethod
import numpy as np

class HookManager(ABC):
  def __init__(self, model, debug = False):
    self.model = model
    self.debug = debug

    # Hooks
    self.layer_output_hooks = []
    self.neuron_hooks = []
    self.ablated_neuron_hooks = dict()
    self.ablated_neurons = dict()

    # Logs of internal model data
    self.attention_maps = []
    self.layer_outputs = []
    self.neuron_activations = []

  @abstractmethod
  def reinit(self):
    pass

  @abstractmethod
  def finalize(self):
    pass

  def get_attention_maps(self):
    if not self.attention_maps:
      return None
    return np.concatenate(self.attention_maps, axis=0)

  def get_layer_outputs(self):
    if not self.layer_outputs:
      return None
    return np.concatenate(self.layer_outputs, axis=0)

  def get_neuron_activations(self):
    if not self.neuron_activations:
      return None
    return np.concatenate(self.neuron_activations, axis=0)




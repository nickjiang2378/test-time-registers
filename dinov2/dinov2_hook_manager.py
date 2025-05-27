from shared.hook_manager import HookManager
from shared.hook_fn import log_internal, activate_on_registers
from functools import partial

class Dinov2HookManager(HookManager):
  def attn_post_softmax_component(self, layer):
    return self.model.blocks[layer].attn.identity

  def layer_output_component(self, layer):
    return self.model.blocks[layer]

  def neuron_activation_component(self, layer):
    return self.model.blocks[layer].mlp.act

  def num_layers(self):
    return len(self.model.blocks)
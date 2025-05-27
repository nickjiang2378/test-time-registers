from shared.hook_manager import HookManager # Need to add shared to sys path for this import to work
from functools import partial
from shared.hook_fn import log_internal

class ClipHookManager(HookManager):
  def attn_post_softmax_component(self, layer):
    return self.model.visual.transformer.resblocks[layer].attn.post_softmax_identity

  def layer_output_component(self, layer):
    return self.model.visual.transformer.resblocks[layer]

  def neuron_activation_component(self, layer):
    return self.model.visual.transformer.resblocks[layer].mlp.gelu

  def num_layers(self):
    return len(self.model.visual.transformer.resblocks)
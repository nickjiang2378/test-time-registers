from shared.hook_manager import HookManager
from dinov2.layers.swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused

class Dinov2HookManager(HookManager):
  def attn_output_component(self, layer):
    return self.model.blocks[layer].attn_out_identity

  def attn_pre_softmax_component(self, layer):
    return self.model.blocks[layer].attn.pre_softmax_identity

  def attn_post_softmax_component(self, layer):
    return self.model.blocks[layer].attn.post_softmax_identity

  def layer_output_component(self, layer):
    return self.model.blocks[layer]

  def neuron_activation_component(self, layer):
    if isinstance(self.model.blocks[layer].mlp, SwiGLUFFN) or isinstance(self.model.blocks[layer].mlp, SwiGLUFFNFused):
      return self.model.blocks[layer].mlp.hidden_identity
    else:
      return self.model.blocks[layer].mlp.act

  def num_layers(self):
    return len(self.model.blocks)
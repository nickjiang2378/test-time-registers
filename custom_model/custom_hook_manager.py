from shared.hook_manager import HookManager # Need to add "shared/" to sys path for this import to work

class CustomHookManager(HookManager):
  def attn_pre_softmax_component(self, layer):
    pass

  def attn_post_softmax_component(self, layer):
    pass

  def layer_output_component(self, layer):
    pass

  def neuron_activation_component(self, layer):
    pass

  def num_layers(self):
    pass
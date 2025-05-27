from shared.hook_manager import HookManager
from shared.hook_fn import log_internal, activate_on_registers
from functools import partial

class Dinov2HookManager(HookManager):
  def initialize_log_hooks(self):
    for layer in range(len(self.model.blocks)):
      # Attention maps
      log_attention_hook = self.model.blocks[layer].attn.identity.register_forward_hook(partial(log_internal, store = self.logs["attention_maps"]))
      self.hooks["log_attention_maps"].append(log_attention_hook)

      # Layer outputs
      log_layer_output_hook = self.model.blocks[layer].register_forward_hook(partial(log_internal, store = self.logs["layer_outputs"]))
      self.hooks["log_layer_outputs"].append(log_layer_output_hook)

      # Neuron activations
      log_neuron_activations_hook = self.model.blocks[layer].mlp.act.register_forward_hook(partial(log_internal, store = self.logs["neuron_activations"]))
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

        intervene_register_neurons_hook = self.model.blocks[layer].mlp.act.register_forward_hook(intervene_register_neurons_fn)
        self.hooks["intervene_register_neurons"].append(intervene_register_neurons_hook)

    # Initialize log hooks
    if self.debug:
      self.initialize_log_hooks()
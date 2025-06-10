import torch
from .utils import sign_max

def log_internal(module, input, output, store):
  store.append(output.detach().cpu().numpy())

def replace_internal(module, input, output, new_value):
  output[0] = new_value

def apply_func_on_internal(module, input, output, func):
  output[0] = func(output[0])

def activate_on_registers(module, input, output, num_registers, neuron_indices, scale = 1.0, normal_values = "zero"):
  # For all register neurons, set the activations of the extra registers to their max activation across patches
  if isinstance(scale, list):
    assert len(scale) == num_registers
    for i in range(num_registers):
      output[0, -num_registers + i, neuron_indices] = scale[i] * sign_max(output[0, :, neuron_indices])
  else:
    output[0, -num_registers:, neuron_indices] = scale * sign_max(output[0, :, neuron_indices]).unsqueeze(0).expand(num_registers, -1)
  if normal_values == "zero":
    # Set all image patch activations to 0
    output[0, 1:-num_registers, neuron_indices] = 0
  elif normal_values == "mean":
    # Set all image patch activations to the mean activation
    patch_activations = output[0, 1:-num_registers, neuron_indices]
    mean_activation = torch.mean(patch_activations, dim=0)  # Average across patches
    output[0, 1:-num_registers, neuron_indices] = mean_activation.unsqueeze(0).expand(output.shape[1] - num_registers - 1, -1)
  elif normal_values == "only_outliers":
    # Set only the outliers within the image patches to the mean activation
    patch_activations = output[0, 1:-num_registers, neuron_indices].clone()

    # Calculate threshold for outliers (1 std above mean activation)
    means = torch.mean(patch_activations, dim=0)
    stds = torch.std(patch_activations, dim=0)
    outlier_thresholds = means + stds

    # Replace only outliers with the mean activation
    mask = patch_activations > outlier_thresholds.unsqueeze(0)
    patch_activations[mask] = means.unsqueeze(0).expand_as(patch_activations)[mask]
    output[0, 1:-num_registers, neuron_indices] = patch_activations
  elif normal_values == "same":
    # Keep all the image patch activations the same
    pass
  else:
    raise ValueError(f"Invalid normal_values: {normal_values}")

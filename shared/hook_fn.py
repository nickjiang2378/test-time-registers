import torch

def log_internal(module, input, output, store):
  store.append(output.detach().cpu().numpy())

def replace_value(module, input, output, new_value):
  output[0] = new_value

def activate_on_registers(module, input, output, num_registers, neuron_indices, scale = 1.0, normal_values = "zero", bottom_frac = 0.75):
  # For all register neurons, set the activations of the extra registers to their max activation across patches
  output[0, -num_registers:, neuron_indices] = scale * torch.max(output[0, :, neuron_indices], dim = 0).values.unsqueeze(0).expand(num_registers, -1)
  if normal_values == "zero":
    output[0, 1:-num_registers, neuron_indices] = 0
  elif normal_values == "mean":
    # Take mean of bottom 75% of values
    patch_activations = output[0, 1:-num_registers, neuron_indices]
    sorted_activations, _ = torch.sort(patch_activations, dim=0)  # Sort along patch dimension
    cutoff_idx = int(bottom_frac * sorted_activations.shape[0])
    mean_low_activation = torch.mean(sorted_activations[:cutoff_idx, :], dim=0)  # Average across patches
    output[0, 1:-num_registers, neuron_indices] = mean_low_activation.unsqueeze(0).expand(output.shape[1] - num_registers - 1, -1)
  elif normal_values == "only_outliers":
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
    pass
  else:
    raise ValueError(f"Invalid normal_values: {normal_values}")

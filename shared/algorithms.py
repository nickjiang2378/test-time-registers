import torch
from tqdm import tqdm
from .utils import load_images

def find_register_neurons(
  model_state,
  image_path,
  register_norm_threshold: float = 30,
  detect_outliers_layer: int = -1,
  device: str = "cuda:0",
  processed_image_cnt: int = 500,
  apply_sparsity_filter: bool = False,
  sparsity_frac_threshold: float = 0.5,
  sparsity_activation_threshold: float = 0.5,
):
  model = model_state["model"]
  preprocess = model_state["preprocess"]
  run_model = model_state["run_model"]
  hook_manager = model_state["hook_manager"]

  num_layers = model_state["num_layers"]
  num_neurons = model_state["num_neurons_per_layer"]
  random_images = load_images(image_path, count=processed_image_cnt)
  neuron_scores = torch.zeros((len(random_images), num_layers, num_neurons), device=device)
  image_count = 0

  hook_manager.set_debug(True)

  for i in tqdm(range(len(random_images)), desc="Processing random images"):
    image = preprocess(random_images[i]).unsqueeze(0).to(device)
    hook_manager.reinit()
    hook_manager.finalize()

    representation = run_model(model, image)

    baseline_neuron_acts = torch.from_numpy(hook_manager.get_neuron_activations()).to(device)
    baseline_layer_outputs = torch.from_numpy(hook_manager.get_layer_outputs()).to(device)

    # Calculate norm map using torch
    norm_map = torch.norm(baseline_layer_outputs[detect_outliers_layer], dim=1)
    filtered_norms = norm_map.clone()
    filtered_norms[filtered_norms < register_norm_threshold] = 0

    # Get register locations as a tensor
    register_locations = torch.where(filtered_norms > register_norm_threshold)[0]

    if len(register_locations) == 0:
      continue

    image_count += 1

    # Process all layers vectorized
    for layer in range(num_layers):
      if apply_sparsity_filter:
        # Get activations for all neurons in this layer
        act_layer = baseline_neuron_acts[layer]  # Shape: [seq_len, num_neurons]

        # Check sparsity condition for all neurons at once
        sparse_neurons = torch.sum(act_layer < sparsity_activation_threshold, dim=0) >= sparsity_frac_threshold * act_layer.shape[0]  # Shape: [num_neurons]

        # Skip computation if no neurons meet the condition
        if not torch.any(sparse_neurons):
          continue

      # Get values at register locations for all neurons simultaneously
      # This creates a tensor of shape [num_register_locations, num_neurons]
      register_values = act_layer[register_locations]

      # For neurons that pass sparsity condition, compute mean at register locations
      # First, compute mean for all neurons (this is fast)
      neuron_means = register_values.mean(dim=0)  # Shape: [num_neurons]

      # Then zero out means for neurons that don't pass sparsity condition
      neuron_means = neuron_means * sparse_neurons.float()

      # Store in score tensor
      neuron_scores[i, layer] = neuron_means

  assert image_count > 0, "No images processed: either lower the register norm threshold or increase the number of processed images"

  # Rest of the code remains the same
  mean_neuron_scores = neuron_scores[:image_count].mean(dim=0)

  # Flatten and find top values
  flattened_scores = mean_neuron_scores.flatten()
  sorted_values, sorted_indices = torch.sort(flattened_scores, descending=True)

  # Convert indices to layer/neuron pairs
  top_indices = [(idx.item() // num_neurons, idx.item() % num_neurons) for idx in sorted_indices]

  register_norms = [
    (layer, neuron, sorted_values[i].item())
    for i, (layer, neuron) in enumerate(top_indices)
  ]

  return register_norms
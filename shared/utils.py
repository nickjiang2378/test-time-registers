import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

def load_images(imagenet_path, count = 10, images_only = True):
  # Sample random images from imagenet
  file_list = []

  for root, dirs, files in os.walk(imagenet_path):
      for file in files:
          file_list.append(os.path.join(root, file))

  if count > len(file_list):
    sampled_files = file_list
  else:
    sampled_files = random.sample(file_list, count)

  image_files = []

  for filename in sampled_files:
    image_files.append(Image.open(filename))
  print("Loaded {} images".format(len(image_files)))

  if images_only:
    return image_files
  else:
    return image_files, sampled_files

def plot_images_with_max_per_row(images, max_per_row = 4, image_title = "Layer"):
  # Calculate number of rows needed (4 images per row)
  num_images = len(images)
  num_rows = (num_images + max_per_row - 1) // max_per_row  # Ceiling division

  # Create figure with appropriate size
  fig, axes = plt.subplots(num_rows, max_per_row, figsize=(20, 5*num_rows))
  axes = axes.flatten()  # Flatten for easier indexing

  # Plot each layer's patch norms
  for i in range(num_images):
    im = axes[i].imshow(images[i], cmap='viridis')
    axes[i].set_title(f'{image_title} {i}', fontsize=16)  # Increased font size
    axes[i].axis('off')
    # Add colorbar with same height as image
    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

  # Hide any unused subplots
  for i in range(num_images, len(axes)):
    axes[i].axis('off')

  return plt

def plot_attn_maps(attn_maps):
  num_layers, num_heads, patch_height, patch_width = attn_maps.shape

  # Create a grid of plots for all layers and heads
  fig, axes = plt.subplots(num_layers, num_heads, figsize=(2*num_heads, 2*num_layers))
  fig.suptitle(f'Attention Maps', fontsize=16)

  # Import the correct module for make_axes_locatable
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  # Plot each layer-head combination
  for layer in range(num_layers):
      # Determine min and max for this layer for consistent colorbar scaling within the layer
      layer_vmin = attn_maps[layer].min().item()
      layer_vmax = attn_maps[layer].max().item()

      for head in range(num_heads):
          # Get the current axis (handle both 2D and 1D cases)
          if num_layers == 1 and num_heads == 1:
              ax = axes
          elif num_layers == 1:
              ax = axes[head]
          elif num_heads == 1:
              ax = axes[layer]
          else:
              ax = axes[layer, head]

          # Plot the attention shift map with layer-specific normalization
          im = ax.imshow(attn_maps[layer, head], cmap='viridis', vmin=layer_vmin, vmax=layer_vmax)

          # Remove ticks for cleaner appearance
          ax.set_xticks([])
          ax.set_yticks([])

          # Add layer and head labels only on the edges
          if head == 0:
              ax.set_ylabel(f'Layer {layer}')
          if layer == num_layers-1:
              ax.set_xlabel(f'Head {head}')

          # Add a colorbar for each layer (only once per row)
          if head == num_heads-1:
              # Create a colorbar that's properly sized relative to the plot
              divider = make_axes_locatable(ax)
              cax = divider.append_axes("right", size="5%", pad=0.05)
              plt.colorbar(im, cax=cax)

  # Adjust layout to make room for the colorbars
  plt.tight_layout()
  return plt

def filter_highest_layer(register_norms, highest_layer):
  return [norm for norm in register_norms if norm[0] <= highest_layer]

def filter_lowest_layer(register_norms, lowest_layer):
  return [norm for norm in register_norms if norm[0] >= lowest_layer]

def filter_layers(register_norms, highest_layer = -1, lowest_layer = 0):
  if highest_layer == -1:
    return [norm for norm in register_norms if norm[0] >= lowest_layer]
  else:
    return [norm for norm in register_norms if norm[0] >= lowest_layer and norm[0] <= highest_layer]

def sign_max(tensor):
   pos_max = torch.max(tensor)
   neg_max = torch.min(tensor)
   return pos_max if abs(pos_max) > abs(neg_max) else neg_max
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sys
import os 
sys.path.append(os.path.abspath(os.path.join("", "..")))
from dinov2.models.vision_transformer import *




class DinoVisionTransformerWithIntervention(torch.nn.Module):
    """
    A wrapper around DinoVisionTransformer that enables neuron intervention
    by directly modifying the original model's behavior through hooks.
    """
    def __init__(self, original_model, neurons_to_ablate, num_register_tokens):
        super().__init__()
        
        # Store the original model
        self.model = original_model
        
        # Set the register tokens for the model
        self.model.num_register_tokens = num_register_tokens
        
        # Store intervention parameters
        self.neurons_to_ablate = neurons_to_ablate
        self.device = next(original_model.parameters()).device
        
        # Initialize hooks list
        self.hooks = []
        
        # Set up the hooks for each block
        self._setup_intervention_hooks()
    
    def _intervention_hook(self, module, input, output):
        """Hook function that modifies neuron activations"""
        layer_idx = getattr(module, "_layer_idx", None)
        if layer_idx is not None and layer_idx in self.neurons_to_ablate:
            neurons = self.neurons_to_ablate[layer_idx]
            
            # Create new activation map for specified neurons
            new_activation_map = torch.zeros(
                (output.shape[0], output.shape[1], len(neurons)),
                device=output.device
            )
            
            # Compute max values for these neurons
            max_values = torch.max(output[:, :, neurons], dim=1, keepdim=True).values
            
            # Fill the register token positions with max values
            if self.model.num_register_tokens > 0:
                new_activation_map[:, -self.model.num_register_tokens:] = max_values
            
            # Modify only the specified neurons
            output[:, :, neurons] = new_activation_map
        
        return output
    
    def _setup_intervention_hooks(self):
        """Set up hooks for all blocks in the model"""
        # Remove any existing hooks
        self._remove_hooks()
        
        # Add new hooks
        for i, block in enumerate(self.model.blocks):
            # Handle either chunked or non-chunked blocks
            if hasattr(block, '__iter__'):  # It's a list/sequence of blocks
                for j, sub_block in enumerate(block):
                    if hasattr(sub_block.mlp, 'act'):
                        # Store layer index as attribute on the module
                        sub_block.mlp.act._layer_idx = i * len(block) + j
                        # Register the hook
                        hook = sub_block.mlp.act.register_forward_hook(self._intervention_hook)
                        self.hooks.append(hook)
            else:  # It's a single block
                if hasattr(block.mlp, 'act'):
                    # Store layer index as attribute on the module
                    block.mlp.act._layer_idx = i
                    # Register the hook
                    hook = block.mlp.act.register_forward_hook(self._intervention_hook)
                    self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def forward(self, x, is_training = False):
        """Forward pass through the model"""
        return self.model(x, is_training=is_training)
    
    def __del__(self):
        """Clean up hooks when object is deleted"""
        self._remove_hooks()






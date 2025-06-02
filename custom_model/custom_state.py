import os
import torch
from custom_hook_manager import CustomHookManager

def run_model(model, image, num_registers = None):
  pass

def preprocess(image):
  pass

def get_num_neurons_per_mlp(model):
  pass

def get_num_layers(model):
  pass

def get_num_heads(model):
  pass

def load_custom_state(config):

  model = ...

  hook_manager = CustomHookManager(model)

  return dict(
    config=config,
    model=model,
    num_heads=get_num_heads(model),
    num_layers=get_num_layers(model),
    num_neurons_per_layer=get_num_neurons_per_mlp(model),
    patch_size=...,
    run_model=run_model,
    preprocess=preprocess,
    hook_manager=hook_manager,
  )
import os
import torch
from dinov2.data.transforms import make_classification_eval_transform
from dinov2_hook_manager import Dinov2HookManager
from dinov2.layers.swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused

def run_model(model, image, num_registers = None):
  if num_registers is not None:
    prev_num_registers = model.num_register_tokens
    model.num_register_tokens = num_registers

  with torch.no_grad():
    representation = model(image)

  if num_registers is not None:
    model.num_register_tokens = prev_num_registers
  return representation

def preprocess(image):
  scale_factor = 1
  rescaled_image = image.resize((scale_factor * image.width, scale_factor * image.height))
  rescaled_image = rescaled_image.convert('RGB')
  classification_transform = make_classification_eval_transform()
  transformed_image = classification_transform(rescaled_image)
  return transformed_image

def get_num_neurons_per_mlp(backbone_model):
  if isinstance(backbone_model.blocks[0].mlp, SwiGLUFFN) or isinstance(backbone_model.blocks[0].mlp, SwiGLUFFNFused):
    return backbone_model.blocks[0].mlp.w3.in_features
  else:
    return backbone_model.blocks[0].mlp.fc1.out_features

def get_num_layers(backbone_model):
  return len(backbone_model.blocks)

def get_num_heads(backbone_model):
  return backbone_model.num_heads

def load_dinov2_state(config):
  os.environ["XFORMERS_DISABLED"] = "1" # Disable xformers for easier access to attention maps

  backbone_size = config["backbone_size"]

  backbone_arch = backbone_size
  backbone_name = f"dinov2_{backbone_arch}"

  backbone_model = torch.hub.load(repo_or_dir="dinov2/", model=backbone_name, source="local") # ensure that the repo path points to the `dinov2` directory in repo
  backbone_model.eval()
  backbone_model.cuda()

  hook_manager = Dinov2HookManager(backbone_model)

  return dict(
    config=config,
    model=backbone_model,
    num_heads=get_num_heads(backbone_model),
    num_layers=get_num_layers(backbone_model),
    num_neurons_per_layer=get_num_neurons_per_mlp(backbone_model),
    patch_size=backbone_model.patch_size,
    run_model=run_model,
    preprocess=preprocess,
    hook_manager=hook_manager,
  )
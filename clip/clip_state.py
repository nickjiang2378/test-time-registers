from clip_hook_manager import ClipHookManager
from clip.factory import create_model_and_transforms, get_tokenizer
import torch

def run_model(model, image, num_registers = None):
  if num_registers is not None:
    with torch.no_grad():
      representation = model.encode_image(
        image, attn_method="direct", normalize=False, extra_tokens = num_registers
      )
  else:
    with torch.no_grad():
      representation = model.encode_image(
        image, attn_method="direct", normalize=False
      )
  return representation

def get_num_neurons_per_mlp(model):
  return model.visual.transformer.resblocks[0].mlp.c_fc.out_features

def get_num_layers(model):
  return len(model.visual.transformer.resblocks)

def get_num_heads(model):
  return model.visual.transformer.resblocks[0].attn.num_heads

def load_clip_state(config):
  model_name = config["model_name"]
  pretrained = config["pretrained"]
  device = config["device"]

  tokenizer = get_tokenizer(model_name)
  model, _, preprocess = create_model_and_transforms(
      model_name, pretrained=pretrained, force_quick_gelu=True
  )
  model.to(device)
  model.eval()
  num_heads = get_num_heads(model)
  num_layers = get_num_layers(model)
  num_neurons_per_layer = get_num_neurons_per_mlp(model)

  hook_manager = ClipHookManager(model)

  return dict(
    config=config,
    tokenizer=tokenizer,
    model=model,
    preprocess=preprocess,
    num_heads=num_heads,
    num_layers=num_layers,
    num_neurons_per_layer=num_neurons_per_layer,
    patch_size=model.visual.patch_size[0],
    run_model=run_model,
    hook_manager=hook_manager,
  )
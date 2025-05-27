# from clip_hook_manager import ClipHookManager
from clip.factory import create_model_and_transforms, get_tokenizer
import torch

def run_model(model, image, num_registers = 0):
  with torch.no_grad():
    representation = model.encode_image(
      image, attn_method="head", normalize=False, extra_tokens = num_registers
    )
  return representation

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
  num_heads = model.visual.transformer.resblocks[0].attn.num_heads
  num_layers = len(model.visual.transformer.resblocks)

  # hook_manager = ClipHookManager(model)

  return dict(
    config=config,
    tokenizer=tokenizer,
    model=model,
    preprocess=preprocess,
    num_heads=num_heads,
    num_layers=num_layers,
    run_model=run_model,
    # hook_manager=hook_manager,
  )
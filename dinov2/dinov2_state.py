import os
import torch
from dinov2.data.transforms import make_classification_eval_transform

backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "large_reg": "vitl14_reg",
    "giant": "vitg14",
}

def run_model(model, image, num_registers = 0):
  prev_num_registers = model.num_register_tokens
  model.num_register_tokens = num_registers
  with torch.no_grad():
    representation = model(image)
  model.num_register_tokens = prev_num_registers
  return representation

def preprocess(image):
  scale_factor = 1
  rescaled_image = image.resize((scale_factor * image.width, scale_factor * image.height))
  classification_transform = make_classification_eval_transform()
  transformed_image = classification_transform(rescaled_image)
  return transformed_image

def load_dinov2_state(config):
  os.environ["XFORMERS_DISABLED"] = "1"

  backbone_size = config["backbone_size"]

  backbone_arch = backbone_archs[backbone_size]
  backbone_name = f"dinov2_{backbone_arch}"

  backbone_model = torch.hub.load(repo_or_dir="dinov2/", model=backbone_name, source="local") # ensure that the repo path points to the `dinov2` directory in repo
  backbone_model.eval()
  backbone_model.cuda()

  return dict(
    config=config,
    model=backbone_model,
    run_model=run_model,
    preprocess=preprocess,
  )
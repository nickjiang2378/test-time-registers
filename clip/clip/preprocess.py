## Imports
import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path
import cv2
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
import einops
import plotly.express as px
import torch.nn.functional as F
import tqdm
import json
import albumentations
import glob
from torchvision import transforms


def _convert_to_rgb(image):
    return image.convert('RGB')

def _resize(image):
    image = np.array(image)
    image = albumentations.augmentations.geometric.resize.LongestMaxSize(interpolation=Image.BICUBIC, 
                                                                        max_size=224)(image=image)
    return Image.fromarray(image['image'])

preprocess = transforms.Compose([
    _resize,
    transforms.CenterCrop(size=(224, 224)),
    _convert_to_rgb,
])


both_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                         std=(0.26862954, 0.26130258, 0.27577711)),
])
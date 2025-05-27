import os
import random
from PIL import Image

def load_images(imagenet_path, count = 10, images_only = True):
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

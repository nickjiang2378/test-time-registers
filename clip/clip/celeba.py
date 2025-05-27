import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Text

# adapted from https://github.com/umd-huang-lab/perceptionCLIP/blob/main/src/datasets/celeba.py


class CelebADataset(Dataset):
    _normalization_stats = {"mean": OPENAI_DATASET_MEAN, "std": OPENAI_DATASET_STD}

    def __init__(
        self,
        root_dir,
        target_name: Text = "Bald", # "Blond_Hair",
        attribute_name: Text = "Young", # "Male",
        split: Text = "train",
        transform=None,
    ):
        self.root_dir = root_dir
        self.target_name = target_name
        self.attribute_name = attribute_name
        # Only support 1 attribute for now as in official benchmark
        self.split = split

        self.split_dict = {"train": 0, "val": 1, "test": 2}

        self.data_dir = self.root_dir

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset first."
            )

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, "list_attr_celeba.csv")
        )
        self.split_df = pd.read_csv(
            os.path.join(self.data_dir, "list_eval_partition.csv")
        )
        # Filter for data split ('train', 'val', 'test')
        self.metadata_df["partition"] = self.split_df["partition"]
        self.metadata_df = self.metadata_df[
            self.split_df["partition"] == self.split_dict[self.split]
        ]

        # Get the y values and attribute values
        self.y_array = self.metadata_df[self.target_name].values
        self.attribute_array = self.metadata_df[attribute_name].values
        self.y_array[self.y_array == -1] = 0
        self.attribute_array[self.attribute_array == -1] = 0
        self.n_classes = len(np.unique(self.y_array))
        self.n_attributes = len(np.unique(self.attribute_array))

        # Extract filenames and splits
        self.filename_array = self.metadata_df["image_id"].values
        self.split_array = self.metadata_df["partition"].values

        self.targets = torch.tensor(self.y_array)
        self.attributes = torch.tensor(self.attribute_array)

        # Image transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transform_celeba()

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.targets[idx]
        a = self.attributes[idx]
        img_filename = os.path.join(
            self.data_dir, "img_align_celeba", self.filename_array[idx]
        )
        img = Image.open(img_filename).convert("RGB")
        img = self.transform(img)

        return img, (y, a)


def get_transform_celeba():
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    transform = transforms.Compose(
        [
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CelebADataset._normalization_stats["mean"],
                std=CelebADataset._normalization_stats["std"],
            ),
        ]
    )
    return transform


# classnames = ["dark hair", "blond hair"]
classnames = ["hair", "bald"]
# attrnames = ["female", "male"]
attrnames = ["old", "young"]

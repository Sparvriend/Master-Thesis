import os
import pandas as pd
import random
import torch
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset


class ProjDataset(Dataset):
    """ProjDataset class, which serves as a parent to any dataset
    class defined for this project. The child classes inherit the common
    arguments as well as the __len__ and __getitem__ functions.
    """
    def __init__(self, data_path: str, transform: T.Compose):
        self.img_paths = []
        self.img_labels = []
        self.data_path = data_path
        self.data_type = os.path.normpath(self.data_path).split(os.sep)[2]
        self.transform = transform

    # Function to return the length of the dataset
    def __len__(self) -> int:
        return len(self.img_paths)
    
    # Function to return attributes per item in the dataset
    # The sep_collate function in train.py ensures that for batches,
    # only the label and images are returned.
    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, int]:
        path = self.img_paths[idx]
        raw_image = Image.open(path)
        # Converting to RGB if image is grayscale
        if raw_image.mode == "L":
            raw_image = raw_image.convert("RGB")

        # Transforming the image, it is augmented if from the training dataset
        image = self.transform(raw_image)
        if self.data_type == "train" or self.data_type == "val":
            # Saving the label if it exists.
            label = self.img_labels[idx]
        else:
            label = None
        return path, image, label


class NTZFilterDataset(ProjDataset):
    """NTZFilterDataset class, to use for any dataset formed out of NTZ filter
    images. The class has different possibilities depending on if the type is
    training, validation or testing data.
    """
    def __init__(self, data_path: str, transform: T.Compose):
        super().__init__(data_path, transform)
        self.label_map = {0: "fail_label_crooked_print",
                          1: "fail_label_half_printed",
                          2: "fail_label_not_fully_printed",
                          3: "no_fail"}
        self.n_classes = len(self.label_map)	

        # Setting the paths for each image and a label if it concerns training
        # or validation data, labels are enumerated over
        for label, dir_name in enumerate(os.listdir(self.data_path)):
            files = os.listdir(os.path.join(self.data_path, dir_name))
            for file_name in files:
                self.img_paths.append(os.path.join(self.data_path, 
                                                   dir_name, file_name))
                if self.data_type == "train" or self.data_type == "val":
                    self.img_labels.append(label)


class NTZFilterSyntheticDataset(NTZFilterDataset):
    """NTZFilterSyntheticDataset class, to use for any dataset formed out of
    partial synthetic data. In terms of setup it is exactly the same as the 
    NTZFilterdataset."""
    def __init__(self, data_path: str, transform: T.Compose):
        super().__init__(data_path, transform)


class CIFAR10Dataset(ProjDataset):
    """CIFAR10Dataset class, to use for any dataset formed out of CIFAR images.
    Dataset taken from:
    https://www.kaggle.com/competitions/cifar-10/data?select=train.7z
    With minimal adjustements, located in data_processing.py"""
    def __init__(self, data_path: str, transform: T.Compose):
        super().__init__(data_path, transform)

        # Setting the proportion of the dataset to use
        # Test set is never used so this reduces time
        if self.data_type == "train" or self.data_type == "val":
            self.prop = 1
        elif self.data_type == "test":
            self.prop = 0.01

        # Reading training labels and encoding to integers
        label_df = pd.read_csv(os.path.join("data", "CIFAR10", "trainLabels.csv"))
        unqiue_labels = label_df["label"].unique().tolist()
        self.label_map = {label: i for i, label in enumerate(unqiue_labels)}
        self.n_classes = len(self.label_map)

        # Setting image paths and training labels
        # Randomly sampling like this does not ensure an equal amount of images
        # If self.prop = 1, then this randomly shuffles the data
        files = os.listdir(self.data_path)
        files = random.sample(files, int(len(files) * self.prop))
        for file in files:
            self.img_paths.append(os.path.join(self.data_path, file))
            if self.data_type == "train" or self.data_type == "val":
                # Cutting off png, retrieving the label and appending encoded version
                file = int(file.removesuffix(".png"))
                label_string = label_df.loc[label_df["id"] == file, "label"].values[0]
                self.img_labels.append(self.label_map[label_string])
        
        # Swapping the label map, since it needs to be saved in the opposite way
        # Not {"frog": 0}, but {0: "frog"} for usage in the code.
        self.label_map = {v: k for k, v in self.label_map.items()}
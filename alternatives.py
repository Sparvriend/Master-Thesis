import numpy as np
from PIL import Image
import torch

"""This file contains alternative classes/functions that are not used but
might be used later on. The purpose of this file is to save their existence.
"""


class Standardize:
    """This is a class that allows for a standardization transform to be
    combined with other transforms in a torchvision Compose element.
    Instead of doing transform, resize, crop, ToTensor and normalize,
    this class gives the option of doing the last two steps by hand.
    To use it, ToTensor() and Normalize() can be replaced by
    Standardize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).
    """
    def __init__(self, mean: list, std: list):
        self.mean = mean
        self.std = std

    # 0-255 range image is first converted to float, then divided by 255.
    # This is done so that the normalization will work. The normalization
    # is applied per channel.
    def __call__(self, img: Image) -> torch.Tensor:
        img = (np.array(img)).astype(np.float32) / 255.0
        # Subtracting mean and dividing by std over each channel
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]
        # Converting to a torch tensor
        # from_numpy is expecting the input to be (channels, height, width)
        # instead of (height, width, channels), so the axes are swapped.
        img = np.swapaxes(img, 0, 2)
        return torch.from_numpy(img)
import torchvision.transforms as T
from imagecorruptions import corrupt
import torch
import os
import random
from PIL import Image
import numpy as np
import warnings

# This is a class that allows for the corrupt function to be joined in a list format with the Pytorch transforms
class CustomCorruption:
    def __init__(self, corruption_name):
        self.corruption_name = corruption_name

    def __call__(self, img):
        # Convert to numpy ndarray since that is required for the corrupt function from the imagecorruption library
        if type(img) != np.ndarray:
            img = np.array(img)
        # Ignoring the futureWarning, since I can not do anything about it
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            img = corrupt(img, corruption_name=self.corruption_name)
        # Converting back to a PIL image and returning
        return Image.fromarray(img)

# Function that converts labels to a list and then saves paths and labels to appropriate prediction directories
def save_test_predicts(predicted_labels, paths):
    prediction_list = []

    # Flattening both path and label lists, also converting predictions from tensors to lists
    for tensor in predicted_labels:
        prediction_list.append(tensor.tolist())
    prediction_list = flatten_list(prediction_list)
    paths = flatten_list(paths)

    # Dictionary for the labels to use in saving
    label_dict = {0: "fail_label_crooked_print", 1: "fail_label_half_printed", 2: "fail_label_not_fully_printed", 3: "no_fail"}	
    prediction_dir = "data/test_predictions"

    # Saving each image to the predicted folder
    for idx, path in enumerate(paths):
        name = os.path.normpath(path).split(os.sep)[-1]
        img = Image.open(path)
        img.save(os.path.join(prediction_dir, label_dict[prediction_list[idx]], name)) 

def flatten_list(list):
    return [item for sublist in list for item in sublist]

# Manual replacement of default collate function provided by PyTorch
# The function removes the augmentation and the path that is normally returned by using __getitem__ as well as transforming to lists of tensors
def sep_collate(batch):
    _, images, labels = zip(*batch)
    tensor_labels = []

    # Labels are ints, which is why they need to be converted to tensors before being entered into a torch stack
    for label in labels:
        tensor_labels.append(torch.tensor(label))

    # Converting both images and label lists of tensors to torch stacks
    images = torch.stack(list(images), dim = 0)
    labels = torch.stack(list(tensor_labels), dim = 0)

    return images, labels

# Manual collate function for testing dataloader
def sep_test_collate(batch):
    path, images, _ = zip(*batch)

    # Converting image lists of tensors to torch stack
    images = torch.stack(list(images), dim = 0)

    return images, path

# This function splits up augmentation into four phases, each with a different general augmentation techniques
# The idea behind it is to only select one augmentation from each phase, to ensure that the images are still classifiable after augmentation
def get_categorical_transforms():
    # Augmentations phase 1 (moving the image around):
    transforms_phase_1 = [T.RandomRotation(degrees = (0, 360)), T.RandomAffine(degrees = 0, translate = (0.1, 0.3), scale = (0.75, 1.0), shear = (0, 0.2)),
                          T.RandomHorizontalFlip(p = 1.0), T.RandomVerticalFlip(p = 1.0)]
    
    # Augmentations phase 2 (Simple color changes)
    transforms_phase_2 = [T.Grayscale(num_output_channels = 3), T.RandomAdjustSharpness(sharpness_factor = 2, p = 1.0),
                          T.RandomAutocontrast(p = 1.0), T.RandomEqualize(p = 1.0)]

    # Augmentations phase 3 (Advanced color changes)
    transforms_phase_3 = [T.ColorJitter(brightness = (0.3, 1), contrast = (0.3, 1), saturation = (0.3, 1), hue = (-0.5, 0.5)),
                          T.RandomInvert(p = 1.0), T.RandomPosterize(3, p = 1.0), T.RandomSolarize(threshold = random.randint(100, 200), p=0.5)]

    # Augmentations phase 4 (Adding noise) -> from the imagecorruption library https://github.com/bethgelab/imagecorruptions:
    transforms_phase_4 = [CustomCorruption(corruption_name = "gaussian_blur"), CustomCorruption(corruption_name = "shot_noise"),
                          CustomCorruption(corruption_name = "impulse_noise"), CustomCorruption(corruption_name = "motion_blur"),
                          CustomCorruption(corruption_name = "zoom_blur"), CustomCorruption(corruption_name = "pixelate"),
                          CustomCorruption(corruption_name = "jpeg_compression")]

    combined_transforms = transforms_phase_1 + transforms_phase_2 + transforms_phase_3 + transforms_phase_4
    categorical_transforms = T.Compose([T.RandomChoice(transforms_phase_1), T.RandomChoice(transforms_phase_2), 
                                        T.RandomChoice(transforms_phase_3), T.RandomChoice(transforms_phase_4)])

    return combined_transforms, categorical_transforms

def get_transforms(transform_type = "categorical"):
    # Pytorch augmentations listed and explained:
    # The augmentations should only be randomized by RandomApply/RandomChoice, not by any random probability in the functions themselves.
    # Any randomization of the function should indicate a randomization of the input parameters, not if it is applied or not.
    # Standard transforms:
    # - T.RandomRotation(degrees = (0, 360)) - Randomly rotates the image by a random angle between 0 and 360 degrees
    # - T.GrayScale(num_output_channels = 3) - Converts the image to grayscale with three output channels
    # - T.ColorJitter(brightness = (0.3, 1), contrast = (0.3, 1), saturation = (0.3, 1), hue = (-0.5, 0.5)) - Randomly changes the brightness, contrast, saturation and hue of the image
    # - T.RandomAffine(degrees = 90, translate = (0.1, 0.3), scale = (0.5, 1.0), shear = (0, 0.2)) - Randomly rotates, translates, scales and shears the image
    # - T.RandomHorizontalFlip(p = 1.0) - Flips the image horizontally
    # - T.RandomVerticalFlip(p = 1.0) - Flips the image vertically
    # - T.GaussianBlur(kernel_size = (3, 3)) - Applies Gaussian Blur to the image
    # - T.RandomInvert(p = 1.0) - Inverts the colors of the image
    # - T.RandomPosterize(3, p=1.0) - Posterizes the image, by reducing it from 8 to 3 channels.
    # - T.RandomSolarize(threshold = random.randint(100, 200), p=0.5) - Solarizes the image, by inverting all pixels above a random threshold
    # - T.RandomAdjustSharpness(sharpness_factor = 2, p = 1.0) - Sharpens the image, since GaussianBlur already blurs
    # - T.RandomAutocontrast(p = 1.0) - Applies autocontrast to the image, makes light pixels lighter and dark pixels darker
    # - T.RandomEqualize(p = 1.0) - Equalizes the image, modifies intensity distribution of the image

    combined_transforms, categorical_transforms = get_categorical_transforms()
    transform_options = {"random_choice": T.RandomChoice(combined_transforms), "categorical": categorical_transforms,
                         "auto_augment": T.AutoAugment(policy = T.AutoAugmentPolicy.IMAGENET), "rand_augment": T.RandAugment()}

    transform = T.Compose([
        transform_options[transform_type],
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    return transform
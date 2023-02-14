import torchvision.transforms as T
import torch
import os
import random

TEST_PATH_NO_LABEL = "data/test_no_label"

# Function that converts labels to a list and then saves paths and labels to appropriate prediction directories
def save_test_predicts(predicted_labels, paths):
    # Converting predicted labels to a list and converting it from a list of lists to a flattened list, similarly for the paths lists of lists
    prediction_list = []

    for tensor in predicted_labels:
        prediction_list.append(tensor.tolist())
    prediction_list = flatten_list(prediction_list)
    paths = flatten_list(paths)

    # TODO: Save the images to their predicted folders
    print("Construction in progress")

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

def get_transforms():
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
    # - T.RandomAdjustSharpness(sharpness_factor = 0, p = 1.0) - Sharpens the image, since GaussianBlur already blurs
    # - T.RandomAutocontrast(p = 1.0) - Applies autocontrast to the image, makes light pixels lighter and dark pixels darker
    # - T.RandomEqualize(p = 1.0) - Equalizes the image, modifies intensity distribution of the image

    # TODO: Randomize input for GaussianBlur and Solarize/Posterize for each image
    # -> T.RandomSolarize(threshold = random.randint(100, 200), p=0.5) 
    # -> This only randomizes it per DataLoader, not per image
    # TODO: Change RandomApply procedure, since it turns the images into unclassifiable versions, alternative is using RandomChoice
    # -> Turn into handcrafted algorithm which applies a few transforms on top of each other but keeps the image classifiable
    # TODO: Look into automatic augmentation transforms (https://pytorch.org/vision/main/transforms.html)

    random_transforms = [T.RandomRotation(degrees = (0, 360)), T.Grayscale(num_output_channels = 3), 
                                         T.ColorJitter(brightness = (0.3, 1), contrast = (0.3, 1), saturation = (0.3, 1), hue = (-0.5, 0.5)),
                                         T.RandomAffine(degrees = 90, translate = (0.1, 0.3), scale = (0.5, 1.0), shear = (0, 0.2)),
                                         T.RandomHorizontalFlip(p = 1.0), T.RandomVerticalFlip(p = 1.0), T.GaussianBlur(kernel_size = (3, 3)),
                                         T.RandomInvert(p = 1.0), T.RandomSolarize(threshold = random.randint(100, 200), p=0.5),
                                         T.RandomPosterize(3, p=1.0), T.RandomAdjustSharpness(sharpness_factor = 2, p = 1.0),
                                         T.RandomAutocontrast(p = 1.0), T.RandomEqualize(p = 1.0)]

    # Defining transforms for training data, based on information from https://pytorch.org/hub pytorch_vision_mobilenet_v2/
    transform = T.Compose([
        #T.RandomApply(random_transforms, p = 0.8),
        T.RandomChoice(random_transforms),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    return transform
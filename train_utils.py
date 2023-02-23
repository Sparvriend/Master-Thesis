import datetime
import numpy as np
import json
import os
from PIL import Image
import random
import torch
from torch import nn, optim
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

from imagecorruptions import corrupt
import warnings

class CustomCorruption:
    """This is a class that allows for the corrupt function to be joined in a
    list format with the Pytorch transforms.
    """
    def __init__(self, corruption_name: str):
        self.corruption_name = corruption_name

    def __call__(self, img: Image) -> Image:
        # Convert to numpy ndarray since that is required for the corrupt
        # function from the imagecorruption library
        if type(img) != np.ndarray:
            img = np.array(img)
        # Ignoring the futureWarning, since I can not do anything about it
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            img = corrupt(img, corruption_name=self.corruption_name)
        # Converting back to a PIL image and returning
        return Image.fromarray(img)


def save_test_predicts(predicted_labels: list, paths: list):
    """Function that converts labels to a list and then saves paths and labels
    to appropriate prediction directories.

    Args:
        predicted_labels: list of tensors with predicted labels.
        paths: list of lists with paths (strings) to images.
    """
    # Flattening both path and label lists, 
    # also converting predictions from tensors to lists
    prediction_list = []
    for tensor in predicted_labels:
        prediction_list.append(tensor.tolist())
    prediction_list = flatten_list(prediction_list)
    paths = flatten_list(paths)

    # Dictionary for the labels to use in saving
    label_dict = {0: "fail_label_crooked_print", 1: "fail_label_half_printed",
                  2: "fail_label_not_fully_printed", 3: "no_fail"}	
    prediction_dir = "data/test_predictions"

    # Saving each image to the predicted folder
    for idx, path in enumerate(paths):
        name = os.path.normpath(path).split(os.sep)[-1]
        img = Image.open(path)
        img.save(os.path.join(prediction_dir, label_dict[prediction_list[idx]], name))


def flatten_list(list: list) -> list:
    """Function that takes a list of lists and flattens it into
    a single list.

    Args:
        list: list of lists.
    Returns:
        Flattened list.
    """
    return [item for sublist in list for item in sublist]


def sep_collate(batch: list) -> tuple[torch.stack, torch.stack]:
    """Manual replacement of default collate function provided by PyTorch.
    The function removes the augmentation and the path that is normally
    returned by using __getitem__ as well as transforming to lists of tensors.
    
    Args:
        batch: batch of data items from a dataloader.
    Returns:
        images and labels as torch stacks.
    """
    # Labels are ints, which is why they need to be converted to tensors
    # before being entered into a torch stack
    _, images, labels = zip(*batch)
    tensor_labels = []
    for label in labels:
        tensor_labels.append(torch.tensor(label))

    # Converting both images and label lists of tensors to torch stacks
    images = torch.stack(list(images), dim = 0)
    labels = torch.stack(list(tensor_labels), dim = 0)

    return images, labels


def sep_test_collate(batch: list) -> tuple[torch.stack, list]:
    """ Manual collate function for testing dataloader.
    It converts the images to a torch stack and returns the paths.

    Args:
        batch: batch of data items from a dataloader.
    Returns:
        images as torch stack and paths.
    """
    path, images, _ = zip(*batch)

    images = torch.stack(list(images), dim = 0)

    return images, path


def setup_tensorboard(experiment_name: str) -> tuple[list[SummaryWriter], str]:
    """Function that provides tensorboard writers for training and validation.
    
    Args:
        experiment_name: Name of the experiment that is run.
    Returns:
        List of tensorboard writers.
    """
    
    current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
    experiment_path = os.path.join("Master-Thesis-Experiments", (experiment_name
                                                                 + current_time)) 
    train_dir = os.path.join(experiment_path, "train")
    val_dir = os.path.join(experiment_path, "val")
    hyp_dir = os.path.join(experiment_path, "hyp")
    train_writer = SummaryWriter(train_dir)
    validation_writer = SummaryWriter(val_dir)
    hyp_writer = SummaryWriter(hyp_dir)

    return {"train": train_writer, "val": validation_writer, "hyp": hyp_writer}, experiment_path


def setup_hyp_file(writer: SummaryWriter, hyp_dict: dict):
    """Function that writes all hyperparameters for a run to the tensorboard
    text plugin.

    Args:
        writer: Tensorboard writer for writing text.
        hyp_dict: Dictionary with hyperparameters.
    """
    for key, value in hyp_dict.items():
        writer.add_text(key, str(value))


def setup_hyp_dict(experiment_name: str) -> dict:
    """This function retrieves the hyperparameters from the JSON file and
    sets them up in a dictionary from which the hyperparameters can be used.

    Args:
        experiment_name: Name of the experiment that is run.
    Returns:
        Dictionary with hyperparameters.
    """
    experiment_location = os.path.join("Master-Thesis-Experiments",
                                       (experiment_name + ".json"))
    with open(experiment_location, "r") as f:
        hyp_dict = json.load(f)

    # The model already has to be defined, since the optimizer requires it
    model = eval(hyp_dict["Model"])

    # Evaluating the string expressions in the JSON experiment setup file
    for key, value in hyp_dict.items():
        try:
            hyp_dict[key] = eval(value)
        except NameError:
            hyp_dict[key] = str(value) 
    return hyp_dict


def get_categorical_transforms() -> tuple[list, T.Compose]:
    """This function splits up augmentation into four phases, each with
    different general augmentation techniques. The idea behind it is to only
    select one augmentation from each phase, to ensure that the images are
    still classifiable after augmentation. Each phase also contains a dummy
    lambda option, which is to ensure that sometimes no augmentation is
    applied for a phase.

    Vertical flip is removed from phase 1. Phase 4 should only be utilized
    later, since it comprises adverserial training of the model which is
    not a focus in early stages.

    Returns:
        A composed element of PyTorch RandomChoices for each category and
        the combined list of categorical tarnsforms.
    """
    # Augmentations phase 1 (moving the image around):
    transforms_phase_1 = [T.RandomRotation(degrees = (0, 30)),
                          T.RandomHorizontalFlip(p = 1.0),
                          T.RandomAffine(degrees = 0, translate = (0.1, 0.3),
                                         scale = (0.75, 1.0), shear = (0, 0.2)),
                          T.Lambda(lambda x: x)]
    
    # Augmentations phase 2 (Simple color changes)
    transforms_phase_2 = [T.Grayscale(num_output_channels = 3), 
                          T.RandomAdjustSharpness(sharpness_factor = 2, p = 1.0),
                          T.RandomAutocontrast(p = 1.0), T.RandomEqualize(p = 1.0),
                          T.Lambda(lambda x: x)]

    # Augmentations phase 3 (Advanced color changes)
    transforms_phase_3 = [T.RandomInvert(p = 1.0), 
                          T.RandomPosterize(3, p = 1.0),
                          T.RandomSolarize(threshold = random.randint(100, 200), p = 0.5),
                          T.ColorJitter(brightness = (0.3, 1), contrast = (0.3, 1),
                                        saturation = (0.3, 1), hue = (-0.5, 0.5)),
                          T.Lambda(lambda x: x)]

    # Augmentations phase 4 (Adding noise)
    # From the imagecorruption library https://github.com/bethgelab/imagecorruptions
    transforms_phase_4 = [CustomCorruption(corruption_name = "gaussian_blur"),
                          CustomCorruption(corruption_name = "shot_noise"),
                          CustomCorruption(corruption_name = "impulse_noise"),
                          CustomCorruption(corruption_name = "motion_blur"),
                          CustomCorruption(corruption_name = "zoom_blur"),
                          CustomCorruption(corruption_name = "pixelate"),
                          CustomCorruption(corruption_name = "jpeg_compression"),
                          T.Lambda(lambda x: x)]

    # Combining categorical transforms into one list
    # and combining into a composed element of random choices
    combined_transforms = transforms_phase_1 + transforms_phase_2 + \
                          transforms_phase_3
    categorical_transforms = T.Compose([T.RandomChoice(transforms_phase_1),
                                        T.RandomChoice(transforms_phase_2),
                                        T.RandomChoice(transforms_phase_3)])

    return combined_transforms, categorical_transforms


def get_transforms(transform_type: str = "categorical") -> T.Compose:
    """Function that retrieves transforms and combines them into a compose
    element based on which option is selected. The augmentations should only
    be randomized by RandomApply/RandomChoice, not by any random probability
    in the functions themselves. Transformations taken from: 
    https://pytorch.org/vision/stable/transforms.html.

    Args:
        transform_type: The type of transform to be used. Options are
                        "random_choice", "categorical", "auto_augment" and
                        "rand_augment".
    Returns:
        Composed element of transforms.
    """
    combined_transforms, categorical_transforms = get_categorical_transforms()
    transform_options = {"rand_augment": T.RandAugment(),
                         "categorical": categorical_transforms,
                         "random_choice": T.RandomChoice(combined_transforms),
                         "auto_augment": T.AutoAugment(policy = T.AutoAugmentPolicy.IMAGENET),
                         "no_augment": T.Lambda(lambda x: x)}

    transform = T.Compose([
        transform_options[transform_type],
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    return transform
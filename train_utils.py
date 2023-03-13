import datetime
import colorsys
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pandas as pd
import random
import shutil
import time
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchmetrics import ConfusionMatrix
import torchvision
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, \
                               shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights, \
                               resnet18, ResNet18_Weights, \
                               efficientnet_b1, EfficientNet_B1_Weights
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from deepspeed.profiling.flops_profiler import get_model_profile
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


def convert_to_list(labels: list) -> list:
    """Function that converts a list of tensors to a list of lists.

    Args:
        labels: list of tensors.
    Returns:
        List of lists.
    """
    label_list = []
    for tensor in labels:
        label_list.append(tensor.tolist())
    return flatten_list(label_list)


def flatten_list(list: list) -> list:
    """Function that takes a list of lists and flattens it into
    a single list.

    Args:
        list: list of lists.
    Returns:
        Flattened list.
    """
    return [item for sublist in list for item in sublist]


def add_confusion_matrix(combined_labels: list, combined_labels_pred: list,
                         tensorboard_writer: SummaryWriter):
    """Function that adds a confusion matrix to the tensorboard.
    Only saved for the last epoch to the hyperparameter writer.
    The class labels are defined as:
    0: fail_label_crooked_print
    1: fail_label_half_printed
    2: fail_label_not_fully_printed
    3: no_fail

    Args:
        conf_mat: Confusion matrix as a torch tensor.
        tensorboard_writer: hyperparameter writer.
    """
    # Creating confusion matrix from predictions and actual
    conf_matrix = ConfusionMatrix(task = "multiclass", num_classes = 4)
    combined_labels = convert_to_list(combined_labels)
    combined_labels_pred = convert_to_list(combined_labels_pred)
    conf_mat = conf_matrix(torch.tensor(combined_labels_pred),
                           torch.tensor(combined_labels))
    classes = ["Crooked print", "Half print", "Not full print", "No fail"]   

    # Plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat, cmap='Blues')

    # Setting x-axis and y-axis labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Add colorbar and title
    ax.figure.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Adding text for each datapoint
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(conf_mat[i, j]), ha = "center",
                           va = "center", color = "w")

    # Adding to tensorboard
    tensorboard_writer.add_figure("Confusion Matrix", fig)


def report_metrics(flag: dict, start_time: float, epoch_length: int, 
                   acc: float, f1_score: float, loss_over_epoch: float,
                   total_imgs: int, writer: SummaryWriter, epoch: int,
                   experiment_path: str):
    """Function that allows for writing performance metrics to the terminal
    and the tensorboard.

    Args:
        flag: Boolean for printing to the terminal or not.
        start_time: Time of start of the epoch.
        epoch_length: Number of batches in the epoch.
        acc: Accumulated accuracy over the batches in the epoch.
        f1_score: Accumulated f1_score over the batches in the epoch
        loss_over_epoch: Accumulated loss over the epoch.
        total_imgs: Total number of images in the epoch.
        writer: Tensorboard writer, either for training or validation.
        epoch: Current epoch to write metrics to.
        experiment_path: Path to the experiment folder.
    """
    # Measuring elapsed time and reporting metrics over epoch
    elapsed_time = time.time() - start_time

    # Different graphs for training and validation metrics
    if writer.log_dir.endswith("train"):
        phase = "Train "
    else:
        phase = "Validation "

    # Calculating accuracy and score, since they are needed for writing.
    mean_accuracy = (acc / epoch_length)
    mean_f1_score = (f1_score / epoch_length)

    if flag["Terminal"] == True:
        print("Loss = " + str(round(loss_over_epoch, 2)))
        print("Accuracy = " + str(round(mean_accuracy, 2)))
        print("F1 score = " + str(round(mean_f1_score, 2)))
        print("FPS = " + str(round(total_imgs / elapsed_time, 2)) + "\n")

    if flag["Tensorboard"] == True:
        # Writing results to tensorboard
        writer.add_scalar(phase + "Loss", loss_over_epoch, epoch)
        writer.add_scalar(phase + "Accuracy", mean_accuracy, epoch)
        writer.add_scalar(phase + "F1 score", mean_f1_score, epoch)
    
    # Writing the results to a txt file as well, for results recording
    with open(experiment_path + "/results.txt", "a") as file:
        file.write("Phase: " + phase + "\n")
        file.write("Epoch: " + str(epoch) + "\n")
        file.write("Loss = " + str(round(loss_over_epoch, 2)) + "\n")
        file.write("Accuracy = " + str(round(mean_accuracy, 2)) + "\n")
        file.write("F1 score = " + str(round(mean_f1_score, 2)) + "\n")
        file.write("FPS = " + str(round(total_imgs / elapsed_time, 2)) + "\n")
        file.write("\n")
    file.close()


def set_classification_layer(model: torchvision.models):
    """This function changes the final classification layer
    from a PyTorch deep learning model to a four output classes version.
    The function edits the model variable, so no need to return it.

    Args: 
        model: This is a default model, with many more classes than 4.
    """
    classes = 4
    if model.__class__.__name__ == "MobileNetV2" or \
       model.__class__.__name__ == "EfficientNet":
        model.classifier[1] = nn.Linear(in_features = 1280, out_features = classes)
    elif model.__class__.__name__ == "ShuffleNetV2":
        model.fc = nn.Linear(in_features = 1024, out_features = classes)
    elif model.__class__.__name__ == "ResNet":
        model.fc = nn.Linear(in_features = 512, out_features = classes)

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

    # Evaluating the string expressions in the JSON experiment setup file
    for key, value in hyp_dict.items():
        try:
            hyp_dict[key] = eval(value)
        except NameError:
            hyp_dict[key] = str(value) 
    return hyp_dict


def calculate_flops(model: torchvision.models, batch_size: int = 32, warm_up: int = 10):
    """ This function uses the DeepSpeed Python library to calculate the FLOPS,
    MACS and the parameters of a model. Keep in mind that the amount of FLOPS
    linearly increases with the batch size.

    Args:
        model: PyTorch model.
        batch_size: Batch size of the model.
        warm_up: Number of warm up iterations before calculations are made.
    """
    flops, macs, params = get_model_profile(model = model,
                                    input_shape = (batch_size, 3, 224, 224),
                                    print_profile = False, detailed = False,
                                    warm_up = warm_up)
    print(f"FLOPS = {flops}\nMACS = {macs}\nparams = {params}")


def merge_experiments(experiment_list: list, path: str):
    """Function that merges experiment folders into one folder.

    Args:
        experiment_list: List of experiments to be merged.
        path: Path to the folder containing the experiments.
    """
    for experiment in experiment_list:
        files = os.listdir(path)
        for file in files:
            if os.path.isdir(os.path.join(path, file)):
                if file.startswith(experiment):
                    shutil.copytree(os.path.join(path, file), os.path.join(path, experiment, file))
                    shutil.rmtree(os.path.join(path, file))


def plot_results(path: str, title: str):
    """This function is used to plot a number of experiment runs
    into a single matplotlib plot, with legend/axis labels etc,
    which are absent from a tensorboard representation.

    Args:
        path: Path to the folder containing the experiment runs.
        title: Title of the plot.
    """

    # First counting how many files to plot
    files = os.listdir(path)
    csv_files = []
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(file)

    # Creating colour list based on amount of files
    num_colors = len(csv_files)
    hue_list = [i/num_colors for i in range(num_colors)]
    random.shuffle(hue_list)
    color_list = []

    # Converting HSV to RGB
    for hue in hue_list:
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        color_list.append(rgb)

    # Plotting all CSV files, n is the amount of charachters that the date/time
    # entails, plus the underscores and the val flag.
    legend = []
    n = 20
    for i, file in enumerate(csv_files):
        data = pd.read_csv(os.path.join(path, file))
        data.drop("Wall time", axis = 1, inplace = True)
        plt.grid()
        plt.plot(data["Step"], data["Value"], color = color_list[i], linewidth = 1.0)
        legend.append(os.path.splitext(file)[0][:-n])
    plt.legend(legend)
    plt.ylabel("Accuracy") 
    plt.xlabel("Epochs")
    plt.title(title)
    plt.savefig(os.path.join(path, title + ".png"))


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
                          transforms_phase_3 + transforms_phase_4
    categorical_transforms = T.Compose([T.RandomChoice(transforms_phase_1),
                                        T.RandomChoice(transforms_phase_2),
                                        T.RandomChoice(transforms_phase_3),
                                        T.RandomChoice(transforms_phase_4)])

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
                         "no_augment": T.Lambda(lambda x: x),
                         "random_apply": T.RandomOrder(combined_transforms)}

    transform = T.Compose([
        transform_options[transform_type],
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    return transform


if __name__ == '__main__':
    # train_utils is only used to calculate GFLOPS of a model or to plot results
    #calculate_flops(model = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT))
    plot_results(os.path.join("Master-Thesis-Experiments", "Experiment reporting"), "Experiments Validation Accuracy")
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import time
import torch
import torchvision
import torchvision.transforms as T
import warnings

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from deepspeed.profiling.flops_profiler import get_model_profile
from imagecorruptions import corrupt
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import ConfusionMatrix
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, \
                               shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights, \
                               resnet18, ResNet18_Weights, \
                               efficientnet_b1, EfficientNet_B1_Weights

from datasets import NTZFilterDataset, NTZFilterSyntheticDataset, \
                     CIFAR10Dataset


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


def get_device() -> torch.device:
    """Function that returns the device, either cuda if gpu is availble or cpu.
    
    Returns:
        Device on which the model is ran.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))
    return device


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


def cutoff_date(folder_name: str):
    """This function takes a folder in the form of a string.
    It cuts off the date and time from the end of the string.
    This function expects the folder name to not be in a folder.

    Args:
        folder_name: Name of the folder.
    """
    return os.path.normpath(folder_name).split(os.sep)[-1][:len(folder_name)-20]


def add_confusion_matrix(combined_labels: list, combined_labels_pred: list,
                         tensorboard_writer: SummaryWriter, label_map: dict):
    """Function that adds a confusion matrix to the tensorboard.
    Only saved for the last epoch to the hyperparameter writer.

    Args:
        combined_labels: List of all true labels.
        combined_labels_pred: List of all predicted labels.
        tensorboard_writer: hyperparameter writer.
        label_map: Dictionary that maps the labels to the correct names.
    """
    # Retrieving class names
    classes = list(label_map.values())

    # Creating confusion matrix from predictions and actual
    conf_matrix = ConfusionMatrix(task = "multiclass", num_classes = len(classes))
    combined_labels = convert_to_list(combined_labels)
    combined_labels_pred = convert_to_list(combined_labels_pred)
    conf_mat = conf_matrix(torch.tensor(combined_labels_pred),
                           torch.tensor(combined_labels))
    
    # Plotting confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat, cmap = "Blues")

    # Setting x-axis and y-axis labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Add colorbar and title, make x labels smaller,
    # because they otherwise overlap
    ax.figure.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(fontsize = 6)

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

    # Calculating accuracy, loss and score, since they are needed for writing.
    mean_accuracy = (acc / epoch_length)
    mean_f1_score = (f1_score / epoch_length)
    loss_over_epoch = (loss_over_epoch / epoch_length)

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
    with open(os.path.join(experiment_path, "results.txt"), "a") as file:
        file.write("Phase = " + phase + "\n")
        file.write("Epoch = " + str(epoch) + "\n")
        file.write("Loss = " + str(round(loss_over_epoch, 2)) + "\n")
        file.write("Accuracy = " + str(round(mean_accuracy, 2)) + "\n")
        file.write("F1 score = " + str(round(mean_f1_score, 2)) + "\n")
        file.write("FPS = " + str(round(total_imgs / elapsed_time, 2)) + "\n")
        file.write("\n")
    file.close()


def find_classification_module(model: torchvision.models) -> tuple:
    """Function that takes in a model and finds the classification layer.
    An assumption is made that the linear layer is either directly located in
    a layer called fc or in a sub module in a layer called classifier.

    Args:
        model: PyTorch deep learning model
    Returns:
        The classification module of the model, its name
        and an index of its location if necessary.
    """
    idx = None
    for name, module in model.named_children():
        # For models that have the linear layer in a sub module "classifier"
        # (MobileNetV2 and EfficientNetB1 have this)
        if name == "classifier":
            idx = 0
            for _, sub_module in module.named_children():
                if isinstance(sub_module, nn.Linear):
                    return name, sub_module, idx
                idx += 1
        # For models that have a linear layer in a module "fc".
        # (Resnet18 and ShuffleNetV2 have this)
        elif name == "fc" and isinstance(module, nn.Linear):
            return name, module, idx
    raise ValueError("No linear layer module found in model.")


def set_classification_layer(model: torchvision.models, classes: int):
    """This function changes the final classification layer
    from a PyTorch deep learning model to a X output classes version,
    depending on the dataset.

    Args: 
        model: This is a default model, with usually a lot more output classes.
        classes: The amount of output classes for the model.
    
    Returns:
        The model with the new classification layer.
    """
    name, module, _ = find_classification_module(model)
    in_features = module.in_features
    model._modules[name] = nn.Linear(in_features = in_features, out_features = classes)
    return model


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
    """Manual collate function for testing dataloader.
    It converts the images to a torch stack and returns the paths.

    Args:
        batch: batch of data items from a dataloader.
    Returns:
        images as torch stack and paths.
    """
    path, images, _ = zip(*batch)
    images = torch.stack(list(images), dim = 0)
    return images, path


def remove_predicts(path):
    """Function that removes all old predictions from a
    test prediction folder given in the path.

    Args:
        path: The path to the folder with the old predictions.
    """
    if os.path.exists(path):
        files = [f for f in os.listdir(path) 
                if os.path.isfile(os.path.join(path, f))]
        for old_file in files:
            file_path = os.path.join(path, old_file)
            os.unlink(file_path)
    else:
        os.mkdir(path)


def save_test_predicts(predicted_labels: list, paths: list,
                       img_destination: str, dataset: Dataset,
                       predicted_uncertainty: list) -> tuple[list, list]:
    """Function that converts labels to a list and then saves paths and labels
    to appropriate prediction directories. The prediction directory in
    img_destination, should already exist, by running remove_predicts
    somewhere before it.

    Args:
        predicted_labels: list of tensors with predicted labels.
        paths: list of lists with paths (strings) to images.
        img_destination: Designated folder to save images to.
        dataset: The dataset the predictions are made for.
        predicted_uncertainty: list of tensors with predicted uncertainty
        if empty, the model is not an rbf model.
    
    Returns:
        Prediction list and paths converted to correct format
    """
    # If the list is not a list of torch tensors, do not convert
    # This is the case in DUQ since the labels are calculated manually
    if type(predicted_labels[0]) == torch.Tensor:
        predicted_labels = convert_to_list(predicted_labels)
    if len(predicted_uncertainty) != 0 and type(predicted_uncertainty[0]) == torch.Tensor:
        predicted_uncertainty = convert_to_list(predicted_uncertainty)
    paths = flatten_list(paths)

    # Getting the dataset label map
    label_dict = dataset.label_map

    # Set the text size, based on the average of width/height of the images
    ex_width, ex_height = Image.open(paths[0]).size
    text_size = int(((ex_height + ex_width) / 2)  / 16)
    text_loc = get_text_loc(dataset)

    # Loading necessary information and then drawing on the label on each image.
    for idx, img_path in enumerate(paths):
        name = os.path.normpath(img_path).split(os.sep)[-1]
        img = Image.open(img_path)
        label_name = label_dict[predicted_labels[idx]]

        # Drawing label
        img = draw_label(img, text_loc, text_size, label_name)

        # Adding uncertainty in a white block if RBF model
        if len(predicted_uncertainty) != 0:
            img = draw_uncertainty_bar(img, predicted_uncertainty[idx], text_size)

        img.save(os.path.join(img_destination, name))
    return predicted_labels, paths


def draw_label(img: Image, text_loc: tuple,
               text_size: int, label_name: str) -> Image:
    """Function that draws a label on the top left of an image.

    Args:
        img: Image to draw the label on.
        text_loc: Location of the text.
        text_size: Size of the text to be drawn.
        label_name: Name of the label to be drawn.
    """
    font = ImageFont.truetype(os.path.join("raw_data", "arial.ttf"), size = text_size)
    draw = ImageDraw.Draw(img)
    draw.text(text_loc, label_name, font = font, fill = 255)
    return img


def draw_uncertainty_bar(img: Image, uncertainty: float,
                         text_size: int, ) -> Image:
    """Function that draws a bar with the uncertainty on the bottom of the image.
    
    Args:
        img: Image to draw the bar on.
        uncertainty: Uncertainty of the prediction.
        text_size: Size of the text to be drawn.

    Returns: Image with the bar drawn on it.
    """
    font = ImageFont.truetype(os.path.join("raw_data", "arial.ttf"), size = text_size)
    uncertainty = round(uncertainty, 2)
    width, height = img.size
    bar_height = text_size
    bar_img = Image.new("RGB", (width, height + bar_height), color = "white")
    bar_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(bar_img)
    draw.text((0, height), "Certainty = " + str(uncertainty), font = font, fill = 255)
    return bar_img


def get_text_loc(dataset: Dataset):
    """Function that retrieves the preferred text location for a dataset
    
    Args:
        dataset: The dataset the predictions are made for.
    """
    # Add other dataset text locations here if necessary
    if isinstance(dataset, NTZFilterDataset):
        text_loc = (10, 10)
    else:
        text_loc = (0, 0)
    return text_loc


def setup_tensorboard(experiment_name: str, folder: str) -> tuple[list[SummaryWriter], str]:
    """Function that provides tensorboard writers for training and validation.
    
    Args:
        experiment_name: Name of the experiment that is run.
        folder: Folder in which the experiment is run.
    Returns:
        List of tensorboard writers.
    """
    current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    experiment_path = os.path.join("Results", folder, (experiment_name + "-" + current_time)) 
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
    It replaces default hyperparameters with the ones from a JSON file.

    Args:
        experiment_name: Name of the experiment that is run.
    Returns:
        Dictionary with hyperparameters.
    """
    # First getting all default arguments from DEFAULT.json
    experiment_location = os.path.join("Experiments", "DEFAULT.json")
    with open(experiment_location, "r") as f:
        def_dict = json.load(f)

    # Getting arguments from the experiment
    experiment_location = os.path.join("Experiments", (experiment_name + ".json"))
    with open(experiment_location, "r") as f:
        exp_dict = json.load(f)

    # Comparing the two dictionaries and replacing default values with
    # values from the experiment if they exist.
    hyp_dict = def_dict.copy()
    hyp_dict.update(exp_dict)

    # Evaluating the string expressions in the JSON experiment setup file
    for key, value in hyp_dict.items():
        try:
            hyp_dict[key] = eval(value)
        except NameError:
            hyp_dict[key] = str(value) 
    return hyp_dict


def calculate_flops():
    """ This function uses the DeepSpeed Python library to calculate the FLOPS,
    MACS and the parameters of a model. Keep in mind that the amount of FLOPS
    linearly increases with the batch size. The FLOPS reported are the
    floating point operations, so not per second (which is different).
    """
    models = [mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT),
              shufflenet_v2_x1_0(weights = ShuffleNet_V2_X1_0_Weights.DEFAULT),
              resnet18(weights = ResNet18_Weights.DEFAULT),
              efficientnet_b1(weights = EfficientNet_B1_Weights.DEFAULT)]

    batch_size = 32
    warm_up = 10
    for model in models:
        flops, macs, params = get_model_profile(model = model,
                                        input_shape = (batch_size, 3, 224, 224),
                                        print_profile = False, detailed = False,
                                        warm_up = warm_up)
        print(str(model.__class__.__name__))
        print(f"FLOPS = {flops}\nMACS = {macs}\nparams = {params}")
        print("\n")


def merge_experiments(experiment_list: list, path: str):
    """Function that merges experiment folders of the same
    experiment runs into a single folder.

    Args:
        experiment_list: List of experiments to be merged.
        path: Path to the folder containing the experiments.
    """
    for experiment in experiment_list:
        files = os.listdir(path)
        for file in files:
            if os.path.isdir(os.path.join(path, file)):
                if file.startswith(experiment):
                    shutil.copytree(os.path.join(path, file),
                                    os.path.join(path, experiment, file))
                    shutil.rmtree(os.path.join(path, file))


def calculate_acc_std(experiment_list: list, path: str):
    """Function that combines the mean and standard deviation
    of the validation accuracy on the final epoch over a number
    of experiment runs. Always run after merge_experiments!!!
    Since this function expects the experiments to be merged.

    Args:
        experiment_list: List of experiments to be combined.
        path: Path to the folder containing the experiments.
    """
    for experiment in experiment_list:
        val_acc_fe = []
        val_loss_fe = []
        files = os.listdir(os.path.join(path, experiment))
        for file in files:
            res_path = os.path.join(path, experiment, file, "results.txt")
            with open(res_path, "r") as res_txt:
                # Read all the lines into a list
                lines = res_txt.readlines()

            # Extract validation accuracy on final epoch
            # Accuracy = is 11 characters, so those are removed
            # Loss = is 7 characters, so also removed
            val_acc_fe.append(float(lines[-4].strip()[11:]))
            val_loss_fe.append(float(lines[-5].strip()[7:]))
        with open(os.path.join(path, experiment, "results.txt"), "a") as com_res:
            com_res.write("Experiment: " + str(experiment) + "\n")
            com_res.write("List: " + str(val_acc_fe) + "\n")
            com_res.write("Mean validation accuracy on final epoch: "
                          + str(np.mean(val_acc_fe)) + "\n")
            com_res.write("Standard deviation of validation accuracy on final epoch: "
                          + str(np.std(val_acc_fe)) + "\n")
            com_res.write("Mean validation loss on final epoch: "
                            + str(np.mean(val_loss_fe)) + "\n")
            com_res.write("Standard deviation of validation loss on final epoch: "
                            + str(np.std(val_loss_fe)) + "\n")
            com_res.write("===========================================================\n")  


def get_default_transform(dataset: Dataset = NTZFilterDataset) -> T.Compose:
    """This function returns a default transformation based on the
    type of dataset. The transformation like this is used for
    validation/testing set.

    Returns:
        Composed elemenent of PyTorch transforms.
    """
    if dataset.__name__ == "NTZFilterDataset" or \
       dataset.__name__ == "NTZFilterSyntheticDataset":
        # Based on MobileNetV2 default transforms
        # Resizing because the images are quite large and not square
        transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
    elif dataset.__name__ == "CIFAR10Dataset":
        transform = T.Compose([
            T.RandomCrop(32, padding = 4),
            T.ToTensor(),
            T.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
        ])
    return transform


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
                                         scale = (0.75, 1.0), shear = (0, 0.2))]
    
    # Augmentations phase 2 (Simple color changes)
    transforms_phase_2 = [T.Grayscale(num_output_channels = 3), 
                          T.RandomAdjustSharpness(sharpness_factor = 2, p = 1.0),
                          T.RandomAutocontrast(p = 1.0), T.RandomEqualize(p = 1.0)]

    # Augmentations phase 3 (Advanced color changes)
    transforms_phase_3 = [T.RandomInvert(p = 1.0), 
                          T.RandomPosterize(3, p = 1.0),
                          T.RandomSolarize(threshold = random.randint(100, 200), p = 0.5),
                          T.ColorJitter(brightness = (0.3, 1), contrast = (0.3, 1),
                                        saturation = (0.3, 1), hue = (-0.5, 0.5))]

    # Augmentations phase 4 (Adding noise)
    # From the imagecorruption library https://github.com/bethgelab/imagecorruptions
    transforms_phase_4 = [CustomCorruption(corruption_name = "gaussian_blur"),
                          CustomCorruption(corruption_name = "shot_noise"),
                          CustomCorruption(corruption_name = "impulse_noise"),
                          CustomCorruption(corruption_name = "motion_blur"),
                          CustomCorruption(corruption_name = "zoom_blur"),
                          CustomCorruption(corruption_name = "pixelate"),
                          CustomCorruption(corruption_name = "jpeg_compression")]

    # Combining categorical transforms into one list
    # and combining into a composed element of random choices
    combined_transforms = transforms_phase_1 + transforms_phase_2 + \
                          transforms_phase_3 + transforms_phase_4
    
    # Adding no augment option to phases
    for phase in [transforms_phase_1, transforms_phase_2, 
                   transforms_phase_3, transforms_phase_4]:
        phase = phase.append(T.Lambda(lambda x: x))

    categorical_transforms = T.Compose([T.RandomChoice(transforms_phase_1),
                                        T.RandomChoice(transforms_phase_2),
                                        T.RandomChoice(transforms_phase_3),
                                        T.RandomChoice(transforms_phase_4)])

    return combined_transforms, categorical_transforms


def get_transforms(dataset: Dataset = NTZFilterDataset,
                   transform_type: str = "categorical") -> T.Compose:
    """Function that retrieves transforms and combines them into a compose
    element based on which option is selected. The augmentations should only
    be randomized by RandomChoice, not by any random probability
    in the functions themselves. Transformations taken from: 
    https://pytorch.org/vision/stable/transforms.html.

    Args:
        transform_type: The type of transform to be used. Options are
                        "random_choice", "categorical", "auto_augment" and
                        "rand_augment".
        dataset: Type of dataset. Options are "NTZFilterDataset" and "CIFAR10".       
    Returns:
        Composed element of transforms.
    """
    combined_transforms, categorical_transforms = get_categorical_transforms()
    transform_options = {"rand_augment": T.RandAugment(),
                         "categorical": categorical_transforms,
                         "random_choice": T.RandomChoice(combined_transforms),
                         "auto_augment": T.AutoAugment(policy = T.AutoAugmentPolicy.IMAGENET),
                         "no_augment": T.Lambda(lambda x: x),
                         "simple": T.RandomHorizontalFlip()}

    # Getting default transform and inserting selected transform type
    transform = get_default_transform(dataset)
    transform.transforms.insert(0, transform_options[transform_type])

    return transform


def get_data_loaders(batch_size: int = 32, transform: T.Compose = get_transforms(),
                     dataset: Dataset = NTZFilterDataset):
    """Function that creates the data loaders for the training, validation
    and testing. The train set is also augmented, while the validation and
    testing sets are not.

    Args:
        batch_size: Batch size for the data loaders.
        transform: Transform to be applied to the data.
        dataset: Dataset for which the data should be loaded.

    Returns:
        Dictionary with the data loaders for training, validation and testing.
    """
    # Setting shuffle and num_workers, it is the same across the entire project.
    shuffle = True
    num_workers = 4

    # Setting the dataset type
    dataset_path = os.path.join("data", dataset.__name__.removesuffix("Dataset"))

    # File paths
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")

    # Getting default transform
    default_transform = get_default_transform(dataset)

    # Creating datasets for training/validation/testing
    train_data = dataset(train_path, transform)
    val_data = dataset(val_path, default_transform)
    test_data = dataset(test_path, default_transform)

    # Creating data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size = batch_size,
                              collate_fn = sep_collate, shuffle = shuffle,
                              num_workers = num_workers)
    val_loader = DataLoader(val_data, batch_size = batch_size,
                            collate_fn = sep_collate, shuffle = shuffle,
                            num_workers = num_workers)
    test_loader = DataLoader(test_data, batch_size = batch_size,
                             collate_fn = sep_test_collate, 
                             shuffle = False, num_workers = num_workers)
    
    data_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    return data_loaders

if __name__ == '__main__':
    # utils is only used to calculate flops and model parameters
    calculate_flops()
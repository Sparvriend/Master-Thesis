from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import random
from tqdm import tqdm
import shutil
import zipfile

from synthetic_data import setup_data_generation 

# Forming file paths for destinations of training, testing and validation data
TRAIN_DESTINATION = os.path.join("data", "NTZFilter", "train")
VAL_DESTINATION = os.path.join("data", "NTZFilter", "val")
TEST_DESTINATION = os.path.join("data", "NTZFilter", "test")

# Forming file paths for source of data, as well as the different classes
DATA_LOCATION_NTZ = os.path.join("raw_data", "NTZ_filter_label_data")
DATA_LOCATION_SYN = os.path.join("raw_data", "NTZ_filter_synthetic", "synthetic_data")
DATA_CLASSES = ["fail_label_crooked_print", "fail_label_half_printed",
                "fail_label_not_fully_printed", "no_fail"]

# Forming train/validation splits
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1


def split_and_move_NTZ():
    """Function that takes images from NTZ_filter_label_data and splits them
    into training, testing and validation data. In this function it is assumed
    that the directories to paste the images in already exist in data/*/*.
    """
    # 80/10/10 split for training/validation/testing data,
    # test split is 1 - train_split - val_split

    # Setting a maximum of samples for each class (70)
    max_samples = 70

    for data_class in DATA_CLASSES:
        # Combining path and listing all files in the path
        path = os.path.join(DATA_LOCATION_NTZ, data_class)
        files = [f for f in os.listdir(path)
                 if os.path.isfile(os.path.join(path, f))]
        # Removal of .ivp files and other things
        cleaned_files = []
        for img_file in files:
            if img_file.endswith("bmp"):
                cleaned_files.append(img_file)
        
        # Shuffling the images [WHEN REDOING THE MOVE, ENSURE THE MODEL IS
        # NEVER TRAINED ON VALIDATION/TESTING DATA FROM PREVIOUS SPLIT]
        random.shuffle(cleaned_files)

        # Selecting only max_samples per class
        if len(cleaned_files) > max_samples:
            cleaned_files = cleaned_files[:max_samples]

        # Splitting data into train, test and validation
        train_prop = int(len(cleaned_files) * TRAIN_SPLIT)
        val_prop = int(len(cleaned_files) * (TRAIN_SPLIT + VAL_SPLIT))
        train, val, test = np.split(cleaned_files, [train_prop, val_prop])

        # Removing all old files present in the folders
        for destination_type in [TRAIN_DESTINATION, TEST_DESTINATION,
                                 VAL_DESTINATION]:
            path = os.path.join(destination_type, data_class)
            files = [f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))]
            for old_file in files:
                file_path = os.path.join(path, old_file)
                os.unlink(file_path)

        # Moving all files to appropriate directories
        for train_file in train:
            destination_path = os.path.join(TRAIN_DESTINATION, data_class)
            file_path = os.path.join(DATA_LOCATION_NTZ, data_class, train_file)
            shutil.copy(file_path, destination_path)
        
        for val_file in val:
            destination_path = os.path.join(VAL_DESTINATION, data_class)
            file_path = os.path.join(DATA_LOCATION_NTZ, data_class, val_file)
            shutil.copy(file_path, destination_path)

        for test_file in test:
            destination_path = os.path.join(TEST_DESTINATION, data_class)
            file_path = os.path.join(DATA_LOCATION_NTZ, data_class, test_file)
            shutil.copy(file_path, destination_path)


def split_and_move_NTZ_synthetic():
    # Setting a maximum of samples for each class (70)
    n = 70

    # Create a synthetic data directory for the NTZ
    # dataset with the max amount of samples per class
    setup_data_generation(n)

    path = os.path.join("data", "NTZFilterSynthetic")
    if not os.path.exists(os.path.join("data", "NTZFilterSynthetic")):
        os.mkdir(path)
        os.mkdir(os.path.join(path, "train"))

    for data_class in DATA_CLASSES:
        os.mkdir(os.path.join(path, "train", data_class))
        class_path = os.path.join(DATA_LOCATION_SYN, data_class)
        class_files = os.listdir(class_path)

        # Removing old files present in the folder
        train_path = os.path.join(os.path.join(path, "train"), data_class)
        files = os.listdir(train_path)
        for old_file in files:
            file_path = os.path.join(train_path, old_file)
            os.unlink(file_path)

        train_files = random.sample(class_files, int(len(class_files) * TRAIN_SPLIT))
        for train_file in train_files:
            file_path = os.path.join(DATA_LOCATION_SYN, data_class, train_file)
            shutil.copy(file_path, train_path)
        

def split_and_move_CIFAR():
    """Function that moves images from the CIFAR dataset to a
    validation directory, from the training directory, based
    on a validation split percentage. The function assumes that
    a directory called CIFAR10 exists in the data directory,
    including a train directory with training images.

    The images can be downloaded from Kaggle:
    https://www.kaggle.com/competitions/cifar-10/data?select=train.7z
    """
    # Defining amount of data to use for validation
    val_split = 0.2

    # Extracting the data from the zip directories directly results in
    # the images existing in train/train, so this is checked and corrected
    if os.path.exists(os.path.join("data", "CIFAR10", "train", "train")):
        train_path = os.path.join("data", "CIFAR10", "train", "train")
        files = os.listdir(train_path)
        for file in files:
            file_path = os.path.join(train_path, file)
            destination_path = os.path.join("data", "CIFAR10", "train", file)
            shutil.move(file_path, destination_path)
        os.rmdir(train_path)

    # Selecting the amount of files to move from the train directory
    train_path = os.path.join("data", "CIFAR10", "train")
    files = os.listdir(train_path)
    move_amount = int(len(files)*val_split)
    files = files[:move_amount]

    # Creating validation directory   
    if not os.path.exists(os.path.join("data", "CIFAR10", "val")):
        os.mkdir(os.path.join("data", "CIFAR10", "val"))

    # Moving files to validation directory
    for file in files:
        file_path = os.path.join(train_path, file)
        destination_path = os.path.join("data", "CIFAR10", "val", file)
        shutil.move(file_path, destination_path)


def create_dirs():
    """Function that creates all directories required for running modules
    of the project. It is assumed that the folder NTZ_filter_label_data
    is already available and is ordered correctly. 
    """
    # Creating some directories for setting up the NTZFilter dataset.
    for destination_type in [TRAIN_DESTINATION, VAL_DESTINATION, 
                             TEST_DESTINATION]:
        for data_class in DATA_CLASSES:
            path = os.path.join(destination_type, data_class)
            if not os.path.exists(path):
                os.makedirs(path)
    
    # Creating all directories needed in Results/
    dirs = ["Experiment-Results", "Explainability-Results", "Test-Predictions"]
    for dir in dirs:
        path = os.path.join("Results", dir)
        if not os.path.exists(path):
            os.mkdir(path)


def extract_file(zip_file, file_info, extract_path):
    zip_file.extract(file_info, extract_path)


def unpack_zip_multithread(zip_name: str, n_threads: int = 16):
    """Function that unpacks a zip file using multithreading.
    The zip file is unpacked in the same directory as it is taken from.

    Args:
        zip_name: Name of the zip file to unpack
        n_threads: Number of threads to use for unpacking
    """
    extract_path = os.path.join(*zip_name.split("/")[:2])

    # Open the zip file
    with zipfile.ZipFile(zip_name, 'r') as zip_file:
        # Get the list of files in the zip archive
        file_list = zip_file.infolist()

        # Create a thread pool for multithreading unpacking
        with ThreadPoolExecutor(max_workers = n_threads) as executor:    
            # Send extraction tasks to thread pool
            futures = [executor.submit(extract_file,
                                       zip_file,
                                       file_info,
                                       extract_path)
                       for file_info in file_list]

            # Wait for all tasks to complete
            for future in futures:
                future.result()


def copy_set(origin_path: str, destination_path: str, class_name: str,
             img_set: list, set_type: str):
    """Function that copies a set of images from a class to a destination.
    
    Args:
        origin_path: Path to the origin of the images
        destination_path: Path to the destination of the images
        class_name: Name of the class to copy images from
        img_set: List of images to copy
        set_type: Type of set to copy to (train/val/test)
    """
    if set_type == "train" or set_type == "val":
        set_type = os.path.join(set_type, class_name)
    for img in img_set:
        shutil.copy(os.path.join(origin_path, class_name, img), os.path.join(destination_path, set_type, img))


def subset_imageNet():
    """Function that extracts images from ImageNet. n_imgs are extracted
    of n_classes. The images are added to the dataset called ImageNet10.
    If the ImageNet dataset is zipped, first extract it with 
    unpack_zip_multithread()."""

    # Define dataset paths
    ImageNet_path = "data/ImageNet/ILSVRC/Data/CLS-LOC/train/"
    ImageNet10_path = "data/ImageNet10"
    if not os.path.exists(ImageNet10_path):
        os.mkdir(ImageNet10_path)
        os.mkdir(os.path.join(ImageNet10_path, "train"))
        os.mkdir(os.path.join(ImageNet10_path, "val"))
        os.mkdir(os.path.join(ImageNet10_path, "test"))

    # Set amount of classes to sample from. 
    # Set the amount of images. The validation set and 
    # testing set both contain 10% of the size of the training set.
    n_classes = 10
    n_imgs = 500

    # Get a subset of ImageNet classes
    imageNet_classes = os.listdir(ImageNet_path)
    subset_classes = random.sample(imageNet_classes, n_classes)

    # Loop over subset classes to randomly sample from them
    for class_name in tqdm(subset_classes):
        class_imgs = os.listdir(os.path.join(ImageNet_path, class_name))
        subset_class_imgs = random.sample(class_imgs, n_imgs + int(n_imgs * 0.2))
        train_set = subset_class_imgs[:n_imgs]
        val_set = subset_class_imgs[n_imgs:n_imgs + int(n_imgs * 0.1)]
        test_set = subset_class_imgs[n_imgs + int(n_imgs * 0.1):]

        # Copy each set to designated directory
        if not os.path.exists(os.path.join(ImageNet10_path, "train", class_name)):
            os.mkdir(os.path.join(ImageNet10_path, "train", class_name))
        if not os.path.exists(os.path.join(ImageNet10_path, "val", class_name)):
            os.mkdir(os.path.join(ImageNet10_path, "val", class_name))
        copy_set(ImageNet_path, ImageNet10_path, class_name, train_set, "train")
        copy_set(ImageNet_path, ImageNet10_path, class_name, val_set, "val")
        copy_set(ImageNet_path, ImageNet10_path, class_name, test_set, "test")


if __name__ == '__main__':
    # Make sure to run create_dirs() before running split_and_move()
    #create_dirs()
    #split_and_move_NTZ()
    split_and_move_NTZ_synthetic()
    #split_and_move_CIFAR()
    #subset_imageNet()
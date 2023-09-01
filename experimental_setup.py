import argparse
import json
import os
import re
import time
import random
import shutil
import cv2
import numpy as np

EX_PATH = "Experiments"


def experiment_1():
    """Experiment 1: Synthetic data study on the NTZFilter Dataset.
    Experiment 1a: Adding synthetic data in different proportions to
    the training set. Only concerns ResNet18 and MobileNetV2.
    Experiment 1b: Training on only synthetic data, validating on real set.
    Experiment 1c: Training on only real data, validating on synthetic set.
    #TODO: Use the other two classifiers for 1b/1c as well? 
    #TODO: Incorporate average loss for synthetic data proportion
    """
    classifiers = ["mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)",
                   "resnet18(weights = ResNet18_Weights.DEFAULT)"]

    # Experiment 1a
    # In experiment 1a, the total size of the dataset is always 48.
    n_runs = 10
    combs = [[12, 0.25], [24, 0.5], [36, 0.75], [48, 1]]
    for comb in combs:
        train_set = comb[0]
        train_ratio = comb[1] 
        for classifier in classifiers:
            # Editing JSON file, creating synthetic data and running experiment
            ex_name = edit_json("experiment_1", ["model"],
                                [classifier, train_set, train_ratio])
            os.system("python3.10 synthetic_data.py " + str(train_set)
                      + " " + str(train_ratio) + " 0 0")
            os.system("python3.10 train.py " + ex_name.replace(".json", "")
                      + " --n_runs " + str(n_runs))
            delete_json(ex_name)

    # Experiment 1b/1c
    # In experiment 1b/1c, the size of the dataset can be larger than 48.
    n_runs = 1
    combs = [[200, 1, 0, 0], [0, 0, 48, 1]]
    for comb in combs:
        train_set = comb[0]
        train_ratio = comb[1]
        val_set = comb[2]
        val_ratio = comb[3]
        for classifier in classifiers:
            # Editing JSON file, creating synthetic data and running experiment
            ex_name = edit_json("experiment_1", ["model", "epochs"], [classifier, "40",
                                                            train_set, train_ratio,
                                                            val_set, val_ratio])
            os.system("python3.10 synthetic_data.py " + str(train_set)
                                                      + " " + str(train_ratio)
                                                      + " " + str(val_set)
                                                      + " " + str(val_ratio))
            os.system("python3.10 train.py " + ex_name.replace(".json", "")
                      + " --n_runs " + str(n_runs))
            delete_json(ex_name)
    
    # Calculating FID score per class
    # Compare roughly equal amount of samples (~70)
    # Create a temporary directory that includes synthetic data with ~70 samples
    classes = os.listdir(os.path.join("data", "NTZFilter", "train"))
    syn_path = os.path.join("raw_data", "NTZ_filter_synthetic", "synthetic_data")
    real_path = os.path.join("data", "NTZFilter", "train")
    os.mkdir(os.path.join(syn_path, "temp"))
    for c in classes:
        syn_files = os.listdir(os.path.join(syn_path, c))
        syn_files = random.sample(syn_files, 70)

        # Copy to temporary directory
        for file in syn_files:
            shutil.copy(os.path.join(syn_path, c, file), os.path.join(syn_path, "temp"))
            
        # Perform computation
        print("FID SCORE FOR CLASS " + c)
        os.system("python3.10 -m pytorch_fid " + os.path.join(syn_path, c) + " " + os.path.join(real_path, c))

        # Remove files
        syn_files = os.listdir(os.path.join(syn_path, "temp"))
        for file in syn_files:
            os.remove(os.path.join(syn_path, "temp", file))
        
    # Remove temp directory
    os.rmdir(os.path.join(syn_path, "temp"))
            

def experiment_2():
    """Experiment 2: Augmentation testing per classifier
    on the NTZFilterSynthetic dataset.
    # TODO: Incorporate average loss per augmentation-classifier
    # combination. Run the experiment 10 times.
    """
    combs = [["mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)", "20"],
                   ["resnet18(weights = ResNet18_Weights.DEFAULT)", "20"],
                   ["shufflenet_v2_x1_0(weights = ShuffleNet_V2_X1_0_Weights.DEFAULT)", "40"],
                   ["efficientnet_b1(weights = EfficientNet_B1_Weights.DEFAULT)", "20"]]
    augmentations = ["rand_augment", "categorical",
                    "random_choice", "auto_augment",]
    n_runs = 10
    
    # Create the dataset to run the experiment on
    create_def_combined()

    for comb in combs:
        classifier = comb[0]
        epochs = comb[1]
        for augment in augmentations:
            # Edit the JSON file, call the experiment and delete the JSON
            ex_name = edit_json("experiment_2", ["model", "augmentation", "epochs"],
                                                [classifier, augment, epochs])
            os.system("python3.10 train.py " + ex_name.replace(".json", "")
                      + " --n_runs " + str(n_runs))
            delete_json(ex_name)


def experiment_3():
    """Experiment 3: Classifier testing on the NTZFilterSynthetic dataset.
    Includes the best augmentation techniques from experiment 2:
    Rand augment consistently gives the best results for all classifiers.
    Includes a feature analsysis with IG.
    TRT vs. no TRT speeds have to be run manually.  
    GFLOPS calculation has to be run manually.
    Show loss graph as well as accuracy graph.
    # TODO: EXPERIMENT 3 USED TEST DATA WITHOUT SYNTHETIC INCLUSION
    # COMPLETE RERUN!!!
    """
    classifiers = ["mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)",
                   "resnet18(weights = ResNet18_Weights.DEFAULT)",
                   "shufflenet_v2_x1_0(weights = ShuffleNet_V2_X1_0_Weights.DEFAULT)",
                   "efficientnet_b1(weights = EfficientNet_B1_Weights.DEFAULT)"]
    n_runs = 1
    
    # Create the dataset to run the experiment on
    create_def_combined()

    for classifier in classifiers:
        # Edit the JSON, run the experiment, run IG and delete JSON
        ex_name = edit_json("experiment_3", ["model"], [classifier])
        ex_name_rm = ex_name.replace(".json", "")
        os.system("python3.10 train.py " + ex_name_rm
                  + " --n_runs " + str(n_runs))
        directory = find_directory(ex_name_rm)
        os.system("python3.10 explainability.py " + directory + " Captum")
        delete_json(ex_name)
        exit()


def experiment_4():
    """Experiment 4: DUQ analysis on CIFAR10 dataset.
    Experiment 4a: Classifier performance when DUQ converted with GP.
    Experiment 4b: Classifier performance when DUQ converted without GP.
    TODO: IS THIS EXPERIMENT EVEN NECESSARY?
    TODO: Rerun ShuffleNetV1 with GP 0.1
    """
    # combs = [["mobilenet_v2()", ["0.1", "None"]],
    #          ["resnet18()", ["0.5", "None"]],
    #          ["shufflenet_v2_x1_0()", ["0.1", "None"]],
    #          ["efficientnet_b1()", ["None"]]]
    combs = ["shufflenet_v2_x1_0()", ["0.1"]],
    n_runs = 1

    # 100 epochs for all models (CIFAR10) in experiment 4
    # MobileNetV2 GP speed: 5 min/epoch
    # MobileNetV2 NO GP speed: 30 sec/epoch

    # Resnet18 GP speed: 1 min/epoch
    # Resnet18 NO GP speed: 30 sec/epoch

    # ShuffleNetV2 GP speed: 3 min 20s/epoch
    # ShuffleNetV2 NO GP speed: 30 sec/epoch

    # EfficientNetB1 GP speed: - min/epoch
    # EfficientNetB1 NO GP speed: 30 sec/epoch

    # Total time = 11 min 20 sec/epoch * 100 = 18 hours 53 min
    # Rerun: ShuffleNetV2 0.1
    # Rerun: EfficientNetB1 None
    # Total time = 3 min 50s/epoch * 100 = 6 hours 23 min

    for comb in combs:
        classifier = comb[0]
        for gp in comb[1]:
            # Edit the JSON file, call the experiment and delete the JSON
            ex_name = edit_json("experiment_4", ["model", "gp_const"],
                                [classifier, gp])
            os.system("python3.10 train_rbf.py " + ex_name.replace(".json", "")
                      + " --n_runs " + str(n_runs))
            delete_json(ex_name)


def experiment_5():
    """Experiment 5: DUQ analysis on NTZFilter dataset.
    Includes feature analysis with IG on a DUQ model.
    Model speeds have to be run manually (No TRT).
    Uses rand augment for all classifiers since that is the best one
    # TODO: EXPERIMENT 5 USED TEST DATA WITHOUT SYNTHETIC INCLUSION
    # COMPLETE RERUN!!!
    """
    combs = [["mobilenet_v2()", "lr = 0.05"],
             ["resnet18()", "lr = 0.01"],
             ["shufflenet_v2_x1_0()", "lr = 0.05"],
             ["efficientnet_b1()", "lr = 0.01"]]
    n_runs = 1

    # Create the dataset to run the experiment on
    create_def_combined()

    # 100 epochs for all models (NTZFilterSynthetic) in experiment 5
    # MobileNetV2 GP speed: 1 min/epoch
    # Resnet18 GP speed: 10s/epoch
    # ShuffleNetV2 GP speed: 20s/epoch
    # EfficientNetB1 GP speed: 1min 20s/epoch
    # Total time = 2 min 50sec/epoch * 100 = 4 hours 43 min
    
    for comb in combs:
        classifier = comb[0]
        lr = comb[1]

        # Editing JSON, running experiment, running IG and deleting JSON.
        ex_name = edit_json("experiment_5", ["model"], [classifier, lr])
        ex_name_rm = ex_name.replace(".json", "")
        os.system("python3.10 train_rbf.py " + ex_name_rm
                  + " --n_runs " + str(n_runs))
        directory = find_directory(ex_name_rm)
        os.system("python3.10 explainability.py " + directory + " Captum")
        delete_json(ex_name)


def experiment_6():
    """Experiment 6: Edge case analysis.
    Experiment 6a: Given a DUQ model trained on CIFAR10, what
    uncertainty does it give when tested on the NTZFilterSynthetic dataset?
    Experiment 6b: Given a DUQ model trained on NTZFiltersynthetic,
    what uncertainty does it give when tested on the NTZFilterSynthetic
    dataset when gaussian noise is added to it?
    TODO: Exp 6A is actually quite hard to implement, since the dataset
    # has to be the one and then the other in different places and the
    # code present is not equipped to deal with that.
    # Skip this experiment or look for a way to do it. 
    """
    # Create the dataset to run the experiment on
    create_def_combined()
    syn_path = os.path.join("data", "NTZFilterSynthetic") 

    # Experiment 6a - Training on CIFAR10, testing on NTZFilterSynthetic
    # n_runs = 1
    # ex_name = edit_json("experiment_4", ["model", "gp_const"],
    #                     ["resnet18()", "0.5"])
    # os.rename(os.path.join(EX_PATH, ex_name),
    #           os.path.join(EX_PATH, ex_name.replace("4", "6a")))
    # ex_name = ex_name.replace("4", "6a")
    # ex_name_rm = ex_name.replace(".json", "")
    
    # os.system("python3.10 train_rbf.py " + ex_name_rm
    #           + " --n_runs " + str(n_runs))
    # # Creating fake CIFAR10 test directory that includes
    # # NTZFilterSynthetic testing images
    # cifar_path = os.path.join("data", "CIFAR10", "test")
    # os.rename(cifar_path, os.path.join("data", "CIFAR10", "test_"))
    # os.mkdir(cifar_path)
    # for c in os.listdir(os.path.join(syn_path, "test")):
    #     for file in os.listdir(c):
    #         shutil.copyfile(file, cifar_path)

    # directory = find_directory(ex_name_rm)
    # os.system("python3.10 explainability.py " + directory + " Captum")
    # delete_json(ex_name)
    # shutil.rmtree(cifar_path)
    # os.rename(os.path.join("data", "CIFAR10", "test_"), cifar_path)
    
    # Experiment 6b - Training on NTZFilterSynthetic, testing on NTZFilterSynthetic with noise
    # Train model
    n_runs = 1
    ex_name = edit_json("experiment_3", ["model", "RBF_flag"], ["resnet18()", "True"])
    os.rename(os.path.join(EX_PATH, ex_name),
              os.path.join(EX_PATH, ex_name.replace("3", "6b")))
    ex_name = ex_name.replace("3", "6b")
    ex_name_rm = ex_name.replace(".json", "")
    os.system("python3.10 train_rbf.py " + ex_name_rm
              + " --n_runs " + str(n_runs))

    # Create a noise dataset (offline augmentation), overwrites existing test dataset
    for c in os.listdir(os.path.join(syn_path, "test")):
        for file in os.listdir(os.path.join(syn_path, "test", c)):
            img = cv2.imread(os.path.join(syn_path, "test", c, file))
            gaussian_noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
            noisy_img = cv2.add(img, gaussian_noise)
            cv2.imwrite(os.path.join(syn_path, "test", c, file), noisy_img)

    # Test model
    directory = find_directory(ex_name_rm)
    os.system("python3.10 explainability.py " + directory + " Captum")
    delete_json(ex_name)


def delete_json(json_name: str):
    """Function that deletes a JSON file.

    Args:
        json_name: Name of the JSON file to delete.
    """
    try:
        os.remove(os.path.join(EX_PATH, json_name))
        print(f"{json_name} has been deleted.")
    except FileNotFoundError:
        print(f"{json_name} does not exist.")


def edit_json(json_name, json_args, json_values):
    """Function that takes a basic JSON file for an experiment,
    edits it and saves that new version.

    Args:
        json_name: Name of the JSON file to edit.
        json_args: List of arguments to edit.
        json_values: List of values to edit the arguments to.
    Returns:
        Name of the experiment results folder.
    """
    with open(os.path.join(EX_PATH, json_name + ".json")) as ex_file:
        data = json.load(ex_file)

    for arg, value in zip(json_args, json_values):
        data[arg] = value
    if json_name == "experiment_5":
        optimizer = data["optimizer"]
        data["optimizer"] = optimizer.replace("lr = 0.05", json_values[1])
        json_values[1] = json_values[1][5:]

    ex_name = json_name
    for value in json_values:
        ex_name += "_" + str(value)
    ex_name += ".json"
    re_pattern = r'\(.*\)'
    ex_name = re.sub(re_pattern, '', ex_name)

    with open(os.path.join(EX_PATH, ex_name), 'w') as temp_file:
        json.dump(data, temp_file)

    return ex_name


def find_directory(ex_name):
    """Function that finds the directory of the experiment results.

    Args:
        ex_name: Name of the experiment results folder.
    Returns:
        Name of the experiments results folder, but including
        the timestamp.
    """

    # Since the folder is saved with a timestamp, to run IG
    # it has to be selected first
    all_directories = os.listdir(os.path.join("Results", "Experiment-Results"))
    for directory in all_directories:
        if directory.startswith(ex_name):
            break
    return directory


def create_def_combined():
    """Function that defines the default call for making
    the combined NTZFilterSynthetic dataset."""
    os.system("python3.10 synthetic_data.py 150 0 20 0 20 0 --no_combine")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type = str)
    args = parser.parse_args()
    start = time.time()

    if args.experiment == "experiment_1":
        # Time estimate (no new synthetic data/1 run): 12 minutes
        experiment_1()
    elif args.experiment == "experiment_2":
        # Time estimate (no new synthetic data/1 run): 60 minutes
        experiment_2()
    elif args.experiment == "experiment_3":
        # Time estimate (no new synthetic data/1 run): 18 minutes
        experiment_3()
    elif args.experiment == "experiment_4":
        # Time estimate: 19 hours
        experiment_4()
    elif args.experiment == "experiment_5":
        # Time estimate: 4 hours
        experiment_5()
    elif args.experiment == "experiment_6":
        experiment_6()
    
    elapsed_time = time.time() - start
    print("Total time for " + args.experiment + " (H/M/S) = ", 
          time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
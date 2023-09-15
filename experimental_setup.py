import argparse
import json
import os
import re
import time
import random
import shutil
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy

from utils import merge_experiments, calculate_acc_std

EX_PATH = "Experiments"


def experiment_1():
    """Experiment 1: Synthetic data study on the NTZFilter Dataset.
    Experiment 1a: Adding synthetic data in different proportions to
    the training set. Only concerns ResNet18 and MobileNetV2.
    Experiment 1b: Training on only synthetic data, validating on real set.
    Experiment 1c: Training on only real data, validating on synthetic set.
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
                      + " " + str(train_ratio) + " 0 0 0 0")
            os.system("python3.10 train.py " + ex_name.replace(".json", "")
                      + " --n_runs " + str(n_runs))
            delete_json(ex_name)

    classifiers = ["mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)",
                   "resnet18(weights = ResNet18_Weights.DEFAULT)",
                   "shufflenet_v2_x1_0(weights = ShuffleNet_V2_X1_0_Weights.DEFAULT)",
                   "efficientnet_b1(weights = EfficientNet_B1_Weights.DEFAULT)"] 

    # Experiment 1b/1c
    # In experiment 1b/1c, the size of the dataset can be larger than 48.
    n_runs = 10
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
                                                      + " " + str(val_ratio)
                                                      + " 0 0")
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
    """
    classifiers = ["mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)",
                   "resnet18(weights = ResNet18_Weights.DEFAULT)",
                   "shufflenet_v2_x1_0(weights = ShuffleNet_V2_X1_0_Weights.DEFAULT)",
                   "efficientnet_b1(weights = EfficientNet_B1_Weights.DEFAULT)"]
    n_runs = 10
    
    # Create the dataset to run the experiment on
    create_def_combined()

    for classifier in classifiers:
        # Edit the JSON, run the experiment, run IG and delete JSON
        ex_name = edit_json("experiment_3", ["model"], [classifier])
        ex_name_rm = ex_name.replace(".json", "")
        os.system("python3.10 train.py " + ex_name_rm
                  + " --n_runs " + str(n_runs))
        directory = find_directory(ex_name_rm)
        if n_runs > 1:
            sub_dirs = os.listdir(os.path.join("Results", "Experiment-Results", directory))
            unmerge_experiments(ex_name_rm)
            for sub_dir in sub_dirs:
                os.system("python3.10 explainability.py " + sub_dir +
                          " Captum --explainability_variant guided_backpropagation")
            merge_experiments([ex_name_rm], os.path.join("Results", "Experiment-Results"))
            merge_experiments([ex_name_rm], os.path.join("Results", "Explainability-Results"))
            merge_experiments([ex_name_rm], os.path.join("Results", "Test-Predictions"))
            calculate_acc_std([ex_name_rm], os.path.join("Results", "Experiment-Results"))
        else:
            os.system("python3.10 explainability.py " + directory +
                      " Captum --explainability_variant guided_backpropagation")
        delete_json(ex_name)


def experiment_4():
    """Experiment 4: DUQ analysis on CIFAR10 dataset.
    Experiment 4a: Classifier performance when DUQ converted with GP.
    Experiment 4b: Classifier performance when DUQ converted without GP.
    """
    combs = [["mobilenet_v2()", ["0.1", "None"]],
             ["resnet18()", ["0.5", "None"]],
             ["shufflenet_v2_x1_0()", ["0.1", "None"]],
             ["efficientnet_b1()", ["None"]]]
    n_runs = 1

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
    """
    combs = [["mobilenet_v2()", "lr = 0.05"],
             ["resnet18()", "lr = 0.01"],
             ["shufflenet_v2_x1_0()", "lr = 0.05"],
             ["efficientnet_b1()", "lr = 0.01"]]
    n_runs = 3

    # Create the dataset to run the experiment on
    create_def_combined()
    
    for comb in combs:
        classifier = comb[0]
        lr = comb[1]

        # Editing JSON, running experiment, running IG and deleting JSON.
        ex_name = edit_json("experiment_5", ["model"], [classifier, lr])
        ex_name_rm = ex_name.replace(".json", "")
        os.system("python3.10 train_rbf.py " + ex_name_rm
                  + " --n_runs " + str(n_runs))
        directory = find_directory(ex_name_rm)
        if n_runs > 1:
            sub_dirs = os.listdir(os.path.join("Results", "Experiment-Results", directory))
            unmerge_experiments(ex_name_rm)
            for sub_dir in sub_dirs:
                os.system("python3.10 explainability.py " + sub_dir + " Captum")
            merge_experiments([ex_name_rm], os.path.join("Results", "Experiment-Results"))
            merge_experiments([ex_name_rm], os.path.join("Results", "Explainability-Results"))
            merge_experiments([ex_name_rm], os.path.join("Results", "Test-Predictions"))
            calculate_acc_std([ex_name_rm], os.path.join("Results", "Experiment-Results"))
        else:
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


def unmerge_experiments(ex_name):
    # I am retarded, but this seems like the best way of doing this,
    # instead of changing explainability.py/test.py so late in the
    # development process.

    for file in os.listdir(os.path.join("Results", "Experiment-Results", ex_name)):
        source = os.path.join("Results", "Experiment-Results", ex_name, file)
        destination = os.path.join("Results", "Experiment-Results", file)
        shutil.move(source, destination)
    shutil.rmtree(os.path.join("Results", "Experiment-Results", ex_name))
    os.remove(os.path.join("Results", "Experiment-Results", "results.txt"))


def create_def_combined():
    """Function that defines the default call for making
    the combined NTZFilterSynthetic dataset."""
    os.system("python3.10 synthetic_data.py 150 0 20 0 20 0 --no_combine")


def graph_experiment_1():
    check_remove()
    collected_data = extract_data("multiple")

    # Experiment 1a first
    sub_list = []
    substrs = ["12", "24", "36", "48"]
    for item in ["mobilenet", "resnet18"]:
        sub_list.append({key: value for key, value in collected_data.items() if item in key and
                         any(substring in key for substring in substrs)})

    for item in sub_list:
        copykeys = copy.deepcopy(item)
        for key, _ in copykeys.items():
            if "12" in key:
                item["Synthetic ratio 0.25"] = item.pop(key)
            elif "24" in key:
                item["Synthetic ratio 0.50"] = item.pop(key)
            elif "36" in key:
                item["Synthetic ratio 0.75"] = item.pop(key)
            elif "48" in key:
                item["Synthetic ratio 1"] = item.pop(key)
    
    names = ["MobileNetV2", "ResNet18"]
    for idx, item in enumerate(sub_list):
        plot_data(item, names[idx] + " Synthetic Ratio ", "", os.path.join("Results", "Experiment-Results"), True)

    # Experiments 1b/1c
    sub_list = []
    for item in ["200", "48"]:
        sub_list.append({key: value for key, value in collected_data.items() if item in key})

    for item in sub_list:
        item = convert_labels_classifier(item)
    
    names = ["Train set", "Validation set"]
    for idx, item in enumerate(sub_list):
        plot_data(item, "", " on Synthetic " + names[idx], os.path.join("Results", "Experiment-Results"), True)

def graph_experiment_2():
    check_remove()
    collected_data = extract_data("multiple")

    full_list = []
    for item in ["efficientnet", "mobilenet", "resnet18", "shufflenet"]:
        full_list.append({key: value for key, value in collected_data.items() if item in key})

    for item in full_list:
        # Basing it on the copy here, because you can not
        # edit dict values and then continue the same
        # iteration process, without issues
        copykeys = copy.deepcopy(item)
        for key, _ in copykeys.items():
            if "auto_augment" in key:
                item["AutoAugment"] = item.pop(key)
            elif "categorical" in key:
                item["Categorical"] = item.pop(key)
            elif "rand_augment" in key:
                item["RandAugment"] = item.pop(key)
            elif "random_choice" in key:
                item["RandomChoice"] = item.pop(key)

    names = ["EfficientNetB1", "MobileNetV2", "ResNet18", "ShuffleNetV2"]
    for idx, item in enumerate(full_list):
        plot_data(item, names[idx] + " Augmentations ", "", os.path.join("Results", "Experiment-Results"), True)


def graph_experiment_3():
    check_remove()
    collected_data = extract_data("multiple")
    collected_data = convert_labels_classifier(collected_data)
    plot_data(collected_data, "Classifier ", "", os.path.join("Results", "Experiment-Results"), True)


def graph_experiment_4():
    check_remove()
    collected_data = extract_data("single")

    full_list = []
    for item in ["None", "0."]:
        full_list.append({key: value for key, value in collected_data.items() if item in key})

    for item in full_list:
        item = convert_labels_classifier(item)

    names = [" without GP", " with GP"]
    for idx, item in enumerate(full_list):
        plot_data(item, "DUQ Classifier ", names[idx], os.path.join("Results", "Experiment-Results")) 


def graph_experiment_5():
    check_remove()
    collected_data = extract_data("multiple")
    collected_data = convert_labels_classifier(collected_data)
    plot_data(collected_data, "DUQ Classifier ", "", os.path.join("Results", "Experiment-Results"), True)


def check_remove():
    for file in os.listdir(os.path.join("Results", "Experiment-Results")):
        if file.endswith(".png"):
            os.remove(os.path.join("Results", "Experiment-Results", file))       


def convert_labels_classifier(collected_data):
    copykeys = copy.deepcopy(collected_data)

    for key, _ in copykeys.items():
        if "efficientnet" in key:
            collected_data["EfficientNetB1"] = collected_data.pop(key)
        elif "mobilenet" in key:
            collected_data["MobileNetV2"] = collected_data.pop(key)
        elif "resnet18" in key:
            collected_data["ResNet18"] = collected_data.pop(key)
        elif "shufflenet" in key:
            collected_data["ShuffleNetV2"] = collected_data.pop(key)
    return collected_data


def extract_data(run_type):
    if run_type == "single":
    # This is how it works for a single runs
        event_loc = os.path.join("Results", "Experiment-Results")
        collected_data = {}
        for dir in os.listdir(event_loc):
            data = extract_single_df(os.path.join(event_loc, dir))
            collected_data[dir] = data

    elif run_type == "multiple":
        # This is how it work for multiple runs
        event_loc = os.path.join("Results", "Experiment-Results")
        collected_data = {}
        for multi_dir in os.listdir(event_loc):
            multi_run = []

            for dir in os.listdir(os.path.join(event_loc, multi_dir)):
                if dir == "results.txt":
                    continue
                data = extract_single_df(os.path.join(event_loc, multi_dir, dir))
                multi_run.append(data)
            combined_df = pd.concat(multi_run, ignore_index=True)
            combined_df = merge_dfs(combined_df)
            collected_data[multi_dir] = combined_df
    return collected_data


def merge_dfs(combined_df):
    # Group the data by the "Step" column
    grouped = combined_df.groupby("Step")

    # Calculate the mean and standard deviation for each group (step)
    mean_df = grouped.mean()
    std_df = grouped.std()

    # Rename the columns for clarity
    mean_df.rename(columns = {"Validation Accuracy": "Mean Validation Accuracy",
                              "Validation Loss": "Mean Validation Loss"},
                               inplace = True)
    std_df.rename(columns = {"Validation Accuracy": "Std Validation Accuracy",
                             "Validation Loss": "Std Validation Loss"},
                             inplace = True)

    # Merge the mean and standard deviation DataFrames
    result_df = pd.concat([mean_df, std_df], axis=1)

    # Reset the index if needed
    result_df.reset_index(inplace=True)

    # Clipping std deviation values, since somehow they can go over 1
    # when combined with the accuracy
    error_rows = result_df[result_df["Mean Validation Accuracy"] + result_df["Std Validation Accuracy"] > 1]
    result_df.loc[error_rows.index, "Std Validation Accuracy"] = 1 - error_rows["Mean Validation Accuracy"]
    # Not checked, but the same might occur for going lower than 1 for
    # The loss and the accuracy as well
    error_rows = result_df[result_df["Mean Validation Accuracy"] - result_df["Std Validation Accuracy"] < 0]
    result_df.loc[error_rows.index, "Std Validation Accuracy"] = result_df["Mean Validation Accuracy"]
    # And for the loss
    error_rows = result_df[result_df["Mean Validation Loss"] - result_df["Std Validation Loss"] < 0]
    result_df.loc[error_rows.index, "Std Validation Loss"] = result_df["Mean Validation Loss"]

    return result_df


def extract_single_df(event_loc):
    val_event = os.listdir(os.path.join(event_loc, "val"))[0]
    val_event = os.path.join(event_loc, "val", val_event)

    # Create an EventFileLoader
    event_file_loader = tf.compat.v1.train.summary_iterator(val_event)

    # Initialize empty lists to store data
    val_acc = []
    val_loss = []
    # Iterate through event files and extract data
    for event_file in event_file_loader:
        for event in event_file.summary.value:
            if event.tag == "Validation Accuracy":
                val_acc.append(event.simple_value) 
            if event.tag == "Validation Loss":
                val_loss.append(event.simple_value)

    # Create a DataFrame from the extracted data
    step = [i for i in range(len(val_acc))]
    data = pd.DataFrame({'Step': step, 'Validation Accuracy': val_acc, 'Validation Loss': val_loss})
    return data


def plot_data(collected_data, title1, title2, path, multi = False):
    # Setting plotstyle and getting df length
    sns.set(style="darkgrid")
    iterable = iter(collected_data.items())
    df_len = next(iterable)[1].shape[0]

    # Validation accuracy plot
    # First plotting and then configuring pyplot
    if multi:
        for key, value in collected_data.items():
           sns.lineplot(data = value, x = "Step", y = "Mean Validation Accuracy",
                        label = key)
           plt.fill_between(value["Step"],
                            -1 * value["Std Validation Accuracy"] + value["Mean Validation Accuracy"],
                            value["Std Validation Accuracy"] + value["Mean Validation Accuracy"], alpha = 0.3)
    else:
        for key, value in collected_data.items():
           sns.lineplot(data = value, x = "Step", y = "Validation Accuracy", label = key)
    title = title1 + "Validation Accuracy" + title2
    plt.xlim(0, df_len - 1)
    if df_len == 20:
        plt.xticks(range(0, df_len - 1, 3))
    plt.ylim(0, 1.1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.savefig(os.path.join(path, title), dpi = 300)
    plt.clf()

    # Validation loss plot
    if multi:
        for key, value in collected_data.items():
           sns.lineplot(data = value, x = "Step", y = "Mean Validation Loss",
                        label = key)
           plt.fill_between(value["Step"],
                            -1 * value["Std Validation Loss"] + value["Mean Validation Loss"],
                            value["Std Validation Loss"] + value["Mean Validation Loss"], alpha = 0.3)
    else:
        for key, value in collected_data.items():
           sns.lineplot(data = value, x = "Step", y = "Validation Loss", label = key)
    title = title1 + "Validation Loss" + title2
    plt.xlim(0, df_len - 1)
    if df_len == 20:
        plt.xticks(range(0, df_len - 1, 3))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.savefig(os.path.join(path, title), dpi = 300)
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type = str)
    args = parser.parse_args()
    start = time.time()

    if args.experiment == "experiment_1":
        # Time estimate (no new synthetic data/1 run): 60 minutes
        # Time estimate (no new synthetic data/10 runs)): 2 hours 48 minutes
        #experiment_1()
        graph_experiment_1()
    elif args.experiment == "experiment_2":
        # Time estimate (no new synthetic data/1 run): 60 minutes
        # Time estimate (no new synthetic data/10 runs): 5 hours 18 minutes
        experiment_2()
        graph_experiment_2()
    elif args.experiment == "experiment_3":
        # Time estimate (no new synthetic data/1 run): 18 minutes
        # Time estimate (no new synthetic data/10 runs): 3 hours
        experiment_3()
        graph_experiment_3()
    elif args.experiment == "experiment_4":
        # Time estimate: 19 hours
        experiment_4()
        graph_experiment_4()
    elif args.experiment == "experiment_5":
        # Time estimate: 4 hours
        # Time estimate (3 runs): 11 hours 30 minutes
        experiment_5()
        graph_experiment_5()
    elif args.experiment == "experiment_6":
        experiment_6()
    
    elapsed_time = time.time() - start
    print("Total time for " + args.experiment + " (H/M/S) = ", 
          time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
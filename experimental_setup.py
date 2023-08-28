import argparse
import json
import os
import re

# TODO: Remove duplicate code and move to functions
# TODO: EXPERIMENT 1 & EXPIMERENT 6

if __name__ == '__main__':
    ex_path = "Experiments"
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type = str)
    args = parser.parse_args()

    if args.experiment == "experiment_1":
        print("Coming soon...")
        # Experiment 1: Synthetic data study on the NTZFilter Dataset.
        # Experiment 1a: Adding synthetic data in different proportions to
        # ResNet18 and MobileNetV2 (or all of them?)
        # Since the original dataset is about 70 per class, in this case
        # the synthetic data samples should not exceed 70 per class.
        # Experiment 1b: Classifier performance on only training on synthetic
        # data and validating on a real set, including model feature analysis
        # through integrated gradients.
        # In this experiment the synthetic set can be larger
        # Experiment 1c: Classifier performance on only training on a real set
        # and validating on a synthetic set, including model feature analysis
        # through integrated gradients.
        # In this experiment the synthetic set can be larger

    if args.experiment == "experiment_2":
        # Experiment 2: Augmentation testing - Compare augmentation techniques
        # per classifier on the NTZFilter dataset.
        n_runs = 5
        classifiers = ["mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)",
                       "resnet18(weights = ResNet18_Weights.DEFAULT)",
                       "shufflenet_v2_x1_0(weights = ShuffleNet_V2_X1_0_Weights.DEFAULT)",
                       "efficientnet_b1(weights = EfficientNet_B1_Weights.DEFAULT)"]
        augmentations = ["rand_augment", "categorical", "random_choice", "auto_augment",
                         "random_apply"]
        
        # Run the experiment
        for classifier in classifiers:
            for augment in augmentations:
                with open(os.path.join(ex_path, "experiment_2.json")) as ex_file:
                    data = json.load(ex_file)
                data["model"] = classifier
                data["augmentation"] = augment

                # Name the experiment and remove the parentheses part
                ex_name = f"experiment_2_{classifier}_{augment}.json"
                re_pattern = r'\(.*\)'
                ex_name = re.sub(re_pattern, '', ex_name)
                with open(os.path.join(ex_path, ex_name), 'w') as temp_file:
                    json.dump(data, temp_file)

                # Call the train file with the json experiment file
                os.system("python3.10 train.py " + ex_name.replace(".json", "") + " --n_runs " + str(n_runs))

                # Delete the JSON file
                try:
                    os.remove(os.path.join(ex_path, ex_name))
                    print(f"{ex_name} has been deleted.")
                except FileNotFoundError:
                    print(f"{ex_name} does not exist.")


    if args.experiment == "experiment_3":
        # Experiment 3: Classifier testing - Comparing classifiers with the best
        # augmentation technique per classifier. Include classifier speeds
        # (also TRT vs. no TRT), memory usage, computing power in GFLOPS.
        # Include a feature analysis through integrated gradients.
        # -> Run TRT vs. no TRT speeds manually
        # -> Run GFLOPS calculation manually

        # TODO: Fill in the best augmentation techniques per classifier here, from expeirment 2
        combs = [["mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)", "rand_augment"],
                 ["resnet18(weights = ResNet18_Weights.DEFAULT)", "rand_augment"],
                 ["shufflenet_v2_x1_0(weights = ShuffleNet_V2_X1_0_Weights.DEFAULT)", "rand_augment"],
                 ["efficientnet_b1(weights = EfficientNet_B1_Weights.DEFAULT)", "rand_augment"]]
        
        for comb in combs:
            classifier = comb[0]
            augment = comb[1]
            with open(os.path.join(ex_path, "experiment_3.json")) as ex_file:
                data = json.load(ex_file)
            data["model"] = classifier
            data["augmentation"] = augment

            # Name the experiment and remove the parentheses part
            ex_name = f"experiment_3_{classifier}_{augment}.json"
            ex_name_rm = ex_name.replace(".json", "")
            re_pattern = r'\(.*\)'
            ex_name = re.sub(re_pattern, '', ex_name)
            with open(os.path.join(ex_path, ex_name), 'w') as temp_file:
                json.dump(data, temp_file)

            # Call the train file with the json experiment file
            os.system("python3.10 train.py " + ex_name_rm + " --n_runs " + str(n_runs))

            # 2nd part running integrated gradients
            # Since the folder is saved with a timestamp, we need to find it first
            all_directories = os.listdir(os.path.join("Results", "Experiment-Results"))
            for directory in all_directories:
                if directory.startswith(ex_name_rm):
                    matching_directory = directory
                    break

            os.system("python3.10 explainability.py " + matching_directory + " Captum")

            # Delete the JSON file
            try:
                os.remove(os.path.join(ex_path, ex_name))
                print(f"{ex_name} has been deleted.")
            except FileNotFoundError:
                print(f"{ex_name} does not exist.") 


    if args.experiment == "experiment_4":
        # Experiment 4: DUQ analysis on CIFAR10 dataset
        # (this experiment can perhaps be fully omitted?)
        # Experiment 4a: Classifier performance on CIFAR10 dataset
        # when DUQ converted (with GP)
        # Experiment 4b: Classifier performance on CIFAR10 dataset
        # when DUQ converted (without GP)
        n_runs = 1
        combs = [["mobilenet_v2()", ["0.1", "0"]],
                 ["resnet18()", ["0.5", "0"]],
                 ["shufflenet_v2_x1_0()", ["0.1", "0"]],
                 ["efficientnet_b1()", ["0.1", "0"]]]
        
        for comb in combs:
            classifier = comb[0]
            for gp in comb[1]:
                with open(os.path.join(ex_path, "experiment_4.json")) as ex_file:
                    data = json.load(ex_file)
                data["model"] = classifier
                data["gp_const"] = gp

                 # Name the experiment and remove the parentheses
                ex_name = f"experiment_4a_{classifier}_{gp}.json"
                re_pattern = r'\(.*\)'
                ex_name = re.sub(re_pattern, '', ex_name)
                with open(os.path.join(ex_path, ex_name), 'w') as temp_file:
                    json.dump(data, temp_file)

                # Call the train file with the json experiment file
                os.system("python3.10 train_rbf.py " + ex_name.replace(".json", "") + " --n_runs " + str(n_runs))

                # Delete the JSON file
                try:
                    os.remove(os.path.join(ex_path, ex_name))
                    print(f"{ex_name} has been deleted.")
                except FileNotFoundError:
                    print(f"{ex_name} does not exist.")


    if args.experiment == "experiment_5":
        # Experiment 5: DUQ analysis on NTZFilter dataset - Classifier 
        # performance on NTZFilter dataset, when DUQ converted
        # Report model speeds, as well as a feature analysis when using DUQ.
        # Show image examples reporting class as well as uncertainty.
        # -> Run speeds manually (no TRT since it does not work)

        # TODO: What about augmentations per experiment? Should it be the optimal one found in experiment 2?
        n_runs = 1
        combs = [["mobilenet_v2()", "lr = 0.05"],
                 ["resnet18()", "lr = 0.01"],
                 ["shufflenet_v2_x1_0()", "lr = 0.05"],
                 ["efficientnet_b1()", "lr = 0.01"]]
        
        for comb in combs:
            classifier = comb[0]
            lr = comb[1]
            with open(os.path.join(ex_path, "experiment_5.json")) as ex_file:
                data = json.load(ex_file)
            data["model"] = classifier
            optimizer = data["optimizer"]
            data["optimizer"] = optimizer.replace("lr = 0.05", lr)

            # Name the experiment and remove the parentheses part
            ex_name = f"experiment_5_{classifier}_{lr}.json"
            ex_name_rm = ex_name.replace(".json", "")
            re_pattern = r'\(.*\)'
            ex_name = re.sub(re_pattern, '', ex_name)
            with open(os.path.join(ex_path, ex_name), 'w') as temp_file:
                json.dump(data, temp_file)

            # Call the train file with the json experiment file
            os.system("python3.10 train_rbf.py " + ex_name_rm + " --n_runs " + str(n_runs))

            # 2nd part running integrated gradients
            # Since the folder is saved with a timestamp, we need to find it first
            all_directories = os.listdir(os.path.join("Results", "Experiment-Results"))
            for directory in all_directories:
                if directory.startswith(ex_name_rm):
                    matching_directory = directory
                    break

            os.system("python3.10 explainability.py " + matching_directory + " Captum")

            # Delete the JSON file
            try:
                os.remove(os.path.join(ex_path, ex_name))
                print(f"{ex_name} has been deleted.")
            except FileNotFoundError:
                print(f"{ex_name} does not exist.")


    if args.experiment == "experiment_6":
        print("Coming soon...")
        # Experiment 6: (Perhaps omit this as well?) Edge case analysis.
        # Experiment 6a: Give a DUQ model a sample that is from a completely
        # different dataset (out of distribution), see what uncertainty is given.
        # Experiment 6b: Add noise to a testing image, see what uncertainty is given.
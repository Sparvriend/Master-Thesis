# Master-Thesis-Code-Repository

## Current TODO
* TODO: Finish explainability of the model  
      -> Make the explainability images overlay the original image with a better colormap  
      -> Uncertainty prediction (DUQ)  
      -> Deep Ensembles  
      -> Combine other dataset (CIFAR10) with uncertainty prediction  
      -> Look into adding augmentation to uncertainty methods  
* TODO: Figure out why the testing FPS is so slow (Overhead probably)  
      -> Report only GPU fps  
      -> Appears to be as fast as doing it on the CPU? (is device CPU?)  
* TODO: Look into TensorRT library, Ratnajit: Useful in addition to torch2trt?  
      -> (https://github.com/NVIDIA/TensorRT)  
      -> Read about TRT/ONNX why is it faster? (Important to understand theoretically)  
* TODO: Look into test.py converting to TRT out of memory issues  
* TODO: Do a rerun of PEP8 in all files  
* TODO: Check if cutoff_date can be used in other spots in the code
* TODO: Add model marking to test predictions like in explainability.py  
* TODO: Remove red colormap/blended representation in explainability.py?  
* TODO: Fix test.py arguments, --explain_model should be optional, with "integrated_gradients" default value  
      -> other explain_model options should be taken from command line input  
* TODO: Report GPU memory usage/Energy usage (KJ) (NVIDIA management library, code from Ratnajit)  
* TODO: Create synthetic data  
       -> for each class, move the filter across the screen and the label across the filter (where applicable)  
* TODO: Setup experiment with tinyImageNet  
* TODO: Text detection model for fourth label class?  
      -> Manual labelling of each date on the filter
      -> Check if the date listed on the label is correct

### Tensorboard tip

To open tensorboard in browser, run the following command in a new terminal: tensorboard --logdir=Master-Thesis-Experiments

## JSON configuration files

The JSON configuration files are used to run experiments with different hyperparameter setups. Each of the arguments in the JSON files are explained here:

    Model: The model the experiment is run with, ususally MobileNetV2.
    Criterion: Experiment criterion for calculating the loss, usually CrossEntropyLoss
    Optimizer: The optimizer used to update the weights of the model in backpropagation, usually SGD.
    Scheduler: The learning rate scheduler used for updating the learning rate of the model. Usually MultiStepLR.
    Epochs: The maximum amount of epochs a single model can be trained for.
    Batch size: The amount of training instances in a single batch.
    Shuffle: Boolean to decided whether the training data should be shuffled before each epoch.
    Num workers: The amount of workers used for loading the data.
    Augmentation: The type of augmentation to use for online augmentation of the images. Choice of rand_augment, categorical, random_choice, auto_augment, no_augment and random_apply.
    PFM Flag: Performance metric recording flags, consists of a dictionary with two keys, Terminal, for printing results to the terminal and Tensorboard for recording results with tensorboard.
    Early Limit: The maximum amount of epochs the model can be trained for without an improvement in the validation accuracy, before stopping early. If set to 0, early stopping is disabled
    Replacement Limit: The maximum amount of epochs a model can be trained for without an improvement in the validation accuracy, before the model is replaced with the previous best model. If set to 0, model replacement is disabled.

## Code explanation

### train.py

train.py can be used to train a model. It can take three types of inputs as command line arguments. The first method is to give a JSON experiment configuration file, from the Master-Thesis-Experiments folder, it is not necessary to combine the path, e.g. Experiments/*.json. The second method is to not give it a JSON experiment configuration file, e.g. giving it no arguments. In doing so all JSON files for training will be selected and used for experimenting three times. The third method is to give it a JSON experiment configuration file, like in method 1, but to follow it by an integer, which is the amount of times the experiment will be run.

### test.py

test.py can be used to test a model. It takes one type of input: an experiment folder name created by running train.py that should contain a .pth file that contains the model. The model is loaded and then used to predict the test data. The test data is annotated with red text that indicates the class that the model thinks the image belongs to.

### test_augmentation.py

test_augmentation.py can be used to get performance results for each augmentation method used for training the models, one by one. It does not take inputs, and takes about 1 hour and a half to run on a RTX 3080. It creates directories for each augmentation type and prints a results.txt file in each directory with an averaged accuracy result per augmentation type.

### data_processing.py

data_processing.py can be used to get data from a folder called "NTZ_filter_label_data" and divides it up into classes with training, validation and testing data.
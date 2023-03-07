# Master-Thesis-Code-Repository

## JSON configuration files

The JSON configuration files are used to run experiments with different hyperparameter setups. Each of the arguments in the JSON files are explained here:

    Model: The model the experiment is run with, ususally MobileNetV2 or ShuffleNetV2.
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

train.py can be used to train a model. It can take three types of inputs as command line arguments. The first method is to give a JSON experiment configuration file, from the Master-Thesis-Experiments folder, it is not necessary to combine the path, e.g. Master-Thesis-Experiments/*.json. The second method is to not give it a JSON experiment configuration file, e.g. giving it no arguments. In doing so all JSON files for training will be selected and used for experimenting three times. The third method is to give
it a JSON experiment configuration file, like in method 1, but to follow it by an integer, which is the amount of times the experiment will be run.

### test.py

test.py can be used to test a model. It takes one type of input: an experiment folder name created by running train.py that should contain a .pth file that contains the model. The model is loaded and then used to predict the test data. The test data is annotated with red text that indicates the class that the model thinks the image belongs to.

### test_augmentation.py

test_augmentation.py can be used to get performance results for each augmentation method used for training the models, one by one. It does not take inputs, and takes about 1 hour and a half to run on a RTX 3080. It creates directories for each augmentation type and prints a results.txt file in each directory with an averaged accuracy result per augmentation type.

### data_processing.py

data_processing.py can be used to get data from a folder called "NTZ_filter_label_data" and divides it up into classes with training, validation and testing data.
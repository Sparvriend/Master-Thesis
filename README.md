# Master-Thesis-Code-Repository

## TODO in order of priority
* TODO: Remove all directories, except for raw_data and go through all files to see if it works (for example update create_dirs() function in data_processing.py)  
* TODO: Update README  
* TODO: Go through repository and cleanup (incl PEP8).  
* TODO: Do PEP8 in experimental_setup.py - its very dirty.  
* TODO: Remove DEU in explainability.py (it is not used)  
* TODO: Check if data_processing.py works correctly, when using the data as instructed  

## Optional TODOs
* OPTIONAL: Do a rerun of PEP8 in all files    
* OPTIONAL: Test all code with laptop to see if it works without CUDA  

### Tensorboard tip

To open tensorboard in browser, run the following command in a new terminal: python3.10 -m tensorboard.main --logdir=Results/Experiment-Results

## JSON configuration files

The JSON configuration files are used to run experiments with different hyperparameter setups. See DEFAULT.json for the default hyperparamter setup. When running an experiment with train.py, train_rbf.py or experimental_setup.py, any experiment can be used, arguments in experiment json files overwrite those from the DEFAULT json file.

JSON configuration file arguments:

    model: The model the experiment is run on, one of [MobileNetV2, ResNet18, EfficientNetB1, ShuffleNetV2], the framework will not work with others, model definitions should include if pretrained weights are used or not.
    criterion: Experiment criterion for calculating the loss, usually CrossEntropyLoss.
    optimizer: The optimizer used to update the weights of the model in backpropagation, usually SGD.
    scheduler: The learning rate scheduler used for updating the learning rate of the model. Usually MultiStepLR.
    Dataset: The dataset used for training the model, one of [NTZFilterDataset, CIFAR10 or NTZFilterSyntheticDataset].
    epochs: The maximum amount of epochs a single model can be trained for.
    batch_size: The amount of training instances in a single batch.
    augmentation: The type of augmentation to use for online augmentation of the images. See utils.py get_transforms() function for transform options.
    PFM_flag: Performance metric recording flags, consists of a dictionary with two keys, Terminal, for printing results to the terminal and Tensorboard for recording results with tensorboard.
    RBF_flag: Deterministic Uncertainty Quantification (DUQ) model flag, if this flag is True, the model is converted to a version that supports DUQ.
    gp_const: DUQ gradient penalty constant, if None, no gradient penalty is applied.
    early_imit: The maximum amount of epochs the model can be trained for without an improvement in the validation accuracy, before stopping early. If set to 0, early stopping is disabled
    replacement_limit: The maximum amount of epochs a model can be trained for without an improvement in the validation accuracy, before the model is replaced with the previous best model. If set to 0, model replacement is disabled.

## Runnable files

### data_processing.py

data_processing.py can be used to get raw data (in raw_data folder) from a folder called "NTZ_filter_label_data" and divides it up into classes with training, validation and testing data. It is also used to setup the CIFAR10 dataset and it contains a function that creates all required directories for results saving. If an error occurs due to a directory missing, then it is likely that create_dirs() has not been run

### train.py

train.py can be used to train a model. It can take three types of inputs as command line arguments. The first method is to give a JSON experiment configuration file, from the Master-Thesis-Experiments folder, it is not necessary to combine the path, e.g. Experiments/*.json. The second method is to not give it a JSON experiment configuration file, e.g. giving it no arguments. In doing so all JSON files for training will be selected and used for experimenting three times. The third method is to give it a JSON experiment configuration file, like in method 1, but to follow it by an integer, which is the amount of times the experiment will be run.

### train_rbf.py

explanation here...

### experimental_setup.py

explanation here...

### test.py

test.py can be used to test a model. It takes one type of input: an experiment folder name created by running train.py that should contain a .pth file that contains the model. The model is loaded and then used to predict the test data. The test data is annotated with red text that indicates the class that the model thinks the image belongs to.

### explainability.py

explanation here...

### GAN.py

explanation here...

### synthetic_data.py

explanation here...

### utils.py

explanation here...

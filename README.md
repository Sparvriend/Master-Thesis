# Fast Industrial Image Classification using Lightweight Deep Networks

## Thesis Abstract
Road cars use special fabric filters to filter dirt and debris from transmission fluids, preventing transmission wear and unsafe traffic situations. During the filter production process, information is printed on the filters, which can go wrong in a number of ways, leading to a classic image classification problem. In this thesis, deep learning and computer vision techniques are utilized to analyse the types of faults and classify them as such. The classification problem is hampered by limited data availability, limited computational power being available and classification speed requirements, hence warranting the usage of lightweight deep learning classifiers, such as ResNet18. The thesis consists of three main parts: 1. Finding an optimal lightweight deep learning classifier. 2. Expanding the dataset distribution. 3. Forming model interpretability and representing model prediction confidence. To find a suitable deep learning classifier, multiple state of the art lightweight models were compared. To expand the given dataset, a handcrafted algorithm was developed to produce synthesized data, increasing data variance. To form model interpretability, feature analysis techniques, such as integrated gradients, were used to analyse if the models are focusing on the right features when making predictions. Additionally, to further increase model interpretability, model prediction confidence scores were measured by incorporating a radial basis function as the model's final layer, in accordance with deterministic uncertainty quantification methodology. Based on the results it is possible to conclude that the synthesized data expands the data variance, the faults in the label printing system can be successfully recognized by the models, reaching validation accuracy values of close to 100%, with feature analysis and model predictions confidence being useful tools for model interpretability.

This repository contains the code utilized during the project to run the experiments. The thesis report is available on request. The project was developed using Python version 3.10 and CUDA framework for running models on the GPU.

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

All the following files are runnable/usable to achieve some parts of the results. A logical order of running things would be to download the CIFAR10 dataset and save it to a folder called data/CIFAR10, then run data_processing.py. After that, experimental_setup.py can be run to get the results from the experiments as well as graphs. Additonally, the other files are also runnable as smaller submodules.

### data_processing.py

data_processing.py can be used to get raw data (in raw_data folder) from a folder called "NTZ_filter_label_data" and divides it up into classes with training, validation and testing data. It is also used to setup the CIFAR10 dataset and it contains a function that creates all required directories for results saving.

### train.py

train.py can be used to train a model. It takes a JSON experiment configuration file as an argument, from the Experiments folder. It is not necessary to combine the path, e.g. Experiments/*.json. An additional argument --n_runs can be used to run the experiment multiple times.

### train_rbf.py

Similar to train.py, but for RBF (networks converted to DUQ) versions. It takes the same inputs as train.py

### experimental_setup.py

experimental_setup.py is used to run experiments as described in the thesis report in its entirity. This includes doing multiple runs, replicating the hyperparamters used in the experiments exactly and plotting graphs. It takes an input of "experiment_x", where x can vary between 1 and 5.

### test.py

test.py can be used to test a model. It takes one type of input: an experiment folder name created by running train.py or train_rbf.py that should contain a .pth file that contains the model. The model is loaded and then used to predict the test data. The test data is annotated with red text that indicates the class that the model thinks the image belongs to. Additionally, the model can be converted to a trt version and the speed for the classifier can be calculated, by flags --convert_trt and --calc_speed respectively.

### explainability.py

explainability.py is used to run feature analysis for a model. It takes one type of input, the experiment folder name, created by running train.py or train_rbf.py. An additional argument, --explainability_variant can be provided with a choice of ["integrated_gradients", "saliency_map", "deeplift", "guided_backpropagation"], that can change the type of feature analysis that is done.

### GAN.py

GAN.py is used to generate synthetic data for the NTZ filter dataset. It takes one input and three optional inputs. The input is the train type, either being seperate for each class or combined. The optinal inputs are --n_imgs, --latent_vector_size and --epochs, which detail the amount of images that will be generated per class, the size of the latent vector and the amount of epochs the GAN is trained for. 

### synthetic_data.py

synthetic_data.py is used to generate synthetic data for the NTZ filter dataset procedurally. Refer to the file itself for explanations on the arguments to run it.

### utils.py

utils.py contains a number of helper functions used across the experiments. When ran, it will calculate the amount of GFLOPS for each model used in this project (standard version, no DUQ).

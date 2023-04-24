# Master-Thesis-Code-Repository

## TODO in order of priority
* TODO: Create synthetic data  
       -> Possibility: Procedural (in synthetic_data.py)  
            -> TODO: Fix angled rectangle mapping  
            -> Alternative: Use a model like MASK-RCNN to detect filter bounding boxes  
       -> Possibility: GANs -> Combine with CIFAR100?  
            -> https://github.com/eriklindernoren/PyTorch-GAN  
            -> https://github.com/ajbrock/BigGAN-PyTorch  
       -> Possibility: Blender  
       -> Possibility: Diffusion models  
* TODO: Finish explainability of the model  
      -> Uncertainty prediction (DUQ)  
          -> Think of a method of expressing the distance as uncertainty  
                -> Other option: Define a model, define a RBF function and replace each Relu layer in the network with a rbf layer.
                Information on how to replace all relu layers in the network by some other activation layer can be found here:
                https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/11
                -> Create a function that takes a model as input, as well as a list of or single layer to replace in the model
                and what to create it by. The function should then replace the layers in the model with the new layers.
                This way any of the models that have been used so far can be used with DUQ. -> Place the RBF conversion function in explainability.py in the DUQ function.
                -> It seems as if Matias in his email is saying that only the final Relu layer needs replacing, which is a much easier option.
                -> Only use the RBF function in the network during interference time
          -> Look into if KL-Divergence is a better metric than euclidean distance  
              -> https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html  
              -> https://machinelearningmastery.com/divergence-between-probability-distributions/  
      -> Deep Ensembles Uncertainty    
      -> Look into adding augmentation to uncertainty methods  
      -> Convert Saliency Map to RGB  
* TODO: Report GPU memory usage/Energy usage (KJ) (NVIDIA management library, code from Ratnajit)  
* TODO: Replace upsampling with InterpolationMode.BICUBIC (Transforms.resize(256) with method in utils)
* TODO: Fix test.py arguments, --explain_model should be optional, with "integrated_gradients" default value  
      -> other explain_model options should be taken from command line input           
* TODO: Figure out why the testing FPS is so slow (Overhead probably)  
      -> Report only GPU fps  
      -> Appears to be as fast as doing it on the CPU? (is device CPU?)  
* TODO: Look into TensorRT library, Ratnajit: Useful in addition to torch2trt?  
      -> (https://github.com/NVIDIA/TensorRT)  
      -> Read about TRT/ONNX why is it faster? (Important to understand theoretically)  
* TODO: Look into test.py converting to TRT out of memory issues   
* TODO: Text detection model for fifth label class?  
      -> Problem: Check if the date corresponds to an input date (current date)  
      -> Manual labelling of each date on the filter  
      -> East Detector  
      -> Do a short visibility study if it turns out to be too complicated  

## Optional TODOs
* OPTIONAL: Do a rerun of PEP8 in all files 
* OPTIONAL: Make a jupyter notebook implementation of some of the code, so that is can easily be shown off to others  
* OPTIONAL: Use 3D plots with color to represent results     
* OPTIONAL: Test all code with laptop to see if it works without CUDA  

### Tensorboard tip

To open tensorboard in browser, run the following command in a new terminal: tensorboard --logdir=Master-Thesis-Experiments


### Class definition for the NTZFilterDataset

    The class labels are defined as:
    0: fail_label_crooked_print
    1: fail_label_half_printed
    2: fail_label_not_fully_printed
    3: no_fail
    4: fail_label_date (Not yet incorporated)

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

### data_processing.py

data_processing.py can be used to get data from a folder called "NTZ_filter_label_data" and divides it up into classes with training, validation and testing data. It is also used to setup the CIFAR10 dataset and it contains a function that creates all required directories for results saving. If an error occurs due to a directory missing, then it is likely that create_dirs() has not been run

### train.py

train.py can be used to train a model. It can take three types of inputs as command line arguments. The first method is to give a JSON experiment configuration file, from the Master-Thesis-Experiments folder, it is not necessary to combine the path, e.g. Experiments/*.json. The second method is to not give it a JSON experiment configuration file, e.g. giving it no arguments. In doing so all JSON files for training will be selected and used for experimenting three times. The third method is to give it a JSON experiment configuration file, like in method 1, but to follow it by an integer, which is the amount of times the experiment will be run.

### test.py

test.py can be used to test a model. It takes one type of input: an experiment folder name created by running train.py that should contain a .pth file that contains the model. The model is loaded and then used to predict the test data. The test data is annotated with red text that indicates the class that the model thinks the image belongs to.

### test_augmentation.py

test_augmentation.py can be used to get performance results for each augmentation method used for training the models, one by one. It does not take inputs, and takes about 1 hour and a half to run on a RTX 3080. It creates directories for each augmentation type and prints a results.txt file in each directory with an averaged accuracy result per augmentation type.
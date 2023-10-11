import argparse
import os
import time
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score
import torchvision
from tqdm import tqdm
from types import SimpleNamespace

from utils import get_transforms, setup_tensorboard, get_data_loaders, \
                  report_metrics, find_classification_module, setup_hyp_dict, \
                  setup_hyp_file, get_device, merge_experiments, calculate_acc_std


class RBF_model(nn.Module):
    """RBF layer definition based on Joost van Amersfoort's implementation
    of Determenistic Uncertainty Quantification (DUQ):
    https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/utils/resnet_duq.py
    and further inspired by Matias Valdenegro Toro implementation:
    https://github.com/mvaldenegro/keras-uncertainty/blob/master/keras_uncertainty/layers/rbf_layers.py

    Variable explanations:
    out_features is the amount of classes of the model.
    in_features is the amount of features inserted into an RBF layer by the model.

    [kernels] holds the representation of conversion to a feature space.
    Any time an output of the feature extractor is calculated it is first
    matrix multiplied (einsum) with the kernels to get a feature space
    representation of the feature extractor output. A parameter of the
    model, hence updated every backwards pass.
    Shape = [in features, classes, in features]

    [N] holds the label counts multiplied by the constant gamma. In essence
    it holds the frequency of each label relative to the other labels.
    Shape = [classes]

    [m] holds the centroid sum multiplied by the constant gamma. The centroid sum
    consists of feature extractor output, combined through matrix multiplication
    (einsum) with the kernels. The result is then again combined (einsum) with 
    the labels to get a sum of the feature extractor output for each label;
    the centroid sum.
    Shape = [in features, classes]

    [m / N] Gives the centroids, it applies the relative label frequency
    of N to m.
    Shape = [in features, classes]

    It is assumed that the labels used for updating the centroids are
    one-hot encoded, for usage in BCE loss.

    Relevant variables for a forwards pass (N/m) and the kernels
    are saved as part of the model by doing torch.save and later loading
    with torch.load.

    The essence of DUQ is that it learns a set of centroids for each class,
    which it can then compare to new inputs during inference time. The
    distance to the closest centroid is the uncertainty metric.
    """
    def __init__(self, fe, in_features, out_features, device, gp_const, batch_size):
        super(RBF_model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = 0.1
        self.gamma = 0.999
        self.device = device
        self.gp_const = gp_const

        # Initializing kernels centroid embedding
        # Set the feature extractor after the initialization of the kernels!
        # otherwise the model does not learn the centroids at all
        self.kernels = nn.Parameter(torch.Tensor(in_features, out_features,
                                                 in_features))
        nn.init.kaiming_normal_(self.kernels, nonlinearity = 'relu')
        self.fe = fe

        self.N = (torch.zeros(out_features) + (batch_size/out_features)).to(device)
        self.m = torch.zeros(in_features, out_features).to(device)
        self.m *= self.N
        
        nn.init.normal_(self.m, 0.05, 1)
 
    
    def forward(self, x):
        # Getting feature output from fe and then applying kernels
        z = self.fe(x)
        z = torch.einsum("ij, mnj->imn", z, self.kernels)

        # Getting embedded centroids
        c = (self.m / self.N.unsqueeze(0)).unsqueeze(0).to(self.device)

        # Getting distances to each centroid
        distances = ((z - c) ** 2).mean(1) / (2 * self.sigma ** 2)

        # With Gaussian distribution
        distances = torch.exp(-1 * distances)
        return distances


    def update_centroids(self, inputs, labels):
        # Defining update function
        update_f = lambda x, y: self.gamma * x + (1 - self.gamma) * y

        # Updating N with summed labels
        self.N = update_f(self.N, labels.sum(0))
        z = self.fe(inputs)

        # Calculating centroid sum
        z = torch.einsum("ij, mnj->imn", z, self.kernels)
        centroid_sum = torch.einsum("ijk, ik->jk", z, labels)

        # Updating m
        self.m = update_f(self.m, centroid_sum)


    def get_gradients(self, inputs, model_output):
        """Function that calculates a gradients for model inputs,
        given the predicted output.

        Args:
            inputs: Model inputs.
            model_output: Predicted labels given input.
        """
        gradients = torch.autograd.grad(outputs = model_output, inputs = inputs,
                                        grad_outputs = torch.ones_like(model_output),
                                        create_graph = True)[0]
        return gradients.flatten(start_dim = 1)


    def get_grad_pen(self, inputs, model_output):
        """Function that calculates the gradient penalty
        based on the gradients of the inputs, its L2 norm,
        applying the two sided penalty and the gradient
        penalty constant. Taken from Joost van Amersfoort
        paper on DUQ (2020).

        Args:
            inputs: Model inputs.
            model_output: Predicted labels given input.
        """
        # First getting gradients
        gradients = self.get_gradients(inputs, model_output)

        # Then computing L2 norm (2 sided)
        L2_norm = gradients.norm(2, dim = 1)

        # Applying the 2 sided penalty
        grad_pen = ((L2_norm - 1) ** 2).mean()

        return grad_pen * self.gp_const


def set_rbf_model(model: torchvision.models, classes: int, device: torch.device,
                  gp_const: float, batch_size: int):
    """This function takes a Pytorch deep Learning model, finds its
    classification layer and then converts that to an RBF layer.

    Args:
        model: Pytorch deep learning model.
        classes: Amount of classes in the dataset.
        device: Device to run the model on.

    Returns:
        Model adapted to an RBF version
    """
    name, module, idx = find_classification_module(model)
    in_features = module.in_features
    if idx != None:
        model._modules[name][idx] = torch.nn.Identity()
    else:
        model._modules[name] = torch.nn.Identity()
    model = RBF_model(model, in_features, classes, device, gp_const, batch_size)
    return model


def preprocess_model(model: torchvision.models):
    """Function that takes care of converting convolutional
    and maxpooling layers to appropriate versions for usage
    in DUQ. This should only happen if the dataset is
    CIFAR10, since the models reduce the images to too small
    size, CIFAR10 is already small (32x32).
    
    Args:
        model: Pytorch deep learning model.
    
    Returns:
        Model adapted appropriate for DUQ.
    """
    if model.__class__.__name__ == "ResNet":
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        model.maxpool = torch.nn.Identity()
    if model.__class__.__name__ == "MobileNetV2":
       model.features[0][0] = torch.nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
    if model.__class__.__name__ == "EfficientNet":
        model.features[0][0] = torch.nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
    if model.__class__.__name__ == "ShuffleNetV2":
        model.conv1[0] = torch.nn.Conv2d(3, 24, kernel_size = 3, stride = 1, padding = 1, bias = False)
        model.maxpool = torch.nn.Identity()
    return model


def train_duq(experiment_name: str):
    """Function used for training a model with DUQ.
    The model is converted to a DUQ model and then trained
    in a training/validation phase setup. All types of models
    and datasets are possible to be used.

    Args:
        experiment_name: Name of the experiment to be used.
    """
    # Setting device to use
    device = get_device()

    # Retrieving hyperparameter dictionary
    hyp_dict = setup_hyp_dict(experiment_name)
    args = SimpleNamespace(**hyp_dict)

    # Checking if the experiment is set to be RBF
    if args.RBF_flag == False:
        print("Not an RBF experiment, exiting ...")
        return

    # Defining the train transforms
    transform = get_transforms(args.dataset, args.augmentation)
    # Retrieving data loaders
    data_loaders = get_data_loaders(args.batch_size, transform, args.dataset)

    # Setting up tensorboard writers and writing hyperparameters
    tensorboard_writers, experiment_path = setup_tensorboard(experiment_name, "Experiment-Results")
    setup_hyp_file(tensorboard_writers["hyp"], hyp_dict)

    # Getting the model and converting to RBF
    model = args.model
    classes = data_loaders["train"].dataset.n_classes
    if args.dataset.__name__ == "CIFAR10Dataset":
        model = preprocess_model(model)
    model = set_rbf_model(model, classes, device, args.gp_const, args.batch_size)
    model.to(device)
    
    # Setting up metrics
    best_loss = 1000
    acc_metric = Accuracy(task = "multiclass", num_classes = classes).to(device)
    f1_metric = F1Score(task = "multiclass", num_classes = classes).to(device)
    training_start = time.time()

    for i in range(args.epochs):
        print("Epoch " + str(i))
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                print("Training phase")
            else:
                model.eval()
                print("Validation phase")

            # Set model metrics to 0 and starting model timer
            loss_over_epoch = 0
            acc = 0
            f1_score = 0
            total_imgs = 0
            start_time = time.time()

            # Setting combined lists of predicted and actual labels
            combined_labels = []
            combined_labels_pred = []

            for inputs, labels in tqdm(data_loaders[phase]):
                model.train()
                args.optimizer.zero_grad()

                inputs = inputs.to(device)
                labels = labels.to(device)

                if args.gp_const != None:
                    inputs.requires_grad_(True)

                # Get model outputs, convert labels to one hot for
                # calculating BCE loss and updating centroids
                model_output = model(inputs)
                predicted_labels = model_output.argmax(dim = 1)
                labels_l = nn.functional.one_hot(labels, classes).float()

                loss = args.criterion(model_output, labels_l)
                acc += acc_metric(predicted_labels, labels).item()
                f1_score += f1_metric(predicted_labels, labels).item()

                # Updating model weights if in training phase
                if phase == "train":
                    if args.gp_const != None:
                        grad_pen = model.get_grad_pen(inputs, model_output)
                        loss += grad_pen
                    # Backwards pass and updating optimizer
                    loss.backward()
                    args.optimizer.step()

                    inputs.requires_grad_(False)

                    with torch.no_grad():
                        model.eval()
                        model.update_centroids(inputs, labels_l)

                # Adding the loss over the epoch and counting
                # total images a prediction was made over
                loss_over_epoch += loss.item()
                total_imgs += len(inputs)

                # Appending prediction and actual labels to combined lists
                combined_labels.append(labels)
                combined_labels_pred.append(predicted_labels)
        
            writer = tensorboard_writers[phase]
            report_metrics(args.PFM_flag, start_time, len(data_loaders[phase]), acc,
                       f1_score, loss_over_epoch, total_imgs, writer, i,
                       experiment_path)

        if phase == "val":
            # Checking if loss improved
            if best_loss > loss_over_epoch:
                best_loss = loss_over_epoch
                early_stop = 0
            # Early stopping if necessary
            else:
                early_stop += 1
                if early_stop > args.early_limit and args.early_limit != 0:
                    print("Early stopping ")
                    return
            # Updating the learning rate if updating scheduler is used
            args.scheduler.step()

    elapsed_time = time.time() - training_start
    print("Total training time (H/M/S) = ", 
          time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    torch.save(model, os.path.join(experiment_path, "model.pth"))

    # Closing tensorboard writers
    for _, writer in tensorboard_writers.items():
        writer.close() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type = str)
    parser.add_argument("--n_runs", type = int, default = 1)
    args = parser.parse_args()
    experiment_name = args.experiment_name
    results_path = os.path.join("Results", "Experiment-Results")

    if experiment_name != None:
        # An experiment was given, check if it exists
        if os.path.exists(os.path.join("Experiments", experiment_name + ".json")):  
            # Run for multiple times if indicated
            for _ in range(args.n_runs):
                    train_duq(experiment_name)
            # Merge runs if multiple
            if args.n_runs != 1:
                merge_experiments([experiment_name], results_path)
                calculate_acc_std([experiment_name], results_path) 
        else:
            print("Experiment does not exist, exiting ...")
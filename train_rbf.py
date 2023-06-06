import time
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchmetrics import Accuracy, F1Score
import torchvision
from tqdm import tqdm

from utils import get_transforms, setup_tensorboard, get_data_loaders, \
                  report_metrics, find_classification_module

from datasets import CIFAR10Dataset
from torchvision.models import resnet18


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

    The essence of DUQ is that it learns a set of centroids for each class,
    which it can then compare to new inputs during inference time. The
    distance to the closest centroid is the uncertainty metric.
    """
    def __init__(self, fe, in_features, out_features, device):
        super(RBF_model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = 0.1
        self.gamma = 0.999
        self.device = device

        # Initializing kernels centroid embedding
        # Set the feature extractor after the initialization of the kernels!
        # otherwise the model does not learn the centroids at all
        self.kernels = nn.Parameter(torch.Tensor(in_features, out_features,
                                                 in_features))
        nn.init.kaiming_normal_(self.kernels, nonlinearity = 'relu')
        self.fe = fe

        self.N = (torch.zeros(out_features) + 13).to(device)
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
        # Gradient penalty constant, taken from DUQ paper
        gp_const = 0.25

        # First getting gradients
        gradients = self.get_gradients(inputs, model_output)

        # Then computing L2 norm (2 sided)
        L2_norm = gradients.norm(2, dim = 1)

        # Applying the 2 sided penalty
        grad_pen = ((L2_norm - 1) ** 2).mean()

        return grad_pen * gp_const


def set_rbf_model(model: torchvision.models, classes: int, device: torch.device):
    """This function takes a Pytorch deep Learning model, finds its
    classification layer and then converts that to an RBF layer.
    """
    name, module, idx = find_classification_module(model)
    in_features = module.in_features
    if idx != None:
        model._modules[name][idx] = torch.nn.Identity()
    else:
        model._modules[name] = torch.nn.Identity()
    model = RBF_model(model, in_features, classes, device)
    return model


if __name__ == '__main__':
    """This is a temproary function that exists to showcase a 
    working setup for a model with an RBF layer. It should be replaced
    in the future with a version that is capable of accepting an
    experiment file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Getting transforms, dataloaders and the model
    transform = get_transforms(CIFAR10Dataset, "simple")
    data_loaders = get_data_loaders(128, True, 4, transform, CIFAR10Dataset)
    model = resnet18()

    # Setting optimizer and the loss to Binary Cross Entropy (as used by Joost in DUQ setup)
    optimizer = optim.SGD(model.parameters(), lr = 0.05, momentum = 0.9, weight_decay = 0.0005)
    criterion = nn.BCELoss()

    # Adapting model calls
    classes = data_loaders["train"].dataset.n_classes
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model = set_rbf_model(model, classes, device)
    model.to(device)

    # Additional function arguments, that are normally taken from an experiment file
    epochs = 100
    pfm_flag = {"Terminal": False, "Tensorboard": True}
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [25, 50, 75], gamma = 0.2)
    early_stop_limit = 0
    tensorboard_writers, experiment_path = setup_tensorboard("RBFMANUAL", "Experiment-Results")
    
    # Setting up metrics
    best_loss = 1000
    acc_metric = Accuracy(task = "multiclass", num_classes = classes).to(device)
    f1_metric = F1Score(task = "multiclass", num_classes = classes).to(device)

    for i in range(epochs):
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
                optimizer.zero_grad()

                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs.requires_grad_(True)

                model_output = model(inputs)
                predicted_labels = model_output.argmax(dim = 1)
                labels_l = nn.functional.one_hot(labels, classes).float()

                loss = criterion(model_output, labels_l)
                acc += acc_metric(predicted_labels, labels).item()
                f1_score += f1_metric(predicted_labels, labels).item()

                # Updating model weights if in training phase
                if phase == "train":
                    grad_pen = model.get_grad_pen(inputs, model_output)
                    loss += grad_pen
                    # Backwards pass and updating optimizer
                    loss.backward()
                    optimizer.step()

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
            report_metrics(pfm_flag, start_time, len(data_loaders[phase]), acc,
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
                if early_stop > early_stop_limit and early_stop_limit != 0:
                    print("Early stopping ")
            # Updating the learning rate if updating scheduler is used
            scheduler.step()
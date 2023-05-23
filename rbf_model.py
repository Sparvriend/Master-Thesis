import torch
import torch.nn as nn

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
        self.fe = fe
        self.device = device

        # Initializing kernels centroid embedding
        self.kernels = nn.Parameter(torch.Tensor(in_features, out_features,
                                                 in_features))
        self.N = (torch.zeros(out_features) + 13).to(device)
        self.m = torch.zeros(in_features, out_features).to(device)
        self.m *= self.N
        
        nn.init.normal_(self.m, 0.05, 1)
        nn.init.kaiming_normal_(self.kernels, nonlinearity = 'relu')
    
    
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

        # Summing labels for updating N
        unique, counts = torch.unique(labels, return_counts = True)
        labels_sum = torch.zeros(self.out_features,
                                 dtype = torch.long).to(self.device)
        labels_sum[unique] = counts
 
        # Update N
        self.N = update_f(self.N, labels_sum)

        # Calculating centroid sum
        z = self.fe(inputs)
        z = torch.einsum("ij, mnj->imn", z, self.kernels)
        labels = labels.unsqueeze(1).cpu()
        z = z.type(torch.LongTensor)
        centroid_sum = torch.einsum("ijk, il->jk", z, labels).to(self.device)

        # Update m
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
        gp_const = 0.5

        # First getting gradients
        gradients = self.get_gradients(inputs, model_output)

        # Then computing L2 norm (2 sided)
        L2_norm = gradients.norm(2, dim = 1)

        # Applying the 2 sided penalty
        grad_pen = ((L2_norm - 1) ** 2).mean()

        return grad_pen * gp_const
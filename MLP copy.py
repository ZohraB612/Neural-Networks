import torch
import torch.nn.functional as f
from Data import get_loaders

train_loader, valid_loader, test_loader = get_loaders()

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
COLOR_CHANNELS = 3
N_CLASSES = 2


#Tarroni, G. (2021). ‘Slides provided by the lecturer at City, University of London’.
#Garcez, A. (2021). ‘Slides provided by the lecturer at City, University of London’.

"""
class to create MLP model
"""

class MLP(torch.nn.Module):
    def __init__(self, config, n_hidden_layers=2, n_hidden_neurons=40, keep_rate=0.5):
        super(MLP, self).__init__()
        self.n_hidden_neurons = n_hidden_neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = config["activation"]
        self.keep_rate = keep_rate
        # Set up perceptron layers and add dropout
        self.fc1 = torch.nn.Linear(IMAGE_WIDTH * IMAGE_WIDTH * COLOR_CHANNELS,
                                   n_hidden_neurons)
        self.fc1_drop = torch.nn.Dropout(1 - keep_rate)
        if n_hidden_layers == 2:
            self.fc2 = torch.nn.Linear(n_hidden_neurons,
                                       n_hidden_neurons)
            self.fc2_drop = torch.nn.Dropout(1 - keep_rate)

        self.out = torch.nn.Linear(n_hidden_neurons, N_CLASSES)

    def forward(self, x):
        x = x.view(-1, IMAGE_WIDTH * IMAGE_HEIGHT * COLOR_CHANNELS)

        activation = f.relu
        if self.activation == "sigmoid":
            activation = torch.nn.Sigmoid()

        x = activation(self.fc1(x))
        x = self.fc1_drop(x)

        if self.n_hidden_layers == 2:
            x = activation(self.fc2(x))
            x = self.fc2_drop(x)

        return f.log_softmax(self.out(x))


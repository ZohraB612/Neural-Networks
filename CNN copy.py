import torch.nn as nn
import torch.nn.functional as f
from prettytable import PrettyTable


#Tarroni, G. (2021). ‘Slides provided by the lecturer at City, University of London’.

"""
This function is used to present the parameters of the model
"""

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

"""
function for CNN model
"""

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=5,
                               kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=5,
                               out_channels=10,
                               kernel_size=3)
        self.conv_drop2 = nn.Dropout2d()
        # self.conv3 = nn.Conv2d(in_channels=10,
        #                       out_channels=20,
        #                       kernel_size=3)
        # self.fc = nn.Linear(3920, 128)
        self.fc = nn.Linear(9000, 64)
        self.out = nn.Linear(64, 2)

    # 0/1
    def forward(self, x):
        x = f.relu(f.max_pool2d(self.conv1(x), 2))
        x = f.relu(f.max_pool2d(self.conv_drop2(self.conv2(x)), 2))
        # x = f.relu(f.max_pool2d(self.conv3(x), 2))
        x = x.view(x.shape[0], -1)
        x = f.relu(self.fc(x))
        x = f.dropout(x, training=self.training)
        # x = f.relu(self.fc2(x))
        x = f.log_softmax(self.out(x))
        return x

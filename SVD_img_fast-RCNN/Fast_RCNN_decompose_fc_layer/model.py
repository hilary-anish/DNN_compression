import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(841, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def layer_decompose():

    net = Net()

    # Generating a random array
    random_array = torch.randn(3, 128,128)

    out = net(random_array)
    # print(net)



    for name, parameters in net.named_parameters():
        # print(name, parameters.shape) 
        if 'fc3.weight' in name:
            fc3_weight = parameters

    # Count the number of parameters
    total_params = sum(p.numel() for p in net.parameters())
    print("Total number of parameters in the model:", total_params)
    
    fc3_weight = fc3_weight.detach().numpy()
    print(fc3_weight.shape, type(fc3_weight))

    return fc3_weight

layer_decompose()
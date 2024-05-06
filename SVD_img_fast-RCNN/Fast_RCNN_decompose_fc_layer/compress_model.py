import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class compressed_Net(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(841, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3_L = nn.Linear(84, k,  bias=False)
        self.fc3_U = nn.Linear(k, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)                # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3_L(x)
        x = self.fc3_U(x)

        return x
    

new_net = compressed_Net(5)

# Count the number of parameters
total_params = sum(p.numel() for p in new_net.parameters())
print("Total number of parameters in the compressed model:", total_params)
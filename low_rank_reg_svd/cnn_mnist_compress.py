import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


    
class compressed_Net(nn.Module):
    def __init__(self, k:list):
        super(compressed_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2_L = nn.Conv2d(32, k[0], (3,1), 1, bias=False)
        self.conv2_U = nn.Conv2d(k[0], 64, (1,3), 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2_L(x)
        x = self.conv2_U(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    

""" new_net = compressed_Net(5)

# Count the number of parameters
total_params = sum(p.numel() for p in new_net.parameters())
print("Total number of parameters in the compressed model:", total_params)

# Generating a random array
random_array = torch.randn(4, 1, 28, 28)

out = new_net(random_array)
print(out.shape) """
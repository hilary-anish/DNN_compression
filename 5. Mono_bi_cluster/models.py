import torch
import torch.nn as nn
import torch.nn.functional as F


# Normal model

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(96, 256, 5)
        self.fc1 = nn.Linear(256 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# Model to apply mono-chromatic approximation

class mono_Net(nn.Module):
    def __init__(self, c_prime):
        super().__init__()
        self.conv1_L = nn.Conv2d(3, c_prime, 1, bias=False)
        self.conv1_U = nn.Conv2d(c_prime, 96, 7, groups=c_prime)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(96, 256, 5)
        self.fc1 = nn.Linear(256 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1_L(x)
        x = self.pool(F.relu(self.conv1_U(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    


# Model to apply bi-clustering

class bicluster_Net(nn.Module):
    def __init__(self, c_prime, g):
        super().__init__()
        self.conv1_L = nn.Conv2d(3, c_prime, 1, bias=False)
        self.conv1_U = nn.Conv2d(c_prime, 96, 7, groups=c_prime)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_L = nn.Conv2d(96, 38, 1, groups=g, bias=False)
        self.conv2_M = nn.Conv2d(38, 48, 5, groups=g, bias=False)
        self.conv2_U = nn.Conv2d(48, 256, 1, groups=g)
        self.fc1 = nn.Linear(256 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1_L(x)
        x = self.pool(F.relu(self.conv1_U(x)))
        x = self.conv2_L(x)
        x = self.conv2_M(x)
        x = self.pool(F.relu(self.conv2_U(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x






ran_in = torch.randn(5, 3, 32, 32)

model = bicluster_Net(c_prime=3, g=2)
#model = Net()
#model = mono_Net(c_prime=3)

out = model(ran_in)
print(out.shape)



total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters in the compressed model:", total_params)
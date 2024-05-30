import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from cnn_mnist import Net 
from torchvision import datasets, transforms
import svd_compress


import yaml

# Path to your YAML file
yaml_file = "args.yaml"

# Read the YAML file
with open(yaml_file, 'r') as file:
    args = yaml.safe_load(file)





def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def test_main():

    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    use_mps = not args['no_mps'] and torch.backends.mps.is_available()

    torch.manual_seed(args['seed'])

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    test_kwargs = {'batch_size': args['test_batch_size']}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    test_data = datasets.MNIST('../data', train=False,
                       transform=transform)
    
    print("Image dimension:", test_data.data.size())

    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)


    model = svd_compress.decomposed_model.to(device)

    test(model, device, test_loader)


if __name__ == '__main__':
    test_main() 

"""     model = Net().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pt')) """
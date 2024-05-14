from cnn_mnist import Net
from cnn_mnist_compress import compressed_Net
import torch
from torch.nn.parameter import Parameter



def transfer_weights(decomposed_tensors:dict):
    
    source_model = Net()
    target_model = compressed_Net(list(decomposed_tensors.values()))
    
    # Load the state_dict of trained model
    state_dict_path = "mnist_cnn.pt"  
    state_dict = torch.load(state_dict_path)
    source_model.load_state_dict(torch.load('mnist_cnn.pt'))
    
    # Define which parameters to transfer based on their names
    params_to_transfer = {}

    for name, parameters in source_model.named_parameters():
        if not name in decomposed_tensors.keys():
            # Add a new key-value pair
            params_to_transfer[name] = name

    #print(params_to_transfer)

    # Iterate through the state_dict
    for parameter_name, wghts in state_dict.items():
        if not parameter_name in decomposed_tensors.keys():
            # Transfer parameters without tracking gradients
            with torch.no_grad():
                for source_name, target_name in params_to_transfer.items():
                    if source_name in source_model.state_dict() and target_name in target_model.state_dict():
                        target_model.state_dict()[target_name].copy_(source_model.state_dict()[source_name])


    return target_model



def transfer_decomposed_weights(target_model, compress_wght_dict):
    for wght_name, wght_params in compress_wght_dict.items():
        target_model.state_dict()[wght_name] = Parameter(wght_params)

    return target_model






'''# Verify the transfer
print("Source Model - Layer 1 Weight:")
print(source_model.conv1.weight)
print("Target Model - Layer 1 Weight (after transfer):")
print(target_model.conv1.weight)'''

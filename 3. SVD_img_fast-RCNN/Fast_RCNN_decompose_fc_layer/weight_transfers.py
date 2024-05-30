from model import Net
from compress_model import compressed_Net
import torch
from torch.nn.parameter import Parameter



def transfer_weights(k):
    
    source_model = Net()
    target_model = compressed_Net(k)

    # Define which parameters to transfer based on their names
    params_to_transfer = {}

    for name, parameters in source_model.named_parameters():
        # print(name, parameters.shape) 
        if not 'fc3.weight' in name:
            # Add a new key-value pair
            params_to_transfer[name] = name

    # print(params_to_transfer)

    # Transfer parameters without tracking gradients
    with torch.no_grad():
        for source_name, target_name in params_to_transfer.items():
            if source_name in source_model.state_dict() and target_name in target_model.state_dict():
                target_model.state_dict()[target_name].copy_(source_model.state_dict()[source_name])
    

    return target_model



def transfer_decomposed_weights(target_model, fc3_L_weight, fc3_U_weight):

    target_model.state_dict()['fc3_L.weight'] = Parameter(fc3_L_weight)
    target_model.state_dict()['fc3_U.weight'] = Parameter(fc3_U_weight)

    return target_model






'''# Verify the transfer
print("Source Model - Layer 1 Weight:")
print(source_model.conv1.weight)
print("Target Model - Layer 1 Weight (after transfer):")
print(target_model.conv1.weight)'''

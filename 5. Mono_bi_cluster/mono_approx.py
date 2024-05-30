import torch
import numpy as np
import torch.backends.cudnn as cudnn


# Get the weights of first layer
state_dict = torch.load( './checkpoint/ckpt.pth')
state_dict = state_dict['net']

for name, param in state_dict.items():
    if 'conv1.weight' in name:
        first_layer_weights = param
print(f'first layer shape: {first_layer_weights.shape}')


# Convert the weight tensor from Torch to Numpy
first_layer_weights = first_layer_weights.detach().cpu().numpy()

out_channel = first_layer_weights.shape[0]
in_channel = first_layer_weights.shape[1]
kernel_size = first_layer_weights.shape[2]

U_weights = []
Vt_weights = []
# Iterate through each kernel and perform SVD
for i in first_layer_weights:
    reshaped_matrix = i.reshape(i.shape[0], -1)
    if reshaped_matrix.shape!=(3,49):
        print('Error in weight reshape')

    r, c = reshaped_matrix.shape
    k = min(r,c)
    rank = 1

    # apply svd on the weight matrix
    U, S, V_t = np.linalg.svd(reshaped_matrix, full_matrices=True)
    #print(U.shape, S.shape, V_t.shape)

    # reshape U vectors
    lower = U[:, :rank] * np.sqrt(S[:rank]) 
    lower = lower.reshape(in_channel, 1, 1)
    U_weights.append(lower)
    #print(lower.shape)

    # reshape V_t vectors
    upper = V_t[:rank,:]* np.sqrt(S)[:rank, np.newaxis]
    upper = upper.reshape(1, kernel_size, kernel_size)
    Vt_weights.append(upper)
    #print(upper.shape)


# as mentioned in the paper
c_prime = 3
F = out_channel
cluster_wght = int(F/c_prime)

# cluster center for U_weights
# Iterate over the list and select subsequent 32 elements in each iteration
U_center = []
for i in range(0, len(U_weights), cluster_wght):
    subset = U_weights[i:i+cluster_wght]
    subset = np.array(subset).mean(axis=0)
    U_center.append(subset)

U_center = np.array(U_center)
U_center = torch.tensor(U_center)
print(type(U_center))

# clustering of Vt_weights
V_group = []
for i in range(0, len(Vt_weights), cluster_wght):
    subset = Vt_weights[i:i+cluster_wght]
    subset = np.array(subset)
    V_group.append(subset)

V_group = np.array(V_group)
V_group = np.concatenate(V_group, axis=0)
V_group = torch.tensor(V_group)
print(type(V_group))
print(f'U shape: {U_center.shape} , Vt shape: {V_group.shape}')


compress_wght_dict = {}

compress_wght_dict['module.conv1_L.weight'] = U_center
compress_wght_dict['module.conv1_U.weight'] = V_group



# weight transfers from trained model to mono Net model


from models import Net, mono_Net
from torch.nn.parameter import Parameter


def transfer_weights():
    
    source_model = Net()
    target_model = mono_Net(c_prime)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the state_dict of trained model

    if device == 'cuda':
        source_model = torch.nn.DataParallel(source_model)
        target_model = torch.nn.DataParallel(target_model)
        cudnn.benchmark = True

    source_model = source_model.to(device)
    target_model = target_model.to(device)
    
    # Load the state_dict of trained model
 
    state_dict = torch.load( './checkpoint/ckpt.pth')
    state_dict = state_dict['net']
    source_model.load_state_dict(state_dict)
    
    decomposed_tensor = ['module.conv1.weight']
    # Define which parameters to transfer based on their names
    params_to_transfer = {}

    for name, parameters in source_model.named_parameters():
        if not name in decomposed_tensor:
            # Add a new key-value pair
            params_to_transfer[name] = name

    #print(params_to_transfer)

    # Iterate through the state_dict
    for parameter_name, wghts in state_dict.items():
        if not parameter_name in decomposed_tensor:
            # Transfer parameters without tracking gradients
            with torch.no_grad():
                for source_name, target_name in params_to_transfer.items():
                    if source_name in source_model.state_dict() and target_name in target_model.state_dict():
                        target_model.state_dict()[target_name].copy_(source_model.state_dict()[source_name])


    return target_model



def transfer_decomposed_weights(target_model, compress_wght_dict):
    with torch.no_grad():
        for wght_name, wght_params in compress_wght_dict.items():
            target_model.state_dict()[wght_name].copy_(Parameter(wght_params))

    return target_model


# decomposed weights and other weights are transferred, but fine tune is required

def mono_chrome_model():
    target_model = transfer_weights()
    mono_decomposed_model = transfer_decomposed_weights(target_model, compress_wght_dict)

    # Check if weights are properly transferred
    for name, parameters in mono_decomposed_model.named_parameters():
        if name == 'module.conv1_U.weight':
            p = parameters.detach().cpu()
            print(torch.allclose(V_group, p, rtol=1e-02, atol=1e-02, equal_nan=False))


    return mono_decomposed_model





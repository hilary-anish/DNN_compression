import torch
import numpy as np
import torch.backends.cudnn as cudnn


# Get the weights of second layer
state_dict = torch.load( './checkpoint/mono_ckpt.pth')
state_dict = state_dict['net']

for name, param in state_dict.items():
    if 'conv2.weight' in name:
        second_layer_weights = param
#print(f'second layer shape: {second_layer_weights.shape}')


# Torch to Numpy - perform Transpose, Reshape 
second_layer_weights = second_layer_weights.detach().cpu().numpy()
out_channel = second_layer_weights.shape[0]
in_channel = second_layer_weights.shape[1]
kernel_size = second_layer_weights.shape[2]
second_layer_weights = second_layer_weights.transpose(1,2,3,0).reshape(96,-1)



# create two clusters of weight
dim1 = second_layer_weights.shape[0]
dim2 = second_layer_weights.shape[1]
first_part = second_layer_weights[0:int(dim1/2), 0:int(dim2/2)]
second_part = second_layer_weights[int(dim1/2):int(dim1), int(dim2/2):int(dim2)]

rank_k1 = 19
rank_k2 = 24

c_prime = 3
g = 2

Lower = []
Middle = []
Upper = []

def bi_clustering_svd(cluster_part):
    # apply SVD in a cluster : k1
    U_k1, S_k1, V_t_k1 = np.linalg.svd(cluster_part, full_matrices=True)
    #print(U_k1.shape, S_k1.shape, V_t_k1.shape)

    # reshape U vectors
    lower = U_k1[:, :rank_k1] * np.sqrt(S_k1[:rank_k1]) 
    lower = lower.reshape(rank_k1, int(in_channel/2), 1, 1)
    Lower.append(lower)
    #print(lower.shape)

    # reshape V_t vectors
    upper = V_t_k1[:rank_k1,:]* np.sqrt(S_k1)[:rank_k1, np.newaxis]
    upper = upper.reshape(-1, int(out_channel/2))
    #print(upper.shape)

    # apply SVD for second time : k2
    U_k2, S_k2, V_t_k2 = np.linalg.svd(upper, full_matrices=True)
    #print(U_k2.shape, S_k2.shape, V_t_k2.shape)

    middle = U_k2[:, :rank_k2] * np.sqrt(S_k2[:rank_k2]) 
    middle = middle.reshape(rank_k2, rank_k1, kernel_size, kernel_size)
    Middle.append(middle)
    #print(middle.shape)

    upper = V_t_k2[:rank_k2,:]* np.sqrt(S_k2)[:rank_k2, np.newaxis]
    upper = upper.reshape(int(out_channel/2), rank_k2, 1, 1)
    Upper.append(upper)
    #print(upper.shape)



bi_clustering_svd(first_part)
bi_clustering_svd(second_part)



# Create approximated weight tensors for conv_2 layer
conv2_L = torch.tensor(np.concatenate((Lower[0], Lower[1]), axis=0))
conv2_M = torch.tensor(np.concatenate((Middle[0], Middle[1]), axis=0))
conv2_U = torch.tensor(np.concatenate((Upper[0], Upper[1]), axis=0))

print(conv2_L.shape, conv2_M.shape, conv2_U.shape)


compress_wght_dict = {}


compress_wght_dict['module.conv2_L.weight'] = conv2_L
compress_wght_dict['module.conv2_M.weight'] = conv2_M
compress_wght_dict['module.conv2_U.weight'] = conv2_U





from models import mono_Net, bicluster_Net
from torch.nn.parameter import Parameter




def transfer_weights():
    
    source_model = mono_Net(c_prime)
    target_model = bicluster_Net(c_prime, g)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        source_model = torch.nn.DataParallel(source_model)
        target_model = torch.nn.DataParallel(target_model)
        cudnn.benchmark = True

    source_model = source_model.to(device)
    target_model = target_model.to(device)

    # Load the state_dict of trained model
 
    state_dict = torch.load( './checkpoint/mono_ckpt.pth')
    state_dict = state_dict['net']
    source_model.load_state_dict(state_dict)
    
    decomposed_tensor = ['module.conv2.weight']
    # Define which parameters to transfer based on their names
    params_to_transfer = {}

    for name, parameters in source_model.named_parameters():
        if not name in decomposed_tensor:
            # Add a new key-value pair
            params_to_transfer[name] = name

    print(params_to_transfer)

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

def bi_cluster_model():
    target_model = transfer_weights()
    bicluster_decomposed_model = transfer_decomposed_weights(target_model, compress_wght_dict)

    # Check if weights are properly transferred
    for name, parameters in bicluster_decomposed_model.named_parameters():
        if name == 'module.conv2_U.weight':
            p = parameters.detach().cpu()
            print(torch.allclose(conv2_U, p, rtol=1e-06, atol=1e-06, equal_nan=False))


    return bicluster_decomposed_model


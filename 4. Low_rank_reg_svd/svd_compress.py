
from cnn_mnist_compress import compressed_Net
from weight_transfers import transfer_weights, transfer_decomposed_weights
import torch
import numpy as np


def svd_decompose(wght_tensor, rank):
    # Check if tensor is a NumPy array else Convert tensor to NumPy array
    if isinstance(wght_tensor, np.ndarray):
        print("Tensor is already a NumPy array")
    else:
        wght_tensor = wght_tensor.numpy()
        print("Converted tensor to NumPy array")
        #print(f'Type of tensor is {type(wght_tensor)}, the shape of tensor is {wght_tensor.shape}')
    
    out_channel, in_channel, k1, k2 = wght_tensor.shape
    wght_matrix = np.transpose(wght_tensor, (1,2,3,0))
    # Reshape to combine the first two dimensions and the last two dimensions
    wght_matrix = wght_matrix.reshape(wght_matrix.shape[0]*wght_matrix.shape[1], wght_matrix.shape[2]*wght_matrix.shape[3])

    #print(f'wght tensor shape:', wght_matrix.shape)

    r, c = wght_matrix.shape
    k = min(r,c)

    #print(k)

    # apply svd on the weight matrix
    U, S, V_t = np.linalg.svd(wght_matrix, full_matrices=True)
    
    # reshape and transpose to fit in lower conv layer
    lower = U[:, :rank] * np.sqrt(S[:rank]) 
    lower = lower.reshape(in_channel, k1, 1, -1)
    lower = lower.transpose(3,0,1,2)
    #print(lower.shape)

    # reshape and transpose to fit in upper conv layer 
    upper = V_t[:rank,:]* np.sqrt(S)[:rank, np.newaxis]
    upper = upper.reshape(-1, 1, k2, out_channel)
    upper = upper.transpose(3,0,1,2)
    #print(upper.shape)

    # Convert NumPy array to PyTorch tensor
    lower = torch.tensor(lower)
    upper = torch.tensor(upper)

    #print(lower.shape, type(lower))
    #print(upper.shape, type(upper))


    return lower, upper
   



def get_decomposed_model(trained_pt_file):
    # Load the state dictionary file
    state_dict = torch.load(trained_pt_file, map_location=torch.device('cpu'))

    wghts_to_decompose = {'conv2.weight':10}
    decomposed_tensors = {}

    # Iterate through the keys and check for the name in wghts_to_decompose
    for parameter_name, wghts in state_dict.items():
        if parameter_name in wghts_to_decompose.keys():
        lower, upper = svd_decompose(wghts, wghts_to_decompose[parameter_name])
        
        # after weight decomposition, new weights are paired with their corresponding names
        parts = parameter_name.split('.')
        parts[0] = parts[0]+'_L' 
        new_name_L = '.'.join(parts)
        parts = parameter_name.split('.')
        parts[0] = parts[0]+'_U'
        new_name_U = '.'.join(parts)

        decomposed_tensors[new_name_L] = lower
        decomposed_tensors[new_name_U] = upper


    # transfer weights from source model to new model : except for the decomposed layers
    decomposed_model = transfer_weights(wghts_to_decompose)

    decomposed_model = transfer_decomposed_weights(decomposed_model, decomposed_tensors)

return decomposed_model



if __name__=="__main__":
    # trained pt file is given as argument to get the decomposed model
    decomposed_model = get_decomposed_model("mnist_cnn.pt")
    torch.save(decomposed_model, "./decom_mod.pth")


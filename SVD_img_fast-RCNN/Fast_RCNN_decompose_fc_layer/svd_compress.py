import numpy as np
from model import layer_decompose
from compress_model import compressed_Net
import torch
from weight_transfers import transfer_weights, transfer_decomposed_weights


np_wght_matrix = layer_decompose()
r, c = np_wght_matrix.shape
k = min(r,c)

print(k)

rank = 5

# apply svd on the weight matrix
U, S, V_t = np.linalg.svd(np_wght_matrix, full_matrices=True)

# reconstructing the weight matrix with 'k'
print(np.allclose(np_wght_matrix, np.dot(U[:, :rank] * S[:rank], V_t[:rank,:])))



# transfer weights from source model to new model : except for the decomposed layers
decomposed_model = transfer_weights(rank)

# transfer decomposed weight matrix to the fc-weights of new model
fc3_L_weight = np.transpose(V_t[:rank,:])  
fc3_U_weight = U[:, :rank] * S[:rank]

# Convert NumPy array to PyTorch tensor
fc3_L_weight = torch.tensor(fc3_L_weight)
fc3_U_weight = torch.tensor(fc3_U_weight)

""" print(fc3_L_weight.shape, type(fc3_L_weight))
print(fc3_U_weight.shape, type(fc3_U_weight)) """

decomposed_model = transfer_decomposed_weights(decomposed_model, fc3_L_weight, fc3_U_weight)



# Generating a random array
random_array = torch.randn(3, 128,128)

out = decomposed_model(random_array)
print(decomposed_model, out.shape)




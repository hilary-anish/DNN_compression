import numpy as np

# VECTOR OUTER PRODUCTS 

# Define the vectors
a = np.array([1, 2])
b = np.array([3, 4, 5, 6])
c = np.array([7, 8, 9])

# Compute the outer product of a and b - 2d MATRIX
outer_product_ab = np.outer(a, b)

# Compute the outer product of the result with c - 3d TENSOR
# We reshape the result of outer_product_ab to be compatible for a final outer product
result = np.tensordot(outer_product_ab, c, axes=0)
tensor = np.einsum('ij,k->ijk', outer_product_ab, c)
three_outer = np.einsum('i,j,k->ijk', a,b, c)


#############################################################################################

# TENSOR MULTIPLICATION

# Reshape the array to nd array shapes
tensor_array = np.arange(1,25).reshape(2, 4, 3)
matrix = np.arange(1,7).reshape(2,3)
vector = np.arange(1,5)

# tensor - matrix product
ten_mat_dot = np.tensordot(tensor_array, matrix, axes=([2], [1]))
tensor_matrix = np.einsum('lk, ijk->ijl', matrix, tensor_array)
print('The tensor_matrix is {} and the shape is {}'.format(tensor_matrix, tensor_matrix.shape))

# tensor - vector product
ten_vec_dot = np.tensordot(tensor_array, vector, axes=([1], [0]))
tensor_vector = np.einsum('j, ijk->ik', vector, tensor_array)
print('The tensor_vector is {} and the shape is {}'.format(tensor_vector, tensor_vector.shape))

print(np.equal(ten_mat_dot,tensor_matrix))
print(np.equal(ten_vec_dot,tensor_vector))

################################################################################################

# CP Decomposition - ALS method

# using CP class

import numpy as np

Original_tensor = np.arange(1,25).reshape(2, 4, 3)
print('tensor is {}'.format(Original_tensor))

import tensorly as tl
from tensorly.decomposition import CP


# Convert to tensorly tensor
T_tensorly = tl.tensor(Original_tensor)

obj = CP(rank=2)
cp_tensor = obj.fit_transform(T_tensorly)

# #print(f' what is this {cp_tensor} and this {error} and now {pp}')
# print(f"decomposed tensor: weights are {cp_tensor[0]} and the factors are {cp_tensor[1]}")

################################


# using Parafac method

import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np



Original_tensor = np.arange(1,25).reshape(2, 4, 3)
print('tensor is {}'.format(Original_tensor))

T_tensorly = tl.tensor(Original_tensor)

# Set the rank for the CP decomposition
rank = 2

# Perform CP decomposition with error tracking
cp_tensor, errors = parafac(T_tensorly, rank=rank, return_errors=True)

# Print the error list
print("Reconstruction errors at each iteration:")
#print(errors)

for e, i in enumerate(cp_tensor[1]):
    if e==0:
        A=i
    if e==1:
        B=i
    if e==2:
        C=i
    print(f' this is the {e} factor {np.array(i)}')

# print(f'the error list is {cp_tensor}')
# print(A.shape, B.shape, C.shape)

rank_1 = []
rank_2 = []

krushal_notation = [A, B, C]

for k in krushal_notation:
    transposed_array = np.array([row for row in zip(*k)])
    for l in range(transposed_array.shape[0]):
        if l==0:
            rank_1.append(transposed_array[l])
        if l==1:
            rank_2.append(transposed_array[l])

rank_1_three_outer = np.einsum('i,j,k->ijk', rank_1[0],rank_1[1], rank_1[2])
rank_2_three_outer = np.einsum('i,j,k->ijk', rank_2[0],rank_2[1], rank_2[2])
approximated_tensor = rank_1_three_outer+rank_2_three_outer

print(f'the outer product is {rank_1_three_outer}')
print(f'the outer product is {rank_2_three_outer}')
print(f'CP approximation of tensors: {approximated_tensor}')
print(Original_tensor)


# Compute the difference between the matrices
difference = approximated_tensor - Original_tensor

# Calculate the Frobenius norm of the difference
frobenius_norm = np.linalg.norm(difference)

print("Frobenius norm between A and B:", frobenius_norm)





# example of frobenius norm
a = np.array([[2,1],[3,4]])
b = np.array([[2,3],[4,5]])
c = np.array([[1,2],[3,4]])

a_b = a-b
a_c = a-c

a_b = np.linalg.norm(a_b)
a_c = np.linalg.norm(a_c)

print(a_b)
print(a_c)

import numpy as np
import scipy.linalg as la

# S3 and its subgroups
S3 = ["()","(12)","(13)","(23)","(123)","(132)"]
Z2_subgroup_S3 = ["()","(12)"]
Z3_subgroup_S3 = ["()","(123)","(132)"]
triv_subgroup_S3 = ["()"]

# irreps of S3 and its subgroups
triv_S3 = {"()":[1], "(12)":[1], "(13)":[1], "(23)":[1], "(123)":[1], "(132)":[1]}
sgn_S3 = {"()":[1], "(12)":[-1], "(13)":[-1], "(23)":[-1], "(123)":[1], "(132)":[1]}
std_S3 = {"()":[[1,0],[0,1]],"(12)":[[-1,1],[0,1]],"(13)":[[0,-1],[-1,0]],"(23)":[[1,0],[1,-1]],"(123)":[[0,-1],[1,-1]],"(132)":[[-1,1],[-1,0]]}
w_Z3 = {"()":[1], "(123)":[np.exp(2*np.pi*1j/3)], "(132)":[np.exp(4*np.pi*1j/3)]}
w2_Z3 = {"()":[1], "(123)":[np.exp(4*np.pi*1j/3)], "(132)":[np.exp(2*np.pi*1j/3)]}

def find_dim_tensor(grp, rep1, rep2):
    maps_tensor_reps = [np.kron(rep1[perm], rep2[perm]) for perm in grp]

    # get basis for eigenspace corresponding to eigenvalue 1 of map_tensor_rep
    bases = [la.null_space(map - np.eye(len(map))) for map in maps_tensor_reps]
    # for each basis in bases, view the basis elements as rows in a matrix
    # find the null space of this matrix, and then view the null space basis elements as rows in a matrix
    # print("bases = ",bases)
    for basis in bases:
        if 0 in basis.shape:
            # some tensor fixes no elements
            return 0

    # print("transposed: ",[np.matrix.transpose(basis) for basis in bases]) 
    bases_realized_as_nullspaces = [np.matrix.transpose(la.null_space(np.matrix.transpose(basis))) for basis in bases]
    # print(bases_realized_as_nullspaces)
    # if any basis element is less than eps, set it equal to 0
    eps = 1e-10
    for basis in bases_realized_as_nullspaces:
        basis[np.abs(basis)<eps] = 0

    if sum([basis.shape[0] for basis in bases_realized_as_nullspaces]) == 0:
        # every element is fixed by tensor product, return dimension of rep
        return bases_realized_as_nullspaces[0].shape[1]

    nonempty_bases_realized_as_nullspaces = [basis for basis in bases_realized_as_nullspaces if basis.shape[0]>0]

    # stack the matrices  in bases_realized_as_nullspaces vertically into a single matrix
    big_matrix = np.vstack(nonempty_bases_realized_as_nullspaces)

    # find the null space of this
    big_matrix_nullspace = la.null_space(big_matrix)
    return big_matrix_nullspace.shape[1]

print("Dimension is", find_dim_tensor(Z3_subgroup_S3, sgn_S3, w2_Z3))
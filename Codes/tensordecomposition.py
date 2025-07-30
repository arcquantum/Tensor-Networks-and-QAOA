import numpy as np
import scipy.linalg as sp

def svd_tensor(tensor, num_left_indices):
    """
    Performs Singular Value Decomposition (SVD) on a given tensor, treating it
    as a matrix by partitioning its dimensions.

    The tensor is reshaped into a 2D matrix where the first 'num_left_indices'
    dimensions form the rows and the remaining dimensions form the columns.
    The SVD is then applied to this matrix, and the resulting U, S, and Vh
    matrices are reshaped back into tensors with appropriate bond dimensions.

    Args:
        tensor (np.ndarray): The input tensor of arbitrary dimensions (ndim >= 2).
        num_left_indices (int): The number of leading indices to group together
                                to form the "left" part (rows) of the matrix
                                for the SVD. This value must be between 1 and
                                (tensor.ndim - 1).

    Returns:
        tuple: A tuple containing:
            - U_tensor (np.ndarray): The left singular tensor, reshaped from U_matrix,
                                     with shape (tensor.shape[:num_left_indices], bond_dim).
            - S_matrix (np.ndarray): The diagonal matrix of singular values.
            - Vh_tensor (np.ndarray): The conjugate transpose of the right singular tensor,
                                      reshaped from Vh_matrix, with shape (bond_dim, tensor.shape[num_left_indices:]).
            - S_Vh_matrix (np.ndarray): The product of S_matrix and Vh_matrix.
            - Vh_matrix (np.ndarray): The Vh matrix from the SVD before reshaping.

    Raises:
        ValueError: If `num_left_indices` is not within the valid range [1, tensor.ndim - 1].
    """
    num_axes = tensor.ndim
    if not (1 <= num_left_indices < num_axes):
        raise ValueError(
            f"num_left_indices ({num_left_indices}) must be between 1 and {num_axes - 1} "
            f"for a tensor with {num_axes} dimensions."
        )

    left_shapes = tensor.shape[:num_left_indices]
    right_shapes = tensor.shape[num_left_indices:]

    dim_left = np.prod(left_shapes, dtype=int)
    dim_right = np.prod(right_shapes, dtype=int)

    matrix = tensor.reshape(dim_left, dim_right)

    U_matrix, S_vector, Vh_matrix = sp.linalg.svd(matrix, full_matrices=False)

    bond_dim = S_vector.shape[0]

    S_matrix = np.diag(S_vector)

    S_Vh_matrix = S_matrix @ Vh_matrix

    U_tensor_shape = (*left_shapes, bond_dim)
    U_tensor = U_matrix.reshape(U_tensor_shape)

    Vh_tensor_shape = (bond_dim, *right_shapes)
    Vh_tensor = Vh_matrix.reshape(Vh_tensor_shape)

    return U_tensor, S_matrix, Vh_tensor, S_Vh_matrix, Vh_matrix

def qr_tensor(tensor, num_left_indices):
    """
    Performs QR decomposition on a given tensor, treating it as a matrix by
    partitioning its dimensions.

    The tensor is reshaped into a 2D matrix where the first 'num_left_indices'
    dimensions form the rows and the remaining dimensions form the columns.
    The QR decomposition is then applied to this matrix, and the resulting Q
    and R matrices are reshaped back into tensors with appropriate bond dimensions.
    The decomposition uses 'economic' mode, meaning Q and R are sized such
    that R is square and Q has orthonormal columns.

    Args:
        tensor (np.ndarray): The input tensor of arbitrary dimensions (ndim >= 2).
        num_left_indices (int): The number of leading indices to group together
                                to form the "left" part (rows) of the matrix
                                for the QR decomposition. This value must be
                                between 1 and (tensor.ndim - 1).

    Returns:
        tuple: A tuple containing:
            - Q_tensor (np.ndarray): The orthogonal tensor, reshaped from Q_matrix,
                                     with shape (tensor.shape[:num_left_indices], bond_dim).
            - R_matrix (np.ndarray): The upper triangular matrix from the QR decomposition.
            - R_tensor (np.ndarray): The upper triangular tensor, reshaped from R_matrix,
                                     with shape (bond_dim, tensor.shape[num_left_indices:]).

    Raises:
        ValueError: If `num_left_indices` is not within the valid range [1, tensor.ndim - 1].
    """
    num_axes = tensor.ndim
    if not (1 <= num_left_indices < num_axes):
        raise ValueError(
            f"num_left_indices ({num_left_indices}) must be between 1 and {num_axes - 1} "
            f"for a tensor with {num_axes} dimensions."
        )

    left_shapes = tensor.shape[:num_left_indices]
    right_shapes = tensor.shape[num_left_indices:]

    dim_left = np.prod(left_shapes, dtype=int)
    dim_right = np.prod(right_shapes, dtype=int)

    matrix = tensor.reshape(dim_left, dim_right)

    Q_matrix, R_matrix = sp.linalg.qr(matrix, mode='economic')

    bond_dim = R_matrix.shape[0]

    Q_tensor_shape = (*left_shapes, bond_dim)
    Q_tensor = Q_matrix.reshape(Q_tensor_shape)

    R_tensor_shape = (bond_dim, *right_shapes)
    R_tensor = R_matrix.reshape(R_tensor_shape)

    return Q_tensor, R_matrix, R_tensor

def rq_tensor(tensor, num_left_indices):
    """
    Performs RQ decomposition on a given tensor, treating it as a matrix by
    partitioning its dimensions.

    The tensor is reshaped into a 2D matrix where the first 'num_left_indices'
    dimensions form the rows and the remaining dimensions form the columns.
    The RQ decomposition is then applied to this matrix, and the resulting R
    and Q matrices are reshaped back into tensors with appropriate bond dimensions.
    The decomposition uses 'economic' mode, meaning R and Q are sized such
    that R is square and Q has orthonormal rows.

    Args:
        tensor (np.ndarray): The input tensor of arbitrary dimensions (ndim >= 2).
        num_left_indices (int): The number of leading indices to group together
                                to form the "left" part (rows) of the matrix
                                for the RQ decomposition. This value must be
                                between 1 and (tensor.ndim - 1).

    Returns:
        tuple: A tuple containing:
            - Q_tensor (np.ndarray): The orthogonal tensor, reshaped from Q_matrix,
                                     with shape (bond_dim, tensor.shape[num_left_indices:]).
            - R_matrix (np.ndarray): The upper triangular matrix from the RQ decomposition.
            - R_tensor (np.ndarray): The upper triangular tensor, reshaped from R_matrix,
                                     with shape (tensor.shape[:num_left_indices], bond_dim).

    Raises:
        ValueError: If `num_left_indices` is not within the valid range [1, tensor.ndim - 1].
    """
    num_axes = tensor.ndim
    if not (1 <= num_left_indices < num_axes):
        raise ValueError(
            f"num_left_indices ({num_left_indices}) must be between 1 and {num_axes - 1} "
            f"for a tensor with {num_axes} dimensions."
        )

    left_shapes = tensor.shape[:num_left_indices]
    right_shapes = tensor.shape[num_left_indices:]

    dim_left = np.prod(left_shapes, dtype=int)
    dim_right = np.prod(right_shapes, dtype=int)

    matrix = tensor.reshape(dim_left, dim_right)

    R_matrix, Q_matrix = sp.linalg.rq(matrix, mode='economic')

    bond_dim = R_matrix.shape[1]

    Q_tensor_shape = (bond_dim, *right_shapes)
    Q_tensor = Q_matrix.reshape(Q_tensor_shape)

    R_tensor_shape = (*left_shapes, bond_dim)
    R_tensor = R_matrix.reshape(R_tensor_shape)

def svd_tensor_permutable(tensor, left_indices):
    """
    Performs Singular Value Decomposition (SVD) on a given tensor after
    rearranging its axes and reshaping it into a 2D matrix.

    This function allows for flexible partitioning of the tensor's indices
    into "left" (row) and "right" (column) groups before performing the SVD.
    The `left_indices` argument explicitly defines which axes should be grouped
    on the left side of the matrix, and the remaining axes are grouped on the right.
    The tensor is permuted according to these groups and then reshaped into a
    2D matrix. The SVD is applied to this matrix, and the resulting U, S, and Vh
    components are reshaped back into tensors, incorporating a new 'bond' dimension.

    Args:
        tensor (np.ndarray): The input tensor of arbitrary dimensions (ndim >= 0).
        left_indices (list or tuple of int): A sequence of integers specifying
                                             the indices of the tensor's axes
                                             that should form the "left" (row)
                                             part of the matrix for SVD. The order
                                             of indices in this list determines
                                             the order of dimensions in the output
                                             U_tensor. If empty, the `dim_left`
                                             will be 1.

    Returns:
        tuple: A tuple containing:
            - U_tensor (np.ndarray): The left singular tensor, reshaped from U_matrix.
                                     Its shape will be (*tensor.shape[i] for i in left_indices, bond_dim).
            - S_matrix (np.ndarray): The diagonal matrix of singular values.
            - Vh_tensor (np.ndarray): The conjugate transpose of the right singular tensor,
                                      reshaped from Vh_matrix. Its shape will be
                                      (bond_dim, *tensor.shape[i] for i in right_indices).
            - S_Vh_matrix (np.ndarray): The product of S_matrix and Vh_matrix.
            - Vh_matrix (np.ndarray): The Vh matrix from the SVD before reshaping.

    Raises:
        ValueError: If any index in `left_indices` is not a valid axis for the input tensor.
    """
    num_axes = tensor.ndim
    all_indices = set(range(num_axes))

    if not set(left_indices).issubset(all_indices):
        raise ValueError(f"left_indices {left_indices} contains invalid axes.")

    right_indices = sorted(list(all_indices - set(left_indices)))
    permutation = (*left_indices, *right_indices)

    permuted_tensor = tensor.transpose(permutation)

    left_shapes = [tensor.shape[i] for i in left_indices]
    right_shapes = [tensor.shape[i] for i in right_indices]

    dim_left = np.prod(left_shapes, dtype=int) if left_shapes else 1
    dim_right = np.prod(right_shapes, dtype=int) if right_shapes else 1

    matrix = permuted_tensor.reshape(dim_left, dim_right)

    U_matrix, S_vector, Vh_matrix = sp.linalg.svd(matrix, full_matrices=False)

    bond_dim = S_vector.shape[0]
    S_matrix = np.diag(S_vector)
    S_Vh_matrix = S_matrix @ Vh_matrix

    U_tensor = U_matrix.reshape(*left_shapes, bond_dim)
    Vh_tensor = Vh_matrix.reshape(bond_dim, *right_shapes)

    return U_tensor, S_matrix, Vh_tensor, S_Vh_matrix, Vh_matrix

def qr_tensor_permutable(tensor, left_indices):
    """
    Performs QR decomposition on a given tensor after rearranging its axes and
    reshaping it into a 2D matrix.

    This function allows for flexible partitioning of the tensor's indices
    into "left" (row) and "right" (column) groups before performing the QR decomposition.
    The `left_indices` argument explicitly defines which axes should be grouped
    on the left side of the matrix, and the remaining axes are grouped on the right.
    The tensor is permuted according to these groups and then reshaped into a
    2D matrix. The QR decomposition (using 'economic' mode) is applied to this matrix,
    and the resulting Q and R components are reshaped back into tensors, incorporating
    a new 'bond' dimension.

    Args:
        tensor (np.ndarray): The input tensor of arbitrary dimensions (ndim >= 0).
        left_indices (list or tuple of int): A sequence of integers specifying
                                             the indices of the tensor's axes
                                             that should form the "left" (row)
                                             part of the matrix for QR decomposition.
                                             The order of indices in this list
                                             determines the order of dimensions
                                             in the output Q_tensor. If empty,
                                             the `dim_left` will be 1.

    Returns:
        tuple: A tuple containing:
            - Q_tensor (np.ndarray): The orthogonal tensor, reshaped from Q_matrix.
                                     Its shape will be (*tensor.shape[i] for i in left_indices, bond_dim).
            - R_matrix (np.ndarray): The upper triangular matrix from the QR decomposition.
            - R_tensor (np.ndarray): The upper triangular tensor, reshaped from R_matrix.
                                     Its shape will be (bond_dim, *tensor.shape[i] for i in right_indices).

    Raises:
        ValueError: If any index in `left_indices` is not a valid axis for the input tensor.
    """
    num_axes = tensor.ndim
    all_indices = set(range(num_axes))

    if not set(left_indices).issubset(all_indices):
        raise ValueError(f"left_indices {left_indices} contains invalid axes.")

    right_indices = sorted(list(all_indices - set(left_indices)))
    permutation = (*left_indices, *right_indices)

    permuted_tensor = tensor.transpose(permutation)

    left_shapes = [tensor.shape[i] for i in left_indices]
    right_shapes = [tensor.shape[i] for i in right_indices]

    dim_left = np.prod(left_shapes, dtype=int) if left_shapes else 1
    dim_right = np.prod(right_shapes, dtype=int) if right_shapes else 1

    matrix = permuted_tensor.reshape(dim_left, dim_right)

    Q_matrix, R_matrix = sp.linalg.qr(matrix, mode='economic')

    bond_dim = R_matrix.shape[0]

    Q_tensor = Q_matrix.reshape(*left_shapes, bond_dim)
    R_tensor = R_matrix.reshape(bond_dim, *right_shapes)

    return Q_tensor, R_matrix, R_tensor

def rq_tensor_permutable(tensor, left_indices):
    """
    Performs RQ decomposition on a given tensor after rearranging its axes and
    reshaping it into a 2D matrix.

    This function allows for flexible partitioning of the tensor's indices
    into "left" (row) and "right" (column) groups before performing the RQ decomposition.
    The `left_indices` argument explicitly defines which axes should be grouped
    on the left side of the matrix, and the remaining axes are grouped on the right.
    The tensor is permuted according to these groups and then reshaped into a
    2D matrix. The RQ decomposition (using 'economic' mode) is applied to this matrix,
    and the resulting R and Q components are reshaped back into tensors, incorporating
    a new 'bond' dimension.

    Args:
        tensor (np.ndarray): The input tensor of arbitrary dimensions (ndim >= 0).
        left_indices (list or tuple of int): A sequence of integers specifying
                                             the indices of the tensor's axes
                                             that should form the "left" (row)
                                             part of the matrix for RQ decomposition.
                                             The order of indices in this list
                                             determines the order of dimensions
                                             in the output R_tensor. If empty,
                                             the `dim_left` will be 1.

    Returns:
        tuple: A tuple containing:
            - Q_tensor (np.ndarray): The orthogonal tensor, reshaped from Q_matrix.
                                     Its shape will be (bond_dim, *tensor.shape[i] for i in right_indices).
            - R_matrix (np.ndarray): The upper triangular matrix from the RQ decomposition.
            - R_tensor (np.ndarray): The upper triangular tensor, reshaped from R_matrix.
                                     Its shape will be (*tensor.shape[i] for i in left_indices, bond_dim).

    Raises:
        ValueError: If any index in `left_indices` is not a valid axis for the input tensor.
    """
    num_axes = tensor.ndim
    all_indices = set(range(num_axes))

    if not set(left_indices).issubset(all_indices):
        raise ValueError(f"left_indices {left_indices} contains invalid axes.")

    right_indices = sorted(list(all_indices - set(left_indices)))
    permutation = (*left_indices, *right_indices)

    permuted_tensor = tensor.transpose(permutation)

    left_shapes = [tensor.shape[i] for i in left_indices]
    right_shapes = [tensor.shape[i] for i in right_indices]

    dim_left = np.prod(left_shapes, dtype=int) if left_shapes else 1
    dim_right = np.prod(right_shapes, dtype=int) if right_shapes else 1

    matrix = permuted_tensor.reshape(dim_left, dim_right)

    R_matrix, Q_matrix = sp.linalg.rq(matrix, mode='economic')

    bond_dim = R_matrix.shape[1]

    Q_tensor = Q_matrix.reshape(bond_dim, *right_shapes)
    R_tensor = R_matrix.reshape(*left_shapes, bond_dim)

    return Q_tensor, R_matrix, R_tensor

def svd_tensor_mode(tensor, mode):
    """
    Performs Singular Value Decomposition (SVD) on a tensor by matricizing it
    along a specified mode.

    The function first permutes the tensor's axes so that the dimension
    corresponding to `mode` becomes the last dimension. The tensor is then
    reshaped into a 2D matrix where all dimensions *except* the specified mode
    are combined into the rows, and the specified mode's dimension forms the
    columns. SVD is applied to this matrix. The resulting U matrix is then
    reshaped back into a tensor and its axes are inverse-permuted to match
    the original tensor's ordering.

    Args:
        tensor (np.ndarray): The input tensor of arbitrary dimensions (rank >= 1).
        mode (int): The index of the axis (mode) along which the SVD is to be
                    performed. This axis will become the 'column' dimension
                    of the reshaped matrix. Must be between 0 and `tensor.ndim - 1`.

    Returns:
        tuple: A tuple containing:
            - U_tensor (np.ndarray): The left singular tensor, reshaped from U_matrix,
                                     with its original axis order restored. Its shape
                                     will be the original tensor's shape with the `mode`
                                     dimension replaced by `bond_dim`.
            - S_matrix (np.ndarray): The diagonal matrix of singular values.
            - Vh_matrix (np.ndarray): The conjugate transpose of the right singular
                                      matrix from the SVD.
            - S_Vh_matrix (np.ndarray): The product of S_matrix and Vh_matrix.

    Raises:
        ValueError: If `mode` is not a valid axis index for the input tensor.
    """
    rank = tensor.ndim
    if not (0 <= mode < rank):
        raise ValueError(f"Mode must be an integer between 0 and {rank - 1}.")

    other_indices = [i for i in range(rank) if i != mode]
    permute_map = other_indices + [mode]
    permuted_tensor = tensor.transpose(permute_map)

    permuted_shape = permuted_tensor.shape
    row_dim = np.prod(permuted_shape[:-1]).astype(int)
    col_dim = permuted_shape[-1]
    matrix = permuted_tensor.reshape(row_dim, col_dim)

    U_matrix, S_vector, Vh_matrix = sp.linalg.svd(matrix, full_matrices=False)

    S_matrix = np.diag(S_vector)
    S_Vh_matrix = S_matrix @ Vh_matrix

    bond_dim = U_matrix.shape[1]
    u_tensor_shape_permuted = permuted_shape[:-1] + (bond_dim,)
    U_tensor_permuted = U_matrix.reshape(u_tensor_shape_permuted)

    final_u_axes = other_indices + [mode]
    inv_permute_map = np.argsort(final_u_axes)
    U_tensor = U_tensor_permuted.transpose(inv_permute_map)

    return U_tensor, S_matrix, Vh_matrix, S_Vh_matrix

def qr_tensor_mode(tensor, mode):
    """
    Performs QR decomposition on a tensor by matricizing it along a specified mode.

    The function first permutes the tensor's axes so that the dimension
    corresponding to `mode` becomes the last dimension. The tensor is then
    reshaped into a 2D matrix where all dimensions *except* the specified mode
    are combined into the rows, and the specified mode's dimension forms the
    columns. QR decomposition (using 'economic' mode) is applied to this matrix.
    The resulting Q matrix is then reshaped back into a tensor and its axes
    are inverse-permuted to match the original tensor's ordering.

    Args:
        tensor (np.ndarray): The input tensor of arbitrary dimensions (rank >= 1).
        mode (int): The index of the axis (mode) along which the QR decomposition
                    is to be performed. This axis will become the 'column' dimension
                    of the reshaped matrix. Must be between 0 and `tensor.ndim - 1`.

    Returns:
        tuple: A tuple containing:
            - Q_tensor (np.ndarray): The orthogonal tensor, reshaped from Q_matrix,
                                     with its original axis order restored. Its shape
                                     will be the original tensor's shape with the `mode`
                                     dimension replaced by `bond_dim`.
            - R_matrix (np.ndarray): The upper triangular matrix from the QR decomposition.

    Raises:
        ValueError: If `mode` is not a valid axis index for the input tensor.
    """
    rank = tensor.ndim
    if not (0 <= mode < rank):
        raise ValueError(f"Mode must be an integer between 0 and {rank - 1}.")

    other_indices = [i for i in range(rank) if i != mode]
    permute_map = other_indices + [mode]
    permuted_tensor = tensor.transpose(permute_map)

    permuted_shape = permuted_tensor.shape
    row_dim = np.prod(permuted_shape[:-1]).astype(int)
    col_dim = permuted_shape[-1]
    matrix = permuted_tensor.reshape(row_dim, col_dim)

    Q_matrix, R_matrix = sp.linalg.qr(matrix, mode='economic')

    bond_dim = Q_matrix.shape[1]
    q_tensor_shape_permuted = permuted_shape[:-1] + (bond_dim,)
    Q_tensor_permuted = Q_matrix.reshape(q_tensor_shape_permuted)

    final_q_axes = other_indices + [mode]
    inv_permute_map = np.argsort(final_q_axes)
    Q_tensor = Q_tensor_permuted.transpose(inv_permute_map)

    return Q_tensor, R_matrix

def verify_isometry(iso_tensor, mode):
    """
    Verifies if a given tensor is an isometry along a specified mode.

    An isometry (or an isometric tensor) is a tensor Q such that when contracted
    with its conjugate transpose along all modes except one (the `mode`),
    the result is an identity matrix in the remaining mode. This function
    checks this property by performing a tensor contraction and comparing the
    result with an identity matrix of appropriate size.

    Args:
        iso_tensor (np.ndarray): The input tensor to be checked for isometry.
        mode (int): The mode (axis) along which the tensor is expected to be
                    an isometry. All other modes will be contracted. Must be
                    between 0 and `iso_tensor.ndim - 1`.

    Returns:
        bool: True if the tensor is an isometry along the specified mode
              (within numerical tolerance), False otherwise.
    """
    rank = iso_tensor.ndim
    contract_indices = [i for i in range(rank) if i != mode]
    contraction_result = np.tensordot(iso_tensor, iso_tensor.conj(), axes=(contract_indices, contract_indices))
    identity = np.eye(iso_tensor.shape[mode])

    return np.allclose(contraction_result, identity)



    return Q_tensor, R_matrix, R_tensor

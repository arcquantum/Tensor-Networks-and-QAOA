import numpy as np
import scipy as sp
import matplotlib.py as plt
import tensordecomposition.py

def quantum_state_to_mps(quantum_state=None, lattice_size=None, phys_dim=2):
    """
    Converts a given 1D quantum state vector into a Matrix Product State (MPS)
    representation using successive Singular Value Decompositions (SVDs).
    Alternatively, if no quantum state is provided, it initializes a simple
    product state (all |0> state) of a specified lattice size.

    For a given quantum state, the function iteratively reshapes the remainder
    of the state into a matrix, performs SVD, and extracts an MPS tensor and
    updates the remainder for the next site. The output MPS is in right-canonical
    form (normalized, with left-most tensors being isometries).

    Args:
        quantum_state (np.ndarray, optional): A 1D NumPy array representing
                                              the quantum state vector. Its size
                                              must be an integer power of `phys_dim`.
                                              If None, a product state is generated.
        lattice_size (int, optional): The number of sites in the MPS. Required
                                      if `quantum_state` is None. Must be a positive integer.
        phys_dim (int, optional): The local physical dimension of each site.
                                  Defaults to 2 (e.g., for qubits).

    Returns:
        list: A list of NumPy arrays, where each array `A_i` represents the
              MPS tensor for site `i`. Each tensor `A_i` has dimensions
              (bond_dim_left, phys_dim, bond_dim_right).

    Raises:
        ValueError: If `lattice_size` is not positive when `quantum_state` is None,
                    if `quantum_state` size is incompatible with `phys_dim`,
                    or if `quantum_state` has zero norm.
        TypeError: If `quantum_state` is not a 1D NumPy array when provided.
    """
    if quantum_state is None:
        if lattice_size is None or lattice_size <= 0:
            raise ValueError("Lattice size must be a positive integer for the default state.")

        mps_tensors = []
        for i in range(lattice_size):
            tensor = np.zeros((1, phys_dim, 1), dtype=complex)
            tensor[0, 0, 0] = 1.0
            mps_tensors.append(tensor)
        return mps_tensors

    if not isinstance(quantum_state, np.ndarray) or quantum_state.ndim != 1:
        raise TypeError("quantum_state must be a 1D NumPy array.")

    N = int(np.round(np.log(quantum_state.size) / np.log(phys_dim)))
    if not np.isclose(phys_dim ** N, quantum_state.size):
        raise ValueError("Size of quantum_state is not compatible with phys_dim.")

    norm = np.linalg.norm(quantum_state)
    if np.isclose(norm, 0):
        raise ValueError("Input quantum state has zero norm.")

    remainder = quantum_state / norm

    mps_tensors = []
    left_bond_dim = 1

    for i in range(N - 1):
        matrix = remainder.reshape(left_bond_dim * phys_dim, -1)

        U, S, Vh = sp.linalg.svd(matrix, full_matrices=False)

        new_bond_dim = S.shape[0]

        new_tensor = U.reshape(left_bond_dim, phys_dim, new_bond_dim)
        mps_tensors.append(new_tensor)

        remainder = np.diag(S) @ Vh
        left_bond_dim = new_bond_dim

    last_tensor = remainder.reshape(left_bond_dim, phys_dim, 1)
    mps_tensors.append(last_tensor)

    return mps_tensors

def mps_to_full_state_vector(mps_tensors):
    """
    Converts a Matrix Product State (MPS) representation back into its
    equivalent full quantum state vector.

    The function iteratively contracts the MPS tensors to build up the
    full state vector. The resulting vector is normalized.

    Args:
        mps_tensors (list): A list of NumPy arrays, where each array `A_i`
                            represents an MPS tensor for site `i`, with
                            dimensions (bond_dim_left, phys_dim, bond_dim_right).

    Returns:
        np.ndarray: A 1D NumPy array representing the normalized full quantum
                    state vector. Returns an empty array if `mps_tensors` is empty.
    """
    if not mps_tensors:
        return np.array([])

    N = len(mps_tensors)

    state = mps_tensors[0]

    for i in range(1, N):

        left_dim, phys_dim_state, _ = state.shape
        _, phys_dim_next, right_dim_next = mps_tensors[i].shape

        state = np.einsum('apb,bqd->apqd', state, mps_tensors[i])

        new_phys_dim = phys_dim_state * phys_dim_next
        state = state.reshape(left_dim, new_phys_dim, right_dim_next)

    final_vector = state.flatten()

    norm = np.linalg.norm(final_vector)
    if np.isclose(norm, 0):
        return final_vector
    else:
        return final_vector / norm

def rand_MPS_init(N, m, d):
    """
    Initializes a random Matrix Product State (MPS) with complex entries.

    Each MPS tensor `A_i` is initialized with random real and imaginary parts
    drawn from a uniform distribution. The bond dimensions are set to `m`,
    except for the edges where they are 1.

    Args:
        N (int): The number of sites in the MPS. Must be a positive integer.
        m (int): The maximum bond dimension for the MPS. Must be a positive integer.
        d (int): The local physical dimension of each site. Must be a positive integer.

    Returns:
        list: A list of NumPy arrays, where each array `A_i` represents a
              random MPS tensor for site `i`, with dimensions
              (bond_dim_left, d, bond_dim_right).

    Raises:
        ValueError: If N, m, or d are not positive integers.
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("Number of sites N must be a positive integer.")
    if not isinstance(m, int) or m <= 0:
        raise ValueError("Bond dimension m must be a positive integer.")
    if not isinstance(d, int) or d <= 0:
        raise ValueError("Physical dimension d must be a positive integer.")

    mps_tensors = []

    for i in range(N):
        left_dim = 1 if i == 0 else m
        right_dim = 1 if i == N - 1 else m

        shape = (left_dim, d, right_dim)

        real_part = np.random.rand(*shape)
        imag_part = np.random.rand(*shape)
        tensor = real_part + 1j * imag_part

        mps_tensors.append(tensor)

    return mps_tensors

def make_left_canon(mps_tensors):
    """
    Transforms a list of MPS tensors into left-canonical form.

    This process involves performing QR decomposition on each tensor from
    left to right (excluding the last one), making the current tensor
    left-isometric (Q) and passing the R matrix to the next tensor.
    The last tensor absorbs all remaining normalization and is then normalized.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.
                            Each tensor `A_i` should have shape
                            (bond_dim_left, phys_dim, bond_dim_right).

    Returns:
        tuple: A tuple containing:
            - left_mps (list): The list of MPS tensors in left-canonical form.
            - original_norm (float): The norm of the original MPS before canonicalization.

    Raises:
        TypeError: If `mps_tensors` is not a list of NumPy arrays.
    """
    if not isinstance(mps_tensors, list) or not all(isinstance(t, np.ndarray) for t in mps_tensors):
        raise TypeError("mps_tensors must be a list of NumPy arrays.")
    if len(mps_tensors) == 0:
        return [], 1.0

    left_mps = [t.copy() for t in mps_tensors]
    N = len(left_mps)

    for i in range(N - 1):
        tensor = left_mps[i]

        Q_tensor, R_matrix, _ = qr_tensor(tensor, 2)

        left_mps[i] = Q_tensor
        left_mps[i+1] = np.einsum('ab,bcd->acd', R_matrix, left_mps[i+1])

    last_tensor = left_mps[N-1]
    norm = np.linalg.norm(last_tensor)

    if np.isclose(norm, 0):
        left_mps[N-1] = last_tensor * 0
        original_norm = 0.0
    else:
        left_mps[N-1] = last_tensor / norm
        original_norm = norm

    return left_mps, original_norm

def verify_left_canon(mps_tensors):
    """
    Verifies if an MPS is in left-canonical form.

    An MPS is in left-canonical form if all tensors (except possibly the
    rightmost one) satisfy the isometry condition:
    sum_{s,j} A_{i,s,j}^* A_{i,s,k} = delta_{j,k} (identity matrix on the right bond).

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.

    Returns:
        bool: True if the MPS is in left-canonical form (within numerical tolerance),
              False otherwise. Prints a message indicating success or failure.
    """
    N = len(mps_tensors)
    for i in range(N - 1):
        tensor = mps_tensors[i]
        _, _, right_dim = tensor.shape

        contraction = np.einsum('isj,isk->jk', tensor.conj(), tensor)

        identity = np.eye(right_dim)
        if not np.allclose(contraction, identity):
            print(f"Verification FAILED at site {i}.")
            print("Contraction result:\n", contraction)
            return False

    print("Verification SUCCESSFUL: MPS is in left-canonical form.")
    return True

def make_right_canon(mps_tensors):
    """
    Transforms a list of MPS tensors into right-canonical form.

    This process involves performing RQ decomposition on each tensor from
    right to left (excluding the first one), making the current tensor
    right-isometric (Q) and passing the R matrix to the previous tensor.
    The first tensor absorbs all remaining normalization and is then normalized.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.
                            Each tensor `A_i` should have shape
                            (bond_dim_left, phys_dim, bond_dim_right).

    Returns:
        tuple: A tuple containing:
            - right_mps (list): The list of MPS tensors in right-canonical form.
            - original_norm (float): The norm of the original MPS before canonicalization.

    Raises:
        TypeError: If `mps_tensors` is not a list of NumPy arrays.
    """
    if not isinstance(mps_tensors, list) or not all(isinstance(t, np.ndarray) for t in mps_tensors):
        raise TypeError("mps_tensors must be a list of NumPy arrays.")
    if len(mps_tensors) == 0:
        return [], 1.0

    right_mps = [t.copy() for t in mps_tensors]
    N = len(right_mps)

    for i in range(N - 1, 0, -1):
        tensor = right_mps[i]

        Q_tensor, R_matrix, _ = rq_tensor(tensor, 1)

        right_mps[i] = Q_tensor
        right_mps[i-1] = np.einsum('abc,cd->abd', right_mps[i-1], R_matrix)

    first_tensor = right_mps[0]
    norm = np.linalg.norm(first_tensor)

    if np.isclose(norm, 0):
        right_mps[0] = first_tensor * 0
        original_norm = 0.0
    else:
        right_mps[0] = first_tensor / norm
        original_norm = norm

    return right_mps, original_norm

def verify_right_canon(mps_tensors):
    """
    Verifies if an MPS is in right-canonical form.

    An MPS is in right-canonical form if all tensors (except possibly the
    leftmost one) satisfy the isometry condition:
    sum_{s,i} A_{i,s,j} A_{k,s,j}^* = delta_{i,k} (identity matrix on the left bond).

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.

    Returns:
        bool: True if the MPS is in right-canonical form (within numerical tolerance),
              False otherwise. Prints a message indicating success or failure.
    """
    N = len(mps_tensors)
    for i in range(1, N):
        tensor = mps_tensors[i]
        left_dim, _, _ = tensor.shape

        contraction = np.einsum('isj,ksj->ik', tensor, tensor.conj())

        identity = np.eye(left_dim)
        if not np.allclose(contraction, identity):
            print(f"Verification FAILED at site {i}.")
            print("Contraction result:\n", contraction)
            return False

        print("Verification SUCCESSFUL: MPS is in right-canonical form.")
        return True

    print("Verification SUCCESSFUL: MPS is in right-canonical form.")
    return True

def make_mixed_canon(mps_tensors, center):
    """
    Transforms a Matrix Product State (MPS) into a mixed-canonical form,
    where tensors to the left of the `center` are left-canonical,
    and tensors to the right of the `center` are right-canonical.
    The tensor at the `center` is not necessarily isometric but contains
    the total norm of the state.

    The process first right-canonicalizes the entire MPS, then proceeds
    from left to `center`, left-canonicalizing tensors and transferring
    the R matrices to the right.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.
                            Each tensor `A_i` should have shape
                            (bond_dim_left, phys_dim, bond_dim_right).
        center (int): The index of the site that will be the orthogonality
                      center. This tensor will contain the norm. Must be
                      between 0 and `len(mps_tensors) - 1`.

    Returns:
        list: The list of MPS tensors in mixed-canonical form.

    Raises:
        ValueError: If `center` is not a valid index for the MPS.
    """
    N = len(mps_tensors)
    if not (0 <= center < N):
        raise ValueError(f"Center must be a valid index (0 <= center < {N}).")

    mixed_mps, _ = make_right_canon(mps_tensors)

    for i in range(center):
        tensor = mixed_mps[i]

        Q_tensor, R_matrix, _ = qr_tensor(tensor, 2)

        mixed_mps[i] = Q_tensor

        next_tensor = mixed_mps[i+1]
        mixed_mps[i+1] = np.einsum('ab,bcd->acd', R_matrix, next_tensor)

    return mixed_mps

def verify_mixed_canon(mps_tensors, center):
    """
    Verifies if an MPS is in mixed-canonical form with a specified center.

    This function checks if tensors to the left of `center` are left-canonical
    and tensors to the right of `center` are right-canonical. The tensor at
    the `center` is not checked for canonicality.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.
        center (int): The expected index of the orthogonality center.

    Returns:
        bool: True if the MPS satisfies the mixed-canonical conditions
              (within numerical tolerance) at the given center, False otherwise.
              Prints messages indicating success or failure and the specific
              site of failure if applicable.
    """
    N = len(mps_tensors)

    for i in range(center):
        tensor = mps_tensors[i]
        _, _, right_dim = tensor.shape
        contraction = np.einsum('isj,isk->jk', tensor.conj(), tensor)
        identity = np.eye(right_dim)
        if not np.allclose(contraction, identity):
            print(f"Verification FAILED: Site {i} is not left-normalized.")
            return False

    for i in range(center + 1, N):
        tensor = mps_tensors[i]
        left_dim, _, _ = tensor.shape
        contraction = np.einsum('isj,ksj->ik', tensor, tensor.conj())
        identity = np.eye(left_dim)
        if not np.allclose(contraction, identity):
            print(f"Verification FAILED: Site {i} is not right-normalized.")
            return False

    print(f"Verification SUCCESSFUL: MPS is in mixed-canonical form with center at site {center}.")
    return True

def _is_left_normalized(tensor, tol=1e-12):
    """
    Internal helper function to check if a single MPS tensor is left-normalized (isometric).

    A tensor A_{i,s,j} is left-normalized if sum_{s,j} A^*_{i,s,j} A_{k,s,j} = delta_{i,k}.
    In matrix form, if A is reshaped to (i, sj), then A A^dagger = I (where dagger is conjugate transpose).
    This function reshapes the tensor into a matrix (left_bond_dim * phys_dim, right_bond_dim)
    and checks if U^dagger U = I, where U is this reshaped matrix.

    Args:
        tensor (np.ndarray): The MPS tensor to check, with shape
                             (bond_dim_left, phys_dim, bond_dim_right).
        tol (float): Tolerance for numerical comparison.

    Returns:
        bool: True if the tensor is left-normalized within tolerance, False otherwise.
              Returns False if the tensor is not 3-dimensional.
    """
    if tensor.ndim != 3: return False

    U = tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])

    contraction = U.conj().T @ U
    identity = np.eye(U.shape[1])

    return np.allclose(contraction, identity, atol=tol)

def _is_right_normalized(tensor, tol=1e-12):
    """
    Internal helper function to check if a single MPS tensor is right-normalized (isometric).

    A tensor A_{i,s,j} is right-normalized if sum_{s,i} A_{i,s,j} A^*_{i,s,k} = delta_{j,k}.
    In matrix form, if A is reshaped to (i, sk), then A A^dagger = I.
    This function reshapes the tensor into a matrix (left_bond_dim, phys_dim * right_bond_dim)
    and checks if B B^dagger = I, where B is this reshaped matrix.

    Args:
        tensor (np.ndarray): The MPS tensor to check, with shape
                             (bond_dim_left, phys_dim, bond_dim_right).
        tol (float): Tolerance for numerical comparison.

    Returns:
        bool: True if the tensor is right-normalized within tolerance, False otherwise.
              Returns False if the tensor is not 3-dimensional.
    """
    if tensor.ndim != 3: return False

    B = tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2])

    contraction = B @ B.conj().T
    identity = np.eye(B.shape[0])

    return np.allclose(contraction, identity, atol=tol)

def find_orthogonality_center(mps_tensors, tol=1e-12):
    """
    Identifies the orthogonality center of an MPS by checking the canonical form
    of its constituent tensors.

    An orthogonality center `l` is found if all tensors to its left (`0` to `l-1`)
    are left-normalized and all tensors to its right (`l+1` to `N-1`) are
    right-normalized.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.
        tol (float): Tolerance for numerical comparison when checking isometry.

    Returns:
        int or None: The index of the orthogonality center if found, otherwise None.
                     If the MPS is empty, returns None. If it has only one tensor,
                     returns 0.
    """
    N = len(mps_tensors)
    if N == 0:
        return None
    if N == 1:
        return 0

    left_checks = [_is_left_normalized(t, tol) for t in mps_tensors]
    right_checks = [_is_right_normalized(t, tol) for t in mps_tensors]

    for l in range(N):

        is_left_part_ok = all(left_checks[0:l])

        is_right_part_ok = all(right_checks[l+1:N])

        if is_left_part_ok and is_right_part_ok:
            return l

    return None

def update_orthogonality_center(mps_tensors, old_center, new_center):
    """
    Shifts the orthogonality center of an MPS from `old_center` to `new_center`.

    This function uses sequential QR or RQ decompositions to move the norm
    from the `old_center` to the `new_center`. If moving right, it applies
    QR decomposition. If moving left, it applies RQ decomposition.
    The tensors between `old_center` and `new_center` become isometric.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.
                            The MPS is assumed to be in mixed-canonical form with
                            `old_center` being the current orthogonality center.
        old_center (int): The current index of the orthogonality center.
        new_center (int): The desired new index for the orthogonality center.

    Returns:
        list: The list of MPS tensors with the orthogonality center shifted
              to `new_center`.

    Raises:
        ValueError: If `old_center` or `new_center` are not valid indices for the MPS.
    """
    N = len(mps_tensors)
    if not (0 <= old_center < N and 0 <= new_center < N):
        raise ValueError(f"Centers must be valid indices (0 to {N-1}).")

    updated_mps = [t.copy() for t in mps_tensors]

    if old_center == new_center:
        return updated_mps

    if new_center > old_center:
        for i in range(old_center, new_center):
            tensor = updated_mps[i]

            Q_tensor, R_matrix, _ = qr_tensor(tensor, 2)

            updated_mps[i] = Q_tensor
            updated_mps[i+1] = np.einsum('ab,bcd->acd', R_matrix, updated_mps[i+1])

    elif new_center < old_center:
        for i in range(old_center, new_center, -1):
            tensor = updated_mps[i]

            Q_tensor, R_matrix, _ = rq_tensor(tensor, 1)

            updated_mps[i] = Q_tensor
            updated_mps[i-1] = np.einsum('abc,cd->abd', updated_mps[i-1], R_matrix)

    return updated_mps

def calculate_norm(mps_tensors):
    """
    Calculates the norm of an MPS using the "zipper" algorithm (also known
    as the transfer matrix method or calculation of the contraction of
    the MPS with its conjugate).

    This method efficiently computes the norm squared by iteratively contracting
    the MPS tensors with their complex conjugates from left to right,
    resulting in a scalar value.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.

    Returns:
        float: The L2 norm of the MPS. Returns 1.0 if the MPS is empty.
    """
    N = len(mps_tensors)
    if N == 0:
        return 1.0

    boundary = np.array([[1.0]], dtype=complex)

    for tensor in mps_tensors:
        boundary = np.einsum('ik,ksl,isj->jl', boundary, tensor, tensor.conj())

    norm_sq = boundary[0, 0]

    return np.sqrt(norm_sq.real)

def calculate_amplitude(mps_tensors, configuration):
    """
    Calculates the amplitude of a specific basis state in an MPS.

    The amplitude is computed by sequentially contracting each MPS tensor
    with the corresponding physical index from the `configuration`,
    and accumulating the result.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.
        configuration (list or tuple): A sequence of integers representing the
                                       physical indices of the basis state. The
                                       length must match the number of sites in
                                       the MPS, and each index must be valid
                                       for the physical dimension of the MPS.

    Returns:
        complex: The complex amplitude of the specified basis state.
                 Returns 1.0 + 0.0j if the MPS is empty.

    Raises:
        ValueError: If the length of `configuration` does not match the number
                    of sites, or if any index in `configuration` is out of bounds
                    for the physical dimension.
    """
    N = len(mps_tensors)
    if N == 0:
        return 1.0 + 0.0j

    if len(configuration) != N:
        raise ValueError("Length of configuration must match the number of sites in the MPS.")

    phys_dim = mps_tensors[0].shape[1]
    if any(s < 0 or s >= phys_dim for s in configuration):
        raise ValueError(f"All values in configuration must be valid physical indices (0 to {phys_dim-1}).")

    amplitude_vec = np.array([[1.0]], dtype=complex)

    for i in range(N):
        tensor = mps_tensors[i]
        phys_index = configuration[i]

        matrix = tensor[:, phys_index, :]

        amplitude_vec = amplitude_vec @ matrix

    return amplitude_vec[0, 0]

def inner_product(mps_bra, mps_ket):
    """
    Calculates the inner product <mps_bra | mps_ket> of two Matrix Product States
    using the "zipper" algorithm.

    This method efficiently computes the inner product by iteratively contracting
    tensors from `mps_ket` with the complex conjugates of corresponding tensors
    from `mps_bra` from left to right.

    Args:
        mps_bra (list): A list of NumPy arrays representing the "bra" MPS (conjugate).
        mps_ket (list): A list of NumPy arrays representing the "ket" MPS.

    Returns:
        complex: The complex inner product <mps_bra | mps_ket>.
                 Returns 1.0 + 0.0j if both MPS are empty.

    Raises:
        ValueError: If the two MPS have a different number of sites.
    """
    if len(mps_bra) != len(mps_ket):
        raise ValueError("The two MPS must have the same number of sites.")
    N = len(mps_ket)

    if N == 0:
        return 1.0 + 0.0j

    boundary = np.array([[1.0]], dtype=complex)

    for i in range(N):
        tensor_ket = mps_ket[i]
        tensor_bra_conj = mps_bra[i].conj()

        boundary = np.einsum('ik,ksl,isj->jl', boundary, tensor_ket, tensor_bra_conj)

    return boundary[0, 0]

def normalize_mps(mps_tensors):
    """
    Normalizes a Matrix Product State (MPS) so that its overall norm is 1.

    This function attempts to find an existing orthogonality center. If found,
    it normalizes the tensor at that center. If no orthogonality center is
    found (meaning the MPS is not in a canonical form), it falls back to
    performing a full left-canonicalization, which normalizes the MPS.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.

    Returns:
        tuple: A tuple containing:
            - normalized_mps (list): The list of MPS tensors normalized to unit norm.
            - norm (float): The original norm of the MPS before normalization.
    """
    if not mps_tensors:
        return [], 1.0

    center = find_orthogonality_center(mps_tensors)

    if center is not None:
        normalized_mps = [t.copy() for t in mps_tensors]
        center_tensor = normalized_mps[center]
        norm = np.linalg.norm(center_tensor)

        if np.isclose(norm, 0):
            return normalized_mps, 0.0

        normalized_mps[center] = center_tensor / norm
        return normalized_mps, norm
    else:
        return make_left_canon(mps_tensors)

def calculate_expectation_value(mps_tensors, hamiltonian_terms):
    """
    Calculates the expectation value of a Hamiltonian (represented as a sum
    of Pauli strings) with respect to a given Matrix Product State (MPS).

    The Hamiltonian is defined as a sum of terms, where each term is a
    coefficient multiplied by a Pauli string (e.g., "XIZ" for site 0 X, site 1 I, site 2 Z).
    The expectation value <psi|H|psi> = sum_k coeff_k * <psi|Pauli_k|psi>.
    Each term <psi|Pauli_k|psi> is calculated by applying the Pauli operators
    to a copy of the MPS and then computing the inner product with the original MPS.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.
                            The MPS is assumed to be normalized.
        hamiltonian_terms (list): A list of tuples, where each tuple contains
                                  (coefficient (complex or float), Pauli string (str)).
                                  Example: [(1.0, "XXI"), (0.5, "IZZ")].
                                  Valid Pauli characters are 'I', 'X', 'Y', 'Z'.

    Returns:
        float: The real part of the total expectation value of the Hamiltonian.

    Raises:
        ValueError: If a Pauli string's length does not match the MPS length,
                    or if an invalid Pauli character is encountered.
    """
    if not mps_tensors:
        return 0.0

    pauli_X = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_Z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_map = {'I': np.eye(2, dtype=complex), 'X': pauli_X, 'Y': pauli_Y, 'Z': pauli_Z}

    total_energy = 0.0

    for coeff, pauli_string in hamiltonian_terms:
        if len(pauli_string) != len(mps_tensors):
            raise ValueError(
                f"Pauli string '{pauli_string}' has length {len(pauli_string)}, "
                f"but the MPS has length {len(mps_tensors)}."
            )

        psi_prime_mps = [t.copy() for t in mps_tensors]

        for site_idx, pauli_char in enumerate(pauli_string):
            if pauli_char not in pauli_map:
                raise ValueError(f"Invalid Pauli character '{pauli_char}' in string '{pauli_string}'.")
            gate_matrix = pauli_map[pauli_char]

            psi_prime_mps[site_idx] = np.einsum(
                'isj,ks->ikj', psi_prime_mps[site_idx], gate_matrix
            )

        expectation_of_term = inner_product(mps_tensors, psi_prime_mps)

        total_energy += coeff * expectation_of_term

    return np.real(total_energy)

def build_full_hamiltonian_matrix(hamiltonian_terms, num_sites):
    """
    Constructs the full (dense) Hamiltonian matrix from a list of Pauli string terms.

    This function is primarily for verification or for small systems, as the
    full matrix scales exponentially with the number of sites. Each Pauli string
    is converted into a Kronecker product of Pauli matrices (or identity) across
    all sites, and then scaled by its coefficient and added to the total Hamiltonian.

    Args:
        hamiltonian_terms (list): A list of tuples, where each tuple contains
                                  (coefficient (complex or float), Pauli string (str)).
                                  Example: [(1.0, "XXI"), (0.5, "IZZ")].
                                  Valid Pauli characters are 'I', 'X', 'Y', 'Z'.
        num_sites (int): The number of quantum sites (e.g., qubits). This defines
                         the dimension of the Hilbert space.

    Returns:
        np.ndarray: The full (dense) complex-valued Hamiltonian matrix.

    Raises:
        ValueError: If a Pauli string's length does not match `num_sites`,
                    or if an invalid Pauli character is encountered.
    """
    phys_dim = 2
    matrix_size = phys_dim**num_sites
    full_H = np.zeros((matrix_size, matrix_size), dtype=complex)

    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    for coeff, pauli_string in hamiltonian_terms:
        if len(pauli_string) != num_sites:
            raise ValueError(
                f"Pauli string '{pauli_string}' has length {len(pauli_string)}, "
                f"but expected length {num_sites} for the given number of sites."
            )
        term_matrix = np.array([[1.0]], dtype=complex)
        for pauli_char in pauli_string:
            if pauli_char not in pauli_map:
                raise ValueError(f"Invalid Pauli character '{pauli_char}' in string '{pauli_string}'.")
            term_matrix = np.kron(term_matrix, pauli_map[pauli_char])
        full_H += coeff * term_matrix

    return full_H

def vidal_gauge_using_right_canon(mps_tensors):
    """
    Converts an MPS into the Vidal gauge (Gamma-Lambda representation) by
    first right-canonicalizing the entire MPS.

    In Vidal gauge, each MPS tensor A[i] is decomposed as Gamma[i] * Lambda[i],
    where Gamma[i] are isometries and Lambda[i] are diagonal matrices
    containing the singular values (bond dimensions). This function first
    ensures the MPS is right-canonical, then iteratively applies SVD from
    left to right.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.

    Returns:
        tuple: A tuple containing:
            - right_canonical_mps (list): The original MPS in right-canonical form.
            - vidal_mps (list): A list of tuples `(gamma_tensor, lambda_vector)`,
                                 where `gamma_tensor` is a NumPy array (Gamma matrix)
                                 and `lambda_vector` is a 1D NumPy array (diagonal of Lambda matrix).
    """
    N = len(mps_tensors)
    if N == 0:
        return [], []

    right_canonical_mps, _ = make_right_canon(mps_tensors)

    temp_mps = [t.copy() for t in right_canonical_mps]

    gammas = []
    lambdas = []

    for i in range(N - 1):
        tensor = temp_mps[i]

        U_tensor, S_matrix, _, S_Vh_matrix, _ = svd_tensor(tensor, 2)

        gammas.append(U_tensor)
        lambdas.append(np.diag(S_matrix))

        next_tensor = temp_mps[i+1]
        temp_mps[i+1] = np.einsum('ab,bcd->acd', S_Vh_matrix, next_tensor)

    last_tensor = temp_mps[N-1]

    U_tensor, S_matrix, _, _, _ = svd_tensor(last_tensor, 2)

    final_gamma = U_tensor
    final_lambda = np.diag(S_matrix)

    gammas.append(final_gamma)
    lambdas.append(final_lambda)

    norm = final_lambda[0]
    lambdas[-1] = np.array([1.0])

    if not np.isclose(norm, 0):
        lambdas[0] = lambdas[0] / norm

    vidal_mps = list(zip(gammas, lambdas))

    return right_canonical_mps, vidal_mps

def reconstruct_from_vidal_gauge_using_right_canon(vidal_mps):
    """
    Reconstructs a right-canonical MPS from its Vidal gauge (Gamma-Lambda) representation.

    Each (Gamma, Lambda) pair is combined to form a tensor A[i] = Gamma[i] * Lambda[i].
    The reconstructed MPS is then made explicitly right-canonical to ensure accuracy
    and normalization.

    Args:
        vidal_mps (list): A list of tuples `(gamma_tensor, lambda_vector)`,
                          representing the Vidal gauge.

    Returns:
        list: The reconstructed MPS in right-canonical form.
    """
    N = len(vidal_mps)
    if N == 0:
        return []

    left_canonical_mps = []
    for i in range(N):
        gamma, lamb = vidal_mps[i]

        lambda_matrix = np.diag(lamb)
        new_tensor = np.einsum('isj,jk->isk', gamma, lambda_matrix)
        left_canonical_mps.append(new_tensor)

    norm_factor = np.linalg.norm(left_canonical_mps[-1])
    if not np.isclose(norm_factor, 0):
        left_canonical_mps[-1] /= norm_factor

    reconstructed_right_canonical, _ = make_right_canon(left_canonical_mps)

    return reconstructed_right_canonical

def vidal_gauge(mps_tensors):
    """
    Converts an MPS into the Vidal gauge (Gamma-Lambda representation).

    This function performs successive SVDs from left to right to decompose
    each MPS tensor A[i] into a Gamma[i] (isometric tensor) and a Lambda[i]
    (diagonal matrix of singular values). The Lambda matrix from one SVD is
    multiplied into the next tensor. The last Gamma tensor is normalized,
    and the overall normalization is handled by scaling the first Lambda.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.

    Returns:
        list: A list of tuples `(gamma_tensor, lambda_vector)`,
              where `gamma_tensor` is a NumPy array (Gamma matrix)
              and `lambda_vector` is a 1D NumPy array (diagonal of Lambda matrix).
    """
    if not mps_tensors:
        return []

    N = len(mps_tensors)
    temp_mps = [t.copy() for t in mps_tensors]

    gammas = []
    lambdas = []

    for i in range(N - 1):
        tensor = temp_mps[i]

        U_tensor, S_matrix, _, S_Vh_matrix, _ = svd_tensor(tensor, 2)

        gammas.append(U_tensor)
        lambdas.append(np.diag(S_matrix))

        temp_mps[i+1] = np.einsum('ab,bcd->acd', S_Vh_matrix, temp_mps[i+1])

    last_tensor = temp_mps[N-1]
    norm = np.linalg.norm(last_tensor)

    if np.isclose(norm, 0):
        final_gamma = last_tensor * 0
    else:
        final_gamma = last_tensor / norm

    gammas.append(final_gamma)
    lambdas.append(np.array([1.0]))

    if not np.isclose(norm, 0):
        lambdas[0] = lambdas[0] / norm

    vidal_mps = list(zip(gammas, lambdas))

    return vidal_mps

def svd_compression_fixed_bond(mps_tensors, max_bond_dim):
    """
    Compresses an MPS by truncating its bond dimensions to a fixed maximum value
    using successive SVDs from right to left (after right-canonicalization).

    This method aims to reduce the computational cost and memory footprint of an MPS
    by discarding small singular values. The MPS is first right-canonicalized.
    Then, for each site from left to right (excluding the last), an SVD is performed,
    and the singular values are truncated to `max_bond_dim`. The discarded singular
    values contribute to a truncation error. The `S*Vh` part of the SVD is passed
    to the next tensor. The final MPS is normalized and right-canonical.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the original MPS tensors.
        max_bond_dim (int): The maximum allowed bond dimension after compression.

    Returns:
        tuple: A tuple containing:
            - normalized_mps (list): The compressed MPS, normalized and in right-canonical form.
            - unnormalized_mps (list): The compressed MPS before final normalization.
            - truncation_errors (list): A list of floats, where each element is the
                                        Frobenius norm of the discarded part of the
                                        singular values at each site, representing
                                        the local truncation error.
    """
    if not mps_tensors:
        return [], [], []

    N = len(mps_tensors)

    unnormalized_mps, _ = make_right_canon(mps_tensors)

    truncation_errors = []

    for i in range(N - 1):

        U_tensor, S_matrix, _, _, Vh_matrix = svd_tensor(unnormalized_mps[i], 2)

        s_vec = np.diag(S_matrix)
        d_old = len(s_vec)
        d_new = min(max_bond_dim, d_old)

        local_error_sq = np.sum(s_vec[d_new:]**2)
        truncation_errors.append(np.sqrt(local_error_sq))

        U_tr = U_tensor[:, :, :d_new]
        S_tr = S_matrix[:d_new, :d_new]
        Vh_tr = Vh_matrix[:d_new, :]

        unnormalized_mps[i] = U_tr

        S_Vh_matrix = S_tr @ Vh_tr
        unnormalized_mps[i+1] = np.einsum('ij,jkl->ikl', S_Vh_matrix, unnormalized_mps[i+1])

    normalized_mps, _ = make_right_canon([t.copy() for t in unnormalized_mps])

    return normalized_mps, unnormalized_mps, truncation_errors

def svd_compression_fixed_tol(mps_tensors, tol):
    """
    Compresses an MPS by truncating its bond dimensions based on a desired
    Frobenius norm truncation tolerance `tol`, using successive SVDs from
    right to left (after right-canonicalization).

    Similar to `svd_compression_fixed_bond`, this method reduces bond dimensions,
    but instead of a fixed maximum, it truncates singular values whose squared sum
    is less than `tol^2`. The MPS is first right-canonicalized. Then, for each
    site from left to right (excluding the last), an SVD is performed. Singular
    values are discarded if their cumulative sum of squares (from smallest)
    exceeds `tol^2`. The `S*Vh` part of the SVD is passed to the next tensor.
    The final MPS is normalized and right-canonical.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the original MPS tensors.
        tol (float): The maximum allowed Frobenius norm of the discarded singular
                     values at each site. Squared tolerance `tol**2` is used for comparison.

    Returns:
        tuple: A tuple containing:
            - normalized_mps (list): The compressed MPS, normalized and in right-canonical form.
            - unnormalized_mps (list): The compressed MPS before final normalization.
            - truncation_errors (list): A list of floats, where each element is the
                                        Frobenius norm of the discarded part of the
                                        singular values at each site.
    """
    if not mps_tensors:
        return [], [], []

    N = len(mps_tensors)

    unnormalized_mps, _ = make_right_canon(mps_tensors)

    truncation_errors = []

    for i in range(N - 1):
        U_tensor, S_matrix, Vh_tensor, _, _ = svd_tensor(unnormalized_mps[i], 2)
        s_vec = np.diag(S_matrix)

        if len(s_vec) == 0:
            truncation_errors.append(0.0)
            continue

        s_sq_rev = (s_vec**2)[::-1]
        err_cumsum_rev = np.cumsum(s_sq_rev)

        num_discard = np.searchsorted(err_cumsum_rev, tol**2)
        d_new = len(s_vec) - num_discard

        if num_discard > 0:
            local_error_sq = err_cumsum_rev[num_discard - 1]
        else:
            local_error_sq = 0.0
        truncation_errors.append(np.sqrt(local_error_sq))

        U_tr = U_tensor[:, :, :d_new]
        S_tr = S_matrix[:d_new, :d_new]
        Vh_tr = Vh_tensor[:d_new, :]

        unnormalized_mps[i] = U_tr

        S_Vh_matrix = S_tr @ Vh_tr
        unnormalized_mps[i+1] = np.einsum('ij,jkl->ikl', S_Vh_matrix, unnormalized_mps[i+1])

    normalized_mps, _ = make_right_canon([t.copy() for t in unnormalized_mps])

    return normalized_mps, unnormalized_mps, truncation_errors

def calculate_truncation_error_sum(trunc_errors):
    """
    Calculates the total truncation error for an MPS compression from a list
    of individual local truncation errors.

    The total truncation error (Frobenius norm) is the square root of the sum
    of the squares of the local truncation errors, assuming local errors are
    independent (which is an approximation for sequential SVDs).

    Args:
        trunc_errors (list): A list of floats, where each float is a local
                             truncation error (e.g., from `svd_compression`).

    Returns:
        float: The total truncation error.
    """
    return np.sqrt(np.sum(np.array(trunc_errors)**2))

def calculate_fidelity_error(mps_original, mps_compressed):
    """
    Calculates the infidelity between an original MPS and a compressed MPS.

    Infidelity is defined as 1 - |<original | compressed>|.
    Both MPS are first normalized to ensure the fidelity calculation is correct.

    Args:
        mps_original (list): A list of NumPy arrays representing the original MPS.
        mps_compressed (list): A list of NumPy arrays representing the compressed MPS.

    Returns:
        float: The infidelity between the two MPS.
    """
    norm_mps1, _ = make_right_canon(mps_original)
    norm_mps2, _ = make_right_canon(mps_compressed)

    ip = inner_product(norm_mps1, norm_mps2)

    fidelity = abs(ip)

    infidelity = 1.0 - fidelity

    return infidelity

def make_right_canon_unnorm(mps_tensors):
    """
    Transforms a list of MPS tensors into a right-canonical form, but without
    normalizing the leftmost tensor (site 0).

    This function is a variant of `make_right_canon` where the final normalization
    step for the first tensor is skipped. This can be useful in iterative algorithms
    like variational compression where the normalization is handled elsewhere or
    is not desired at this intermediate step.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.
                            Each tensor `A_i` should have shape
                            (bond_dim_left, phys_dim, bond_dim_right).

    Returns:
        list: The list of MPS tensors in right-canonical form (except for the
              first tensor, which is not normalized).

    Raises:
        TypeError: If `mps_tensors` is not a list of NumPy arrays.
    """
    if not isinstance(mps_tensors, list) or not all(isinstance(t, np.ndarray) for t in mps_tensors):
        raise TypeError("mps_tensors must be a list of NumPy arrays.")
    if len(mps_tensors) == 0:
        return []

    right_mps = [t.copy() for t in mps_tensors]
    N = len(right_mps)

    for i in range(N - 1, 0, -1):
        tensor = right_mps[i]
        Q_tensor, R_matrix, _ = rq_tensor(tensor, 1)
        right_mps[i] = Q_tensor
        right_mps[i-1] = np.einsum('abc,cd->abd', right_mps[i-1], R_matrix)

    return right_mps

def variational_compression(target_mps, guess_mps, max_sweeps=1000, tol=1e-25):
    """
    Compresses a `target_mps` to match the bond dimensions of a `guess_mps`
    (or lower if `guess_mps` has smaller bond dimensions) using a variational
    optimization (DMRG-like sweep).

    This algorithm iteratively optimizes the tensors of the `guess_mps` to
    maximize its overlap with the `target_mps`. It involves sweeps from left
    to right and right to left, calculating environment tensors and performing
    local optimizations. The `target_mps` is canonicalized and normalized.

    Args:
        target_mps (list): A list of NumPy arrays representing the MPS to be compressed.
                           Assumed to be normalized.
        guess_mps (list): A list of NumPy arrays representing the initial MPS with
                          the desired (typically smaller) bond dimensions. This MPS
                          will be optimized.
        max_sweeps (int, optional): The maximum number of full left-right and
                                    right-left sweeps to perform. Defaults to 1000.
        tol (float, optional): The convergence tolerance. The sweep stops if the
                               change in the `guess_mps` (measured by 1 - |overlap|)
                               between consecutive sweeps falls below this value.
                               Defaults to 1e-25.

    Returns:
        tuple: A tuple containing:
            - phi (list): The optimized `guess_mps` (normalized and right-canonical).
            - fidelity (float): The fidelity (overlap) between the `target_mps` and
                                the final optimized `guess_mps`.

    Raises:
        ValueError: If the `target_mps` and `guess_mps` do not have the same number of sites.
    """
    N = len(target_mps)
    if N != len(guess_mps):
        raise ValueError("Target and guess MPS must have the same length.")

    psi, _ = make_right_canon(target_mps)
    phi = make_right_canon_unnorm(guess_mps)

    left_envs = [np.array([[1.0]], dtype=complex)] * (N + 1)
    right_envs = [np.array([[1.0]], dtype=complex)] * (N + 1)

    for sweep in range(max_sweeps):
        phi_old = [t.copy() for t in phi]

        right_envs[N] = np.array([[1.0]], dtype=complex)
        for i in range(N - 1, -1, -1):
            right_envs[i] = np.einsum('pq,ksp,msq->km', right_envs[i+1], psi[i], phi[i].conj())

        for i in range(N - 1):
            b_tensor = np.einsum('km,ksp,pq->msq', left_envs[i], psi[i], right_envs[i+1])
            Q_tensor, R_matrix, _ = qr_tensor(b_tensor, 2)
            phi[i] = Q_tensor

            phi[i+1] = np.einsum('ab,bcd->acd', R_matrix, phi[i+1])

            left_envs[i+1] = np.einsum('km,ksp,msq->pq', left_envs[i], psi[i], phi[i].conj())

        for i in range(N - 1, 0, -1):
            b_tensor = np.einsum('km,ksp,pq->msq', left_envs[i], psi[i], right_envs[i+1])
            Q_tensor, R_matrix, _ = rq_tensor(b_tensor, 1)
            phi[i] = Q_tensor

            phi[i-1] = np.einsum('abc,cd->abd', phi[i-1], R_matrix)

            right_envs[i] = np.einsum('pq,ksp,msq->km', right_envs[i+1], psi[i], phi[i].conj())

    phi_normalized, _ = make_right_canon(phi)
    phi_old_normalized, _ = make_right_canon(phi_old)
    overlap = inner_product(phi_old_normalized, phi_normalized)
    state_change = 1 - abs(overlap)

    fidelity = abs(inner_product(psi, phi_normalized))

    if state_change < tol:
        phi = phi_normalized
        return phi, fidelity

    if sweep == max_sweeps - 1:
        print("\nWarning: Maximum number of sweeps reached without convergence.")
        phi = phi_normalized
        return phi, fidelity

    final_fidelity = abs(inner_product(psi, phi))
    return phi, final_fidelity

def apply_single_qubit_gate(mps_tensors, site_index, gate_matrix):
    """
    Applies a single-qubit gate to a specified site within an MPS.

    This function applies a given `gate_matrix` to the physical index of
    the MPS tensor at `site_index`. It returns the modified tensor and
    a new list of MPS tensors with the update applied.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.
        site_index (int): The index of the site to which the gate is applied.
                          Must be a valid index within the MPS length.
        gate_matrix (np.ndarray): A 2x2 complex NumPy array representing the
                                  single-qubit gate. Its dimensions must match
                                  the physical dimension of the MPS tensor.

    Returns:
        tuple: A tuple containing:
            - new_tensor (np.ndarray): The modified MPS tensor at `site_index`
                                       after applying the gate.
            - updated_mps (list): A new list of MPS tensors with the gate applied.

    Raises:
        ValueError: If `site_index` is out of bounds or if `gate_matrix`
                    dimensions do not match the physical dimension of the tensor.
    """
    if not 0 <= site_index < len(mps_tensors):
        raise ValueError(f"site_index {site_index} is out of bounds for an MPS of length {len(mps_tensors)}.")

    if gate_matrix.shape != (mps_tensors[site_index].shape[1], mps_tensors[site_index].shape[1]):
        raise ValueError("Gate matrix dimensions must match the physical dimension of the MPS tensor.")

    updated_mps = [t.copy() for t in mps_tensors]

    tensor_to_update = updated_mps[site_index]

    new_tensor = np.einsum('isj,ks->ikj', tensor_to_update, gate_matrix)

    updated_mps[site_index] = new_tensor

    return new_tensor, updated_mps

def apply_two_qubit_gate(mps_tensors, site_indices, gate_matrix):
    """
    Applies a two-qubit gate to two neighboring sites within an MPS.

    This function contracts the two MPS tensors at `site_indices` into a
    single larger tensor, applies the `gate_matrix` to their combined physical
    dimensions, and then re-decomposes the resulting tensor back into two
    MPS tensors using SVD. This operation changes the bond dimension between
    the two sites.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.
        site_indices (tuple): A tuple `(n1, n2)` representing the indices of
                              the two neighboring sites to which the gate is applied.
                              Must satisfy `n2 == n1 + 1`.
        gate_matrix (np.ndarray): A 4x4 (for `phys_dim=2`) complex NumPy array
                                  representing the two-qubit gate. Its dimensions
                                  must match the square of the physical dimension
                                  of the MPS tensors.

    Returns:
        tuple: A tuple containing:
            - big_tensor_before_svd (np.ndarray): The combined tensor of the two
                                                 sites after gate application,
                                                 before SVD decomposition.
            - intermediate_mps (list): The MPS with the two sites combined into
                                       a single 'big' tensor after gate application.
            - new_tensors_after_svd (tuple): A tuple `(new_m1, new_m2)` containing
                                            the two new MPS tensors after SVD.
            - final_mps (list): The new list of MPS tensors with the gate applied
                                and the sites re-decomposed.

    Raises:
        ValueError: If `site_indices` are not for neighboring sites, are out of bounds,
                    or if `gate_matrix` dimensions are incorrect.
    """
    n1, n2 = site_indices
    if n2 != n1 + 1:
        raise ValueError("site_indices must be for two neighboring sites (n, n+1).")
    if not (0 <= n1 < len(mps_tensors) - 1):
        raise ValueError(f"Site indices {site_indices} are out of bounds.")

    m1 = mps_tensors[n1]
    m2 = mps_tensors[n2]
    phys_dim = m1.shape[1]

    if gate_matrix.shape != (phys_dim**2, phys_dim**2):
        raise ValueError("Gate matrix dimensions are incorrect for the physical dimension of the MPS.")

    combined_tensor = np.einsum('aim,mjb->aijb', m1, m2)

    gate_tensor = gate_matrix.reshape(phys_dim, phys_dim, phys_dim, phys_dim)
    output_tensor = np.einsum('aijb,klij->aklb', combined_tensor, gate_tensor)

    big_tensor_before_svd = output_tensor.copy()
    intermediate_mps = mps_tensors[:n1] + [big_tensor_before_svd] + mps_tensors[n2+1:]

    U_tensor, S_matrix, _, _, Vh_matrix = svd_tensor(output_tensor, num_left_indices=2)

    new_m1 = U_tensor
    S_Vh_matrix = S_matrix @ Vh_matrix
    new_bond_dim = S_matrix.shape[0]
    right_bond_dim = m2.shape[2]
    new_m2 = S_Vh_matrix.reshape(new_bond_dim, phys_dim, right_bond_dim)

    new_tensors_after_svd = (new_m1, new_m2)
    final_mps = mps_tensors[:n1] + [new_m1, new_m2] + mps_tensors[n2+1:]

    return big_tensor_before_svd, intermediate_mps, new_tensors_after_svd, final_mps

def swap_sites_mps(mps_tensors, site1, site2):
    """
    Swaps the physical indices of two arbitrary sites in an MPS.

    This operation is implemented by repeatedly applying a two-qubit SWAP gate
    to neighboring sites to "bubble" one site past the other. This process
    involves two sequences of `apply_two_qubit_gate` operations.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.
        site1 (int): The index of the first site to swap.
        site2 (int): The index of the second site to swap.

    Returns:
        list: A new list of MPS tensors with the physical indices of `site1`
              and `site2` effectively swapped. The bond dimensions may change.

    Raises:
        ValueError: If `site1` or `site2` are out of bounds.
    """
    N = len(mps_tensors)
    if not (0 <= site1 < N and 0 <= site2 < N):
        raise ValueError("Site indices are out of bounds.")

    if site1 == site2:
        return [t.copy() for t in mps_tensors]

    if site1 > site2:
        site1, site2 = site2, site1

    current_mps = [t.copy() for t in mps_tensors]
    phys_dim = current_mps[0].shape[1]

    swap_gate = np.zeros((phys_dim**2, phys_dim**2), dtype=complex)
    for i in range(phys_dim):
        for j in range(phys_dim):
            k = i * phys_dim + j
            l = j * phys_dim + i
            swap_gate[l, k] = 1.0

    for i in range(site1, site2):
        _, _, _, current_mps = apply_two_qubit_gate(current_mps, (i, i + 1), swap_gate)

    for i in range(site2 - 1, site1, -1):
        _, _, _, current_mps = apply_two_qubit_gate(current_mps, (i - 1, i), swap_gate)

    return current_mps

def run_swap_test(num_sites, phys_dim, bond_dim, site1, site2):
    """
    Runs a test to verify the `swap_sites_mps` function by comparing the swapped
    MPS with a reference state vector obtained by direct permutation.

    It initializes a random MPS, applies random single-qubit gates to it,
    then uses `swap_sites_mps` to swap two specified sites. The resulting MPS
    is converted to a full state vector, which is then compared (via inner product)
    to a reference state vector obtained by directly permuting the indices
    of the initial full state vector.

    Args:
        num_sites (int): The number of sites (qubits) in the MPS.
        phys_dim (int): The local physical dimension of each site.
        bond_dim (int): The maximum bond dimension for the initial random MPS
                        (though `quantum_state_to_mps` determines bond_dim,
                        this parameter might be intended for `rand_MPS_init` if
                        used). This parameter is currently not directly used
                        in `quantum_state_to_mps`.
        site1 (int): The index of the first site to swap for the test.
        site2 (int): The index of the second site to swap for the test.

    Returns:
        None: Prints success or failure messages. Asserts that the overlap
              is close to 1.0.

    Raises:
        ValueError: Propagated from underlying MPS functions.
        AssertionError: If the overlap between the swapped MPS and the reference
                        state vector is not close to 1.0.
    """
    print(f"\n--- Testing swap between sites {site1} and {site2} ---")

    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    Hadamard = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    unique_gates = [X, Y, Z, Hadamard]

    product_mps = quantum_state_to_mps(lattice_size=num_sites, phys_dim=phys_dim)
    initial_mps = [t.copy() for t in product_mps]
    for i in range(num_sites):
        _, initial_mps = apply_single_qubit_gate(initial_mps, i, unique_gates[i % len(unique_gates)])

    initial_mps, _ = normalize_mps(initial_mps)

    psi_initial = mps_to_full_state_vector(initial_mps)

    mps_swapped = swap_sites_mps(initial_mps, site1, site2)
    psi_swapped = mps_to_full_state_vector(mps_swapped)
    print("Swap operation completed on MPS.")

    initial_tensor = psi_initial.reshape([phys_dim] * num_sites)

    perm_map = list(range(num_sites))
    perm_map[site1], perm_map[site2] = perm_map[site2], perm_map[site1]

    reference_tensor = initial_tensor.transpose(perm_map)
    psi_reference = reference_tensor.flatten()
    print("Reference swapped state vector created.")

    overlap = np.vdot(psi_reference, psi_swapped)
    print(f"   |<Reference|Swapped>| = {np.abs(overlap):.8f}")

    assert np.isclose(np.abs(overlap), 1.0), f"Verification FAILED for sites ({site1}, {site2})."
    print(f"SUCCESS: Swapping sites ({site1}, {site2}) is correct.")

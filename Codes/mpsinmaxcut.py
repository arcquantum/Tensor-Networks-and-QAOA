import numpy as np
import scipy as sp
import matplotlib.py as plt
import tensordecomposition.py
import mps.py

def calculate_cut_size(graph_edges, cut_bitstring):
    """
    Calculates the cut size for a given graph and a binary cut configuration.

    The cut size is defined as the number of edges connecting nodes that
    have different values in the `cut_bitstring`. This function is useful
    for evaluating the quality of a solution to a Max-Cut problem.

    Args:
        graph_edges (list): A list of tuples, where each tuple `(i, j)`
                            represents an edge between node `i` and node `j`.
        cut_bitstring (list or str): A sequence (list or string) of binary values
                                     (e.g., 0s and 1s) representing the partition
                                     of nodes. `cut_bitstring[k]` is the partition
                                     assignment for node `k`.

    Returns:
        int: The total number of edges whose endpoints are in different partitions.
    """
    cut_count = 0
    for i, j in graph_edges:
        if cut_bitstring[i] != cut_bitstring[j]:
            cut_count += 1
    return cut_count

def qaoa_maxcut_linear_mps(num_nodes, p):
    """
    Solves the Max-Cut problem for a linear graph (a chain) using the QAOA algorithm
    simulated with Matrix Product States (MPS).

    This function implements the Quantum Approximate Optimization Algorithm (QAOA)
    for a linear graph. It uses a classical optimizer to find optimal QAOA parameters
    (gammas and betas) by minimizing the expectation value of the cost Hamiltonian.
    The quantum circuit simulation is performed using MPS operations (applying
    single- and two-qubit gates). After optimization, it samples bitstrings from
    the final MPS to find the one with the highest probability, which corresponds
    to the approximate solution to Max-Cut.

    Args:
        num_nodes (int): The number of nodes in the linear graph (number of qubits).
        p (int): The number of QAOA layers (depth of the circuit).

    Returns:
        tuple: A tuple containing:
            - solution_bitstring (str): The binary string representing the cut
                                        with the highest probability found from the MPS.
            - final_energy (float): The expectation value of the cost Hamiltonian
                                    for the optimized MPS (this is typically negative
                                    for Max-Cut where 0 is minimum cut).
            - opt_result (scipy.optimize.OptimizeResult): The result object from
                                                          the classical optimization.
    """
    phys_dim = 2

    cost_hamiltonian = []
    for i in range(num_nodes - 1):
        pauli_list = ['I'] * num_nodes
        pauli_list[i] = 'Z'
        pauli_list[i+1] = 'Z'
        cost_hamiltonian.append((1.0, "".join(pauli_list)))

    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def objective_function(params):
        gammas = params[:p]
        betas = params[p:]

        current_mps = quantum_state_to_mps(lattice_size=num_nodes, phys_dim=phys_dim)

        H_gate = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
        for i in range(num_nodes):
            _, current_mps = apply_single_qubit_gate(current_mps, i, H_gate)

        for i in range(p):

            gamma = gammas[i]

            zz_gate_matrix = sp.linalg.expm(-1j * gamma * np.kron(Z, Z))
            for j in range(num_nodes - 1):

                _, _, _, current_mps = apply_two_qubit_gate(current_mps, (j, j+1), zz_gate_matrix)

            beta = betas[i]

            x_gate_matrix = sp.linalg.expm(-1j * beta * X)
            for j in range(num_nodes):
                _, current_mps = apply_single_qubit_gate(current_mps, j, x_gate_matrix)

        energy = calculate_expectation_value(current_mps, cost_hamiltonian)
        return energy

    print(f"Starting classical optimization for p={p} layers...")
    initial_params = np.random.uniform(0, 2*np.pi, 2 * p)
    opt_result = sp.optimize.minimize(objective_function, initial_params, method='COBYLA')
    optimal_params = opt_result.x
    print("Optimization finished.")

    final_energy = objective_function(optimal_params)

    gammas = optimal_params[:p]
    betas = optimal_params[p:]
    final_mps = quantum_state_to_mps(lattice_size=num_nodes, phys_dim=phys_dim)
    H_gate = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    for i in range(num_nodes):
        _, final_mps = apply_single_qubit_gate(final_mps, i, H_gate)

    for i in range(p):
        gamma = gammas[i]
        zz_gate_matrix = sp.linalg.expm(-1j * gamma * np.kron(Z, Z))
        for j in range(num_nodes - 1):
            _, _, _, final_mps = apply_two_qubit_gate(final_mps, (j, j+1), zz_gate_matrix)

        beta = betas[i]
        x_gate_matrix = sp.linalg.expm(-1j * beta * X)
        for j in range(num_nodes):
            _, final_mps = apply_single_qubit_gate(final_mps, j, x_gate_matrix)

    max_prob = -1.0
    solution_bitstring = ""

    for bitstring_tuple in it.product([0, 1], repeat=num_nodes):
        config = list(bitstring_tuple)
        amplitude = calculate_amplitude(final_mps, config)
        probability = np.abs(amplitude)**2
        if probability > max_prob:
            max_prob = probability
            solution_bitstring = "".join(map(str, config))

    return solution_bitstring, final_energy, opt_result

def get_max_bond_dimension(mps_tensors):
    """
    Calculates the maximum bond dimension present in a given list of MPS tensors.

    Args:
        mps_tensors (list): A list of NumPy arrays representing the MPS tensors.

    Returns:
        int: The largest bond dimension found among the MPS tensors. Returns 1
             if the MPS is empty or has only one site (no internal bonds).
    """
    if not mps_tensors or len(mps_tensors) <= 1:
        return 1

    bond_dims = [t.shape[2] for t in mps_tensors[:-1]]

    return max(bond_dims) if bond_dims else 1

def simulate_qaoa_circuit(num_nodes, p, gammas, betas):
    """
    Simulates a QAOA circuit for a linear graph (Max-Cut) using MPS, without
    any compression techniques.

    This function constructs the QAOA circuit layer by layer and applies the
    corresponding unitary operators (ZZ and X gates) to the MPS.
    It begins with an initial state of `|+>...|+>` (obtained by applying Hadamard
    to all |0> states).

    Args:
        num_nodes (int): The number of nodes in the linear graph.
        p (int): The number of QAOA layers.
        gammas (list or np.ndarray): A list or array of `p` gamma parameters for the cost unitary.
        betas (list or np.ndarray): A list or array of `p` beta parameters for the mixer unitary.

    Returns:
        list: The final MPS tensors after simulating the QAOA circuit.
    """
    current_mps = quantum_state_to_mps(lattice_size=num_nodes, phys_dim=2)

    H_gate = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    for i in range(num_nodes):
        _, current_mps = apply_single_qubit_gate(current_mps, i, H_gate)

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    for i in range(p):
        gamma = gammas[i]
        beta = betas[i]

        zz_gate_matrix = sp.linalg.expm(-1j * gamma * np.kron(Z, Z))

        for j in range(0, num_nodes - 1, 2):
            _, _, _, current_mps = apply_two_qubit_gate(current_mps, (j, j+1), zz_gate_matrix)

        for j in range(1, num_nodes - 1, 2):
            _, _, _, current_mps = apply_two_qubit_gate(current_mps, (j, j+1), zz_gate_matrix)

        x_gate_matrix = sp.linalg.expm(-1j * beta * X)
        for j in range(num_nodes):
            _, current_mps = apply_single_qubit_gate(current_mps, j, x_gate_matrix)

    return current_mps

def run_bond_dim_vs_nodes(p_constant, nodes_range):
    """
    Runs a simulation to observe how the maximum bond dimension of an MPS
    in a QAOA circuit grows with the number of nodes (qubits), for a fixed `p` value.

    This function simulates QAOA circuits for varying numbers of nodes,
    using fixed (though arbitrary) `gamma` and `beta` parameters. It records
    the maximum bond dimension encountered in the final MPS for each `num_nodes`.

    Args:
        p_constant (int): The fixed number of QAOA layers to use for all simulations.
        nodes_range (list or np.ndarray): A sequence of integers representing the
                                          number of nodes (qubits) to simulate.

    Returns:
        list: A list of integers, where each element is the maximum bond dimension
              of the final MPS for the corresponding `num_nodes` in `nodes_range`.
    """
    max_dims = []

    gammas = np.full(p_constant, np.pi / 4)
    betas = np.full(p_constant, np.pi / 8)

    for n in nodes_range:
        print(f"Running for N = {n}, p = {p_constant}...")
        final_mps = simulate_qaoa_circuit(n, p_constant, gammas, betas)
        max_dim = get_max_bond_dimension(final_mps)
        max_dims.append(max_dim)

    return max_dims

def run_bond_dim_vs_p(nodes_constant, p_range):
    """
    Runs a simulation to observe how the maximum bond dimension of an MPS
    in a QAOA circuit grows with the number of QAOA layers (`p`), for a fixed
    number of nodes.

    This function simulates QAOA circuits for varying numbers of layers,
    using fixed (though arbitrary) `gamma` and `beta` parameters. It records
    the maximum bond dimension encountered in the final MPS for each `p` value.

    Args:
        nodes_constant (int): The fixed number of nodes (qubits) to use for all simulations.
        p_range (list or np.ndarray): A sequence of integers representing the
                                      number of QAOA layers (`p`) to simulate.

    Returns:
        list: A list of integers, where each element is the maximum bond dimension
              of the final MPS for the corresponding `p_val` in `p_range`.
    """
    max_dims = []

    for p_val in p_range:
        print(f"Running for N = {nodes_constant}, p = {p_val}...")

        gammas = np.full(p_val, np.pi / 4)
        betas = np.full(p_val, np.pi / 8)

        final_mps = simulate_qaoa_circuit(nodes_constant, p_val, gammas, betas)
        max_dim = get_max_bond_dimension(final_mps)
        max_dims.append(max_dim)

    return max_dims

def exponential_func(x, a, b):
    """
    An exponential function used for fitting data, typically to illustrate
    exponential growth of bond dimensions.

    Args:
        x (float or np.ndarray): The independent variable.
        a (float): The scaling factor.
        b (float): The base of the exponential.

    Returns:
        float or np.ndarray: The result of a * b^x.
    """
    return a * np.power(b, x)

def simulate_qaoa_with_svd_layer_compression(num_nodes, p, gammas, betas, max_bond_dim):
    """
    Simulates a QAOA circuit for a linear graph (Max-Cut) with SVD compression
    applied at the end of *each layer*.

    This function constructs the QAOA circuit layer by layer, applying ZZ and X gates.
    After all gates for a given layer are applied, the `current_mps` is compressed
    using `svd_compression_fixed_bond` to ensure the bond dimensions do not exceed
    `max_bond_dim`. This helps control the MPS size and computational cost.

    Args:
        num_nodes (int): The number of nodes in the linear graph.
        p (int): The number of QAOA layers.
        gammas (list or np.ndarray): A list or array of `p` gamma parameters.
        betas (list or np.ndarray): A list or array of `p` beta parameters.
        max_bond_dim (int): The maximum allowed bond dimension for compression
                            at the end of each layer.

    Returns:
        list: The final compressed MPS tensors after simulating the QAOA circuit.
    """
    current_mps = quantum_state_to_mps(lattice_size=num_nodes, phys_dim=2)
    H_gate = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    for i in range(num_nodes):
        _, current_mps = apply_single_qubit_gate(current_mps, i, H_gate)

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    for i in range(p):
        gamma, beta = gammas[i], betas[i]

        zz_gate_matrix = sp.linalg.expm(-1j * gamma * np.kron(Z, Z))
        for j in range(0, num_nodes - 1, 2):
            _, _, _, current_mps = apply_two_qubit_gate(current_mps, (j, j+1), zz_gate_matrix)
        for j in range(1, num_nodes - 1, 2):
            _, _, _, current_mps = apply_two_qubit_gate(current_mps, (j, j+1), zz_gate_matrix)

        x_gate_matrix = sp.linalg.expm(-1j * beta * X)
        for j in range(num_nodes):
            _, current_mps = apply_single_qubit_gate(current_mps, j, x_gate_matrix)

        current_mps, _, _ = svd_compression_fixed_bond(current_mps, max_bond_dim)

    return current_mps

def simulate_qaoa_with_var_layer_compression(num_nodes, p, gammas, betas, max_bond_dim):
    """
    Simulates a QAOA circuit for a linear graph (Max-Cut) with variational compression
    applied at the end of *each layer*.

    This function constructs the QAOA circuit layer by layer. After all gates for
    a given layer are applied, it uses `variational_compression` to reduce the MPS
    to a target bond dimension (implicitly defined by an SVD-compressed guess MPS
    of `max_bond_dim`). This approach aims for higher accuracy at the cost of
    more computation per compression step compared to direct SVD compression.

    Args:
        num_nodes (int): The number of nodes in the linear graph.
        p (int): The number of QAOA layers.
        gammas (list or np.ndarray): A list or array of `p` gamma parameters.
        betas (list or np.ndarray): A list or array of `p` beta parameters.
        max_bond_dim (int): The maximum allowed bond dimension for the compressed MPS.

    Returns:
        list: The final compressed MPS tensors after simulating the QAOA circuit.
    """
    current_mps = quantum_state_to_mps(lattice_size=num_nodes, phys_dim=2)
    H_gate = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    for i in range(num_nodes):
        _, current_mps = apply_single_qubit_gate(current_mps, i, H_gate)

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    for i in range(p):
        gamma, beta = gammas[i], betas[i]

        zz_gate_matrix = sp.linalg.expm(-1j * gamma * np.kron(Z, Z))
        for j in range(0, num_nodes - 1, 2):
            _, _, _, current_mps = apply_two_qubit_gate(current_mps, (j, j+1), zz_gate_matrix)
        for j in range(1, num_nodes - 1, 2):
            _, _, _, current_mps = apply_two_qubit_gate(current_mps, (j, j+1), zz_gate_matrix)

        x_gate_matrix = sp.linalg.expm(-1j * beta * X)
        for j in range(num_nodes):
            _, current_mps = apply_single_qubit_gate(current_mps, j, x_gate_matrix)

        uncompressed_layer_mps = current_mps

        _, guess_for_var, _ = svd_compression_fixed_bond(uncompressed_layer_mps, max_bond_dim)

        print(f"      (Layer {i+1}/{p}: Running variational compression...)")
        current_mps, _ = variational_compression(
            target_mps=uncompressed_layer_mps,
            guess_mps=guess_for_var,
            max_sweeps=1000
        )

    return current_mps

def run_compression_experiment(param_range, const_param, max_bond_dim, mode='vs_nodes'):
    """
    Runs a comparative experiment to analyze entanglement growth and error accumulation
    in QAOA MPS simulations with different compression strategies.

    It compares:
    1. Exact simulation (no compression).
    2. SVD compression applied at the end of each QAOA layer.
    3. SVD compression applied only at the very end of the full QAOA circuit.
    4. Variational compression applied at the end of the full QAOA circuit
       (refining the SVD-compressed result).
    5. Variational compression applied at the end of each QAOA layer.

    The function reports the maximum bond dimension achieved and the fidelity error
    (infidelity) compared to the exact simulation for each method.

    Args:
        param_range (list or np.ndarray): A sequence of values for the varying parameter
                                          (either `num_nodes` or `p`, depending on `mode`).
        const_param (int): The fixed value for the other parameter (`p` if `mode='vs_nodes'`,
                           or `num_nodes` if `mode='vs_p'`).
        max_bond_dim (int): The maximum bond dimension to enforce during compression.
        mode (str, optional): Determines which parameter is varied.
                              'vs_nodes' to vary `num_nodes` (default).
                              'vs_p' to vary `p`.

    Returns:
        dict: A dictionary containing results for each compression strategy.
              Each strategy has a 'dims' list (max bond dimensions) and an
              'errors' list (fidelity errors).
    """
    results = {
        'exact': {'dims': [], 'errors': []},
        'svd_layer': {'dims': [], 'errors': []},
        'svd_end': {'dims': [], 'errors': []},
        'var_end': {'dims': [], 'errors': []},
        'var_layer': {'dims': [], 'errors': []}
    }

    for val in param_range:
        if mode == 'vs_nodes':
            num_nodes, p = val, const_param
        else:
            num_nodes, p = const_param, val
        print(f"Running for N={num_nodes}, p={p}...")
        gammas = np.full(p, np.pi / 4)
        betas = np.full(p, np.pi / 8)

        exact_mps = simulate_qaoa_circuit(num_nodes, p, gammas, betas)
        results['exact']['dims'].append(get_max_bond_dimension(exact_mps))
        results['exact']['errors'].append(0.0)

        print("   -> Simulating with Variational Compression per Layer...")
        var_layer_mps = simulate_qaoa_with_var_layer_compression(num_nodes, p, gammas, betas, max_bond_dim)
        results['var_layer']['dims'].append(get_max_bond_dimension(var_layer_mps))
        results['var_layer']['errors'].append(calculate_fidelity_error(exact_mps, var_layer_mps))

        print("   -> Simulating with SVD Compression per Layer...")
        svd_layer_mps = simulate_qaoa_with_svd_layer_compression(num_nodes, p, gammas, betas, max_bond_dim)
        results['svd_layer']['dims'].append(get_max_bond_dimension(svd_layer_mps))
        results['svd_layer']['errors'].append(calculate_fidelity_error(exact_mps, svd_layer_mps))

        print("   -> Simulating with Compression at the End...")
        svd_end_mps, svd_end_unnorm_guess, _ = svd_compression_fixed_bond(exact_mps, max_bond_dim)
        results['svd_end']['dims'].append(get_max_bond_dimension(svd_end_mps))
        results['svd_end']['errors'].append(calculate_fidelity_error(exact_mps, svd_end_mps))

        print("   -> Refining SVD-at-End with Variational Compression...")
        var_end_mps, _ = variational_compression(
            target_mps=exact_mps,
            guess_mps=svd_end_unnorm_guess,
            max_sweeps=1000
        )
        results['var_end']['dims'].append(get_max_bond_dimension(var_end_mps))
        results['var_end']['errors'].append(calculate_fidelity_error(exact_mps, var_end_mps))

    return results

def plot_zoomed_error_graph(param_range, results_data, zoom_start_index, xlabel, title):
    """
    Plots a zoomed-in graph of infidelity (error) versus a varying parameter
    for different MPS compression strategies.

    This function is designed to visualize the error accumulation behavior
    of various compression methods, focusing on a specific range of the
    independent variable (e.g., number of nodes or QAOA layers). It plots
    infidelity on a logarithmic scale.

    Args:
        param_range (list or np.ndarray): The full range of values for the
                                          independent parameter (e.g., num_nodes, p).
        results_data (dict): A dictionary containing infidelity data for different
                             compression strategies, as returned by
                             `run_compression_experiment`.
        zoom_start_index (int): The starting index in `param_range` from which
                                to begin plotting, effectively "zooming in".
        xlabel (str): The label for the x-axis (e.g., 'Number of Nodes (N)',
                      'QAOA Layers (p)').
        title (str): The title of the plot.
    """
    if zoom_start_index >= len(param_range):
        print(f"Zoom start index ({zoom_start_index}) is out of bounds. Skipping plot.")
        return

    zoomed_range = param_range[zoom_start_index:]
    zoomed_svd_layer_errors = results_data['svd_layer']['errors'][zoom_start_index:]
    zoomed_svd_end_errors = results_data['svd_end']['errors'][zoom_start_index:]
    zoomed_var_end_errors = results_data['var_end']['errors'][zoom_start_index:]
    zoomed_var_layer_errors = results_data['var_layer']['errors'][zoom_start_index:]

    plt.figure(figsize=(10, 6))
    plt.plot(zoomed_range, zoomed_svd_layer_errors, 'b--s', label='SVD per Layer')
    plt.plot(zoomed_range, zoomed_var_layer_errors, 'm-.p', label='Variational per Layer')
    plt.plot(zoomed_range, zoomed_svd_end_errors, 'g-^', label='SVD at End')
    plt.plot(zoomed_range, zoomed_var_end_errors, 'r-x', label='Variational at End')

    plt.xlabel(xlabel)
    plt.ylabel('Infidelity (1 - F)')
    plt.title(title)
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

def simulate_qaoa_inexact_maxcut(num_nodes, p, gammas, betas, max_bond_dim):
    """
    Simulates a QAOA circuit for a linear graph (Max-Cut) with *inexact* MPS
    simulation and layer-by-layer variational compression.

    In this "inexact" simulation, after each layer of quantum gates, the MPS
    is immediately compressed using variational compression. This means that
    errors accumulate throughout the circuit simulation, and the final state
    is an approximation. The function tracks the fidelity at each compression
    step to estimate the total fidelity.

    Args:
        num_nodes (int): The number of nodes in the linear graph.
        p (int): The number of QAOA layers.
        gammas (list or np.ndarray): A list or array of `p` gamma parameters.
        betas (list or np.ndarray): A list or array of `p` beta parameters.
        max_bond_dim (int): The maximum bond dimension to enforce during
                            variational compression at the end of each layer.

    Returns:
        tuple: A tuple containing:
            - current_mps (list): The final compressed MPS tensors after the
                                  inexact QAOA simulation.
            - estimated_infidelity (float): The estimated total infidelity
                                            (1 - total fidelity) accumulated
                                            throughout the simulation.
    """
    current_mps = quantum_state_to_mps(lattice_size=num_nodes, phys_dim=2)
    H_gate = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    for i in range(num_nodes):
        _, current_mps = apply_single_qubit_gate(current_mps, i, H_gate)

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    partial_fidelities = []

    for i in range(p):
        print(f"  (Inexact Layer {i+1}/{p}) Evolving and Compressing...")

        mps_to_evolve = current_mps

        gamma, beta = gammas[i], betas[i]

        zz_gate_matrix = sp.linalg.expm(-1j * gamma * np.kron(Z, Z))
        for j in range(0, num_nodes - 1, 2):
            _, _, _, mps_to_evolve = apply_two_qubit_gate(mps_to_evolve, (j, j+1), zz_gate_matrix)
        for j in range(1, num_nodes - 1, 2):
            _, _, _, mps_to_evolve = apply_two_qubit_gate(mps_to_evolve, (j, j+1), zz_gate_matrix)

        x_gate_matrix = sp.linalg.expm(-1j * beta * X)
        for j in range(num_nodes):
            _, mps_to_evolve = apply_single_qubit_gate(mps_to_evolve, j, x_gate_matrix)

        target_for_compression = mps_to_evolve

        _, guess_for_var, _ = svd_compression_fixed_bond(target_for_compression, max_bond_dim)

        current_mps, partial_fidelity = variational_compression(
            target_mps=target_for_compression,
            guess_mps=guess_for_var,
            max_sweeps=1000
        )

        partial_fidelities.append(partial_fidelity)
        print(f"    -> Partial fidelity for this step: {partial_fidelity:.8f}")

    total_fidelity = np.prod(partial_fidelities)
    estimated_infidelity = 1.0 - total_fidelity

    return current_mps, estimated_infidelity

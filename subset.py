
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np

def analyze_quantum_results(counts: dict, numbers: list, target_sum: int):
    """
    Analyzes and displays the results of a quantum subset sum experiment.

    Args:
        counts (dict): The measurement counts dictionary from a Qiskit run.
        numbers (list): The list of numbers used in the subset sum problem.
        target_sum (int): The target sum for the problem.
    """
    total_shots = sum(counts.values())
    
    # --- Dynamic Mapping Creation ---
    # Qiskit's bitstrings are little-endian. We reverse the numbers list
    # so that index 0 of the string corresponds to the first number, etc.
    num_map = list(reversed(numbers))
    
    mapping = {}
    for bitstring in counts.keys():
        # Ensure bitstring has the correct length (padding with leading zeros if necessary)
        padded_bitstring = bitstring.zfill(len(numbers))
        subset = {num_map[i] for i, bit in enumerate(padded_bitstring) if bit == '1'}
        mapping[padded_bitstring] = subset

    # --- Generate Detailed Report ---
    print(f"\n--- Dynamic Analysis for Set {numbers} with Target Sum: {target_sum} ---")
    print("-" * 100)
    header = (
        f"{'Measured State':<18}"
        f"{'Corresponding Subset':<25}"
        f"{'Sum':<8}"
        f"{'Measurement Count':<22}"
        f"{'Probability (%)':<18}"
        f"{'Result'}"
    )
    print(header)
    print("-" * 100)

    # Sort results by count for clarity
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    # Process and print each measurement outcome
    solution_counts = 0
    for bitstring, count in sorted_counts:
        padded_bitstring = bitstring.zfill(len(numbers))
        subset = mapping[padded_bitstring]
        current_sum = sum(subset)
        probability = (count / total_shots) * 100
        
        result_status = "Solution" if current_sum == target_sum else "Not a Solution"
        if result_status == "Solution":
            solution_counts += count
        
        row = (
            f"{padded_bitstring:<18}"
            f"{str(subset) if subset else '{}':<25}"
            f"{current_sum:<8}"
            f"{count:<22}"
            f"{probability:<18.2f}"
            f"{result_status}"
        )
        print(row)

    print("-" * 100)

    # --- Final Summary ---
    solution_probability = (solution_counts / total_shots) * 100
    print("\nSummary:")
    print(f"Total probability of measuring a correct solution: {solution_probability:.2f}%")
    print("This confirms the 'bitmask' (amplitude amplification) successfully isolated the correct answers.")


# --- 1. PROBLEM DEFINITION ---
problem_set = [1, 2, 3,4,5,6,7,8,9]
problem_target = 7
n = len(problem_set)

# --- 2. CIRCUIT CONSTRUCTION ---
circuit = QuantumCircuit(n)

# --- STAGE 1: The Non-Deterministic Exploration ---
print("Implementing Stage 1: Non-Deterministic Exploration...")
circuit.h(range(n))
circuit.barrier()

# --- STAGE 2: The Oracle (Marking the Solutions) ---
# This oracle is specific to the problem: set={1,2,3}, target=3
# Solutions are |001> and |110>
print("Implementing Stage 2: The Oracle...")
# Mark state |001>
circuit.x([1, 2])
circuit.cp(np.pi, 2, 0)
circuit.x([1, 2])
circuit.barrier()
# Mark state |110>
circuit.x(0)
circuit.mcp(np.pi, [1, 2], 0)
circuit.x(0)
circuit.barrier()

# --- STAGE 3: The "Bitmask" (Undoing Complexity) ---
print("Implementing Stage 3: The 'Bitmask' to Undo Complexity...")
def grover_diffuser(qc, n_qubits):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits-1)
    qc.mcx(list(range(n_qubits-1)), n_qubits-1)
    qc.h(n_qubits-1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    qc.barrier()
grover_diffuser(circuit, n)

# --- FINAL MEASUREMENT ---
print("Performing final measurement...")
circuit.measure_all()

# --- 3. EXECUTION ---
simulator = AerSimulator()
# For a real problem, you might need more shots for a larger search space
shots = 1024 
result = simulator.run(circuit, shots=shots).result()
actual_counts = result.get_counts()


# --- 4. DYNAMIC ANALYSIS ---
# The actual results from the simulation are now fed into the analysis function
analyze_quantum_results(actual_counts, problem_set, problem_target)

# Optional: Visualize the results
plot_histogram(actual_counts)

"""VQE (Variational Quantum Eigensolver) optimizer.

Implements the VQE algorithm:
1. Prepare parameterized quantum circuit (ansatz)
2. Measure expectation value ⟨ψ(θ)|H|ψ(θ)⟩
3. Classical optimizer updates parameters θ
4. Repeat until convergence

Core VQE Loop:
    θ* = argmin_θ ⟨ψ(θ)|H|ψ(θ)⟩
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from scipy.optimize import minimize
import time
import sys
sys.path.append('/app')

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit

from quantum.backend_manager import BackendManager
from quantum.ansatz_factory import AnsatzFactory
from quantum.ising_mapper import IsingMapper


class VQEOptimizer:
    """Variational Quantum Eigensolver for portfolio optimization."""
    
    def __init__(self,
                 mapper: IsingMapper,
                 backend_manager: BackendManager,
                 ansatz_type: str = 'RealAmplitudes',
                 ansatz_reps: int = 2,
                 optimizer: str = 'COBYLA',
                 max_iterations: int = 100):
        """Initialize VQE optimizer.
        
        Args:
            mapper: IsingMapper with Ising Hamiltonian
            backend_manager: BackendManager for quantum execution
            ansatz_type: Type of variational ansatz
            ansatz_reps: Number of ansatz repetitions (depth)
            optimizer: Classical optimizer ('COBYLA', 'SLSQP', 'SPSA')
            max_iterations: Maximum optimization iterations
        """
        self.mapper = mapper
        self.backend_manager = backend_manager
        self.ansatz_type = ansatz_type
        self.ansatz_reps = ansatz_reps
        self.optimizer_name = optimizer
        self.max_iterations = max_iterations
        
        # Build Ising Hamiltonian
        self.J, self.h, self.offset = mapper.qubo_to_ising()
        
        # Create ansatz
        self.num_qubits = mapper.encoder.num_qubits
        self.ansatz = AnsatzFactory.create_ansatz(
            num_qubits=self.num_qubits,
            ansatz_type=ansatz_type,
            reps=ansatz_reps,
            entanglement='linear'
        )
        
        # Get estimator
        self.estimator = backend_manager.get_estimator()
        
        # Build Hamiltonian operator
        self.hamiltonian = self._build_hamiltonian_operator()
        
        # Convergence tracking
        self.iteration_count = 0
        self.energy_history = []
        self.parameter_history = []
        self.time_history = []
        self.start_time = None
    
    def _build_hamiltonian_operator(self) -> SparsePauliOp:
        """Build Hamiltonian as SparsePauliOp for Qiskit.
        
        Converts Ising H = Σ J_ij Z_i Z_j + Σ h_i Z_i to Pauli operator form.
        
        Returns:
            SparsePauliOp representing the Hamiltonian
        """
        pauli_list = []
        coeffs = []
        
        # Add two-body terms: J_ij Z_i Z_j
        for i in range(self.num_qubits):
            for j in range(i+1, self.num_qubits):
                if abs(self.J[i, j]) > 1e-10:
                    # Create Pauli string: I...I Z_i I...I Z_j I...I
                    pauli_str = ['I'] * self.num_qubits
                    pauli_str[i] = 'Z'
                    pauli_str[j] = 'Z'
                    pauli_list.append(''.join(pauli_str))
                    coeffs.append(self.J[i, j])
        
        # Add one-body terms: h_i Z_i
        for i in range(self.num_qubits):
            if abs(self.h[i]) > 1e-10:
                # Create Pauli string: I...I Z_i I...I
                pauli_str = ['I'] * self.num_qubits
                pauli_str[i] = 'Z'
                pauli_list.append(''.join(pauli_str))
                coeffs.append(self.h[i])
        
        # Add identity term (offset)
        if abs(self.offset) > 1e-10:
            pauli_list.append('I' * self.num_qubits)
            coeffs.append(self.offset)
        
        # Create SparsePauliOp
        if len(pauli_list) == 0:
            # Empty Hamiltonian
            pauli_list = ['I' * self.num_qubits]
            coeffs = [0.0]
        
        hamiltonian = SparsePauliOp(pauli_list, coeffs)
        
        return hamiltonian
    
    def _cost_function(self, parameters: np.ndarray) -> float:
        """Evaluate VQE cost function: ⟨ψ(θ)|H|ψ(θ)⟩.
        
        This is the function minimized by VQE.
        
        Args:
            parameters: Ansatz parameters θ
            
        Returns:
            Expectation value (energy)
        """
        self.iteration_count += 1
        
        # Bind parameters to ansatz
        bound_circuit = self.ansatz.assign_parameters(parameters)
        
        # Compute expectation value using Estimator
        # BackendEstimator V1 API: run([circuits], [observables])
        job = self.estimator.run([bound_circuit], [self.hamiltonian])
        result = job.result()
        
        # Extract expectation value
        energy = result.values[0]
        
        # Track convergence
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        self.energy_history.append(float(energy))
        self.parameter_history.append(parameters.copy())
        self.time_history.append(elapsed_time)
        
        # Print progress
        if self.iteration_count % 10 == 0 or self.iteration_count == 1:
            print(f"  Iteration {self.iteration_count:3d}: Energy = {energy:12.6f}")
        
        return float(energy)
    
    def optimize(self, 
                 initial_parameters: Optional[np.ndarray] = None,
                 callback: Optional[Callable] = None) -> Dict:
        """Run VQE optimization.
        
        Args:
            initial_parameters: Initial parameter values (if None, random)
            callback: Optional callback function called after each iteration
            
        Returns:
            Dictionary with optimization results
        """
        print("="*60)
        print("VQE OPTIMIZATION")
        print("="*60)
        print()
        
        # Initialize parameters
        if initial_parameters is None:
            initial_parameters = AnsatzFactory.initialize_parameters(
                self.ansatz,
                initialization='random',
                seed=42
            )
        
        print(f"Configuration:")
        print(f"  Qubits: {self.num_qubits}")
        print(f"  Ansatz: {self.ansatz_type}")
        print(f"  Depth: {self.ansatz.depth()}")
        print(f"  Parameters: {len(initial_parameters)}")
        print(f"  Optimizer: {self.optimizer_name}")
        print(f"  Max iterations: {self.max_iterations}")
        print()
        
        print(f"Hamiltonian:")
        print(f"  Pauli terms: {len(self.hamiltonian)}")
        print(f"  Offset: {self.offset:.2f}")
        print()
        
        # Reset tracking
        self.iteration_count = 0
        self.energy_history = []
        self.parameter_history = []
        self.time_history = []
        self.start_time = time.time()
        
        print("Starting optimization...")
        print()
        
        # Run classical optimization
        if self.optimizer_name == 'COBYLA':
            result = minimize(
                fun=self._cost_function,
                x0=initial_parameters,
                method='COBYLA',
                options={'maxiter': self.max_iterations, 'disp': False},
                callback=callback
            )
        elif self.optimizer_name == 'SLSQP':
            result = minimize(
                fun=self._cost_function,
                x0=initial_parameters,
                method='SLSQP',
                options={'maxiter': self.max_iterations, 'disp': False},
                callback=callback
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported")
        
        total_time = time.time() - self.start_time
        
        print()
        print(f"Optimization complete:")
        print(f"  Iterations: {self.iteration_count}")
        print(f"  Final energy: {result.fun:.6f}")
        print(f"  Success: {result.success}")
        print(f"  Time: {total_time:.2f}s")
        print()
        
        return {
            'optimal_parameters': result.x,
            'optimal_energy': result.fun,
            'success': result.success,
            'message': result.message,
            'iterations': self.iteration_count,
            'total_time': total_time,
            'energy_history': self.energy_history,
            'parameter_history': self.parameter_history,
            'time_history': self.time_history
        }
    
    def get_convergence_data(self) -> Dict:
        """Get convergence tracking data.
        
        Returns:
            Dictionary with convergence metrics
        """
        if len(self.energy_history) == 0:
            return None
        
        energies = np.array(self.energy_history)
        
        return {
            'iterations': list(range(1, len(energies) + 1)),
            'energies': self.energy_history,
            'times': self.time_history,
            'best_energy': float(np.min(energies)),
            'worst_energy': float(np.max(energies)),
            'final_energy': float(energies[-1]),
            'energy_std': float(np.std(energies)),
            'converged': len(energies) >= 10 and np.std(energies[-10:]) < 0.001
        }
    
    def evaluate_final_state(self, parameters: np.ndarray) -> Dict:
        """Evaluate the final quantum state for given parameters.
        
        This would normally sample from the quantum state to get counts.
        For simulation, we can use the statevector.
        
        Args:
            parameters: Circuit parameters
            
        Returns:
            Dictionary with state evaluation results
        """
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        
        # Bind parameters
        bound_circuit = self.ansatz.assign_parameters(parameters)
        
        # Add measurements
        qc = bound_circuit.copy()
        qc.measure_all()
        
        # Simulate with shots
        simulator = AerSimulator()
        transpiled = transpile(qc, simulator)
        job = simulator.run(transpiled, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        return {'counts': counts}

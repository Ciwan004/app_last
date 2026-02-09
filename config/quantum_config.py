"""Quantum execution configuration.

Centralizes quantum-specific parameters: backend selection, shots,
transpiler settings, and VQE convergence criteria.
"""

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class QuantumConfig:
    """Configuration for quantum execution.
    
    Attributes:
        backend_type: Type of backend ('simulator' or 'hardware')
        backend_name: Specific backend name (e.g., 'aer_simulator', 'ibm_brisbane')
        shots: Number of measurement shots
        optimization_level: Qiskit transpiler optimization level (0-3)
        ansatz_type: Type of variational ansatz
        ansatz_reps: Number of ansatz repetitions (depth)
        optimizer_type: Classical optimizer for VQE
        max_iterations: Maximum VQE iterations
        convergence_threshold: Convergence tolerance for energy
        seed: Random seed for reproducibility
    """
    backend_type: Literal['simulator', 'hardware'] = 'simulator'
    backend_name: str = 'aer_simulator'
    shots: int = 1024
    optimization_level: int = 1
    ansatz_type: str = 'RealAmplitudes'
    ansatz_reps: int = 2
    optimizer_type: str = 'COBYLA'
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    seed: Optional[int] = 42
    
    def __post_init__(self):
        """Validate quantum configuration parameters."""
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        if self.optimization_level not in [0, 1, 2, 3]:
            raise ValueError("optimization_level must be 0, 1, 2, or 3")
        if self.ansatz_reps <= 0:
            raise ValueError("ansatz_reps must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be positive")


def get_simulator_config() -> QuantumConfig:
    """Returns default configuration for statevector simulation.
    
    Returns:
        QuantumConfig optimized for fast local simulation
    """
    return QuantumConfig(
        backend_type='simulator',
        backend_name='aer_simulator',
        shots=1024,
        optimization_level=1,
        ansatz_type='RealAmplitudes',
        ansatz_reps=2,
        optimizer_type='COBYLA',
        max_iterations=100,
        seed=42
    )

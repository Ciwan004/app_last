"""Quantum backend management for VQE execution.

Abstracts backend selection, initialization, and session management.
Supports both simulator (Aer) and hardware backends.
"""

import numpy as np
from typing import Optional, Literal
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import sys
sys.path.append('/app')

from config.quantum_config import QuantumConfig


class BackendManager:
    """Manages quantum backend selection and primitive initialization."""
    
    def __init__(self, config: QuantumConfig):
        """Initialize backend manager.
        
        Args:
            config: QuantumConfig with backend settings
        """
        self.config = config
        self.backend = None
        self.estimator = None
        
        # Initialize backend
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize quantum backend based on configuration."""
        if self.config.backend_type == 'simulator':
            self._initialize_simulator()
        else:
            raise NotImplementedError(
                f"Backend type '{self.config.backend_type}' not yet implemented. "
                "Currently only 'simulator' is supported."
            )
    
    def _initialize_simulator(self):
        """Initialize Aer simulator backend."""
        # Create Aer simulator with method='statevector' for unlimited qubits
        # For large problems, use method='automatic' which will choose best method
        self.backend = AerSimulator(method='automatic')
        
        # Set seed for reproducibility
        if self.config.seed is not None:
            self.backend.set_options(seed_simulator=self.config.seed)
        
        # Configure shots
        self.backend.set_options(shots=self.config.shots)
        
        print(f"✓ Backend initialized: {self.backend.name}")
        print(f"  Shots: {self.config.shots}")
        print(f"  Seed: {self.config.seed}")
    
    def get_estimator(self) -> BackendEstimator:
        """Get Qiskit Estimator primitive for expectation value computation.
        
        The Estimator computes ⟨ψ(θ)|H|ψ(θ)⟩ for parameterized circuits.
        This is the core primitive for VQE.
        
        Returns:
            BackendEstimator configured with current backend
        """
        if self.estimator is None:
            # Create estimator with backend
            self.estimator = BackendEstimator(backend=self.backend)
        
        return self.estimator
    
    def get_transpiler(self, num_qubits: int):
        """Get transpiler pass manager for circuit optimization.
        
        Args:
            num_qubits: Number of qubits in the circuit
            
        Returns:
            Preset pass manager for transpilation
        """
        pm = generate_preset_pass_manager(
            optimization_level=self.config.optimization_level,
            backend=self.backend
        )
        return pm
    
    def get_backend_info(self) -> dict:
        """Get backend information.
        
        Returns:
            Dictionary with backend configuration
        """
        return {
            'name': self.backend.name if self.backend else None,
            'type': self.config.backend_type,
            'shots': self.config.shots,
            'seed': self.config.seed,
            'optimization_level': self.config.optimization_level
        }

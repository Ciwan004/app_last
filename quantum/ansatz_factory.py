"""Ansatz factory for variational quantum circuits.

Provides various parameterized ansatz circuits for VQE:
- RealAmplitudes: Rotation + entanglement layers
- EfficientSU2: Hardware-efficient ansatz
- Custom problem-specific ansatze (future)
"""

import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal
from typing import Optional, Literal


class AnsatzFactory:
    """Factory for creating variational ansatz circuits."""
    
    @staticmethod
    def create_ansatz(
        num_qubits: int,
        ansatz_type: str = 'RealAmplitudes',
        reps: int = 2,
        entanglement: str = 'linear',
        insert_barriers: bool = True
    ) -> QuantumCircuit:
        """Create parameterized ansatz circuit.
        
        Args:
            num_qubits: Number of qubits
            ansatz_type: Type of ansatz ('RealAmplitudes', 'EfficientSU2', 'TwoLocal')
            reps: Number of repetitions (depth)
            entanglement: Entanglement pattern ('linear', 'full', 'circular')
            insert_barriers: Whether to insert barriers for visualization
            
        Returns:
            Parameterized QuantumCircuit
        """
        if ansatz_type == 'RealAmplitudes':
            return AnsatzFactory.create_real_amplitudes(
                num_qubits, reps, entanglement, insert_barriers
            )
        elif ansatz_type == 'EfficientSU2':
            return AnsatzFactory.create_efficient_su2(
                num_qubits, reps, entanglement, insert_barriers
            )
        elif ansatz_type == 'TwoLocal':
            return AnsatzFactory.create_two_local(
                num_qubits, reps, entanglement, insert_barriers
            )
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")
    
    @staticmethod
    def create_real_amplitudes(
        num_qubits: int,
        reps: int = 2,
        entanglement: str = 'linear',
        insert_barriers: bool = True
    ) -> RealAmplitudes:
        """Create RealAmplitudes ansatz.
        
        Structure:
        - Rotation layer: RY gates on all qubits
        - Entanglement layer: CX gates following entanglement pattern
        - Repeat for `reps` times
        - Final rotation layer
        
        This ansatz explores the space of real-valued amplitudes
        (no complex phases), suitable for many optimization problems.
        
        Args:
            num_qubits: Number of qubits
            reps: Number of repetitions
            entanglement: Entanglement pattern
            insert_barriers: Insert barriers between layers
            
        Returns:
            RealAmplitudes circuit
        """
        ansatz = RealAmplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            insert_barriers=insert_barriers
        )
        return ansatz
    
    @staticmethod
    def create_efficient_su2(
        num_qubits: int,
        reps: int = 2,
        entanglement: str = 'linear',
        insert_barriers: bool = True
    ) -> EfficientSU2:
        """Create EfficientSU2 ansatz.
        
        Structure:
        - Rotation layer: RY and RZ gates on all qubits
        - Entanglement layer: CX gates
        - More expressive than RealAmplitudes (adds RZ gates)
        
        Hardware-efficient: uses native gates of most quantum processors.
        
        Args:
            num_qubits: Number of qubits
            reps: Number of repetitions
            entanglement: Entanglement pattern
            insert_barriers: Insert barriers between layers
            
        Returns:
            EfficientSU2 circuit
        """
        ansatz = EfficientSU2(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            insert_barriers=insert_barriers
        )
        return ansatz
    
    @staticmethod
    def create_two_local(
        num_qubits: int,
        reps: int = 2,
        entanglement: str = 'linear',
        insert_barriers: bool = True
    ) -> TwoLocal:
        """Create TwoLocal ansatz with custom gate set.
        
        TwoLocal is a flexible ansatz constructor.
        Here we use RY rotations and CX entanglement.
        
        Args:
            num_qubits: Number of qubits
            reps: Number of repetitions
            entanglement: Entanglement pattern
            insert_barriers: Insert barriers between layers
            
        Returns:
            TwoLocal circuit
        """
        ansatz = TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks='ry',
            entanglement_blocks='cx',
            entanglement=entanglement,
            reps=reps,
            insert_barriers=insert_barriers
        )
        return ansatz
    
    @staticmethod
    def get_ansatz_info(ansatz: QuantumCircuit) -> dict:
        """Get information about an ansatz.
        
        Args:
            ansatz: Ansatz circuit
            
        Returns:
            Dictionary with ansatz properties
        """
        return {
            'num_qubits': ansatz.num_qubits,
            'num_parameters': ansatz.num_parameters,
            'depth': ansatz.depth(),
            'gate_count': sum(ansatz.count_ops().values()),
            'gate_types': list(ansatz.count_ops().keys())
        }
    
    @staticmethod
    def initialize_parameters(
        ansatz: QuantumCircuit,
        initialization: str = 'random',
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate initial parameter values for the ansatz.
        
        Args:
            ansatz: Ansatz circuit
            initialization: Initialization strategy ('random', 'zeros', 'ones')
            seed: Random seed for reproducibility
            
        Returns:
            Initial parameter values
        """
        num_params = ansatz.num_parameters
        
        if initialization == 'random':
            if seed is not None:
                np.random.seed(seed)
            # Random parameters in [0, 2π]
            return np.random.uniform(0, 2 * np.pi, num_params)
        elif initialization == 'zeros':
            return np.zeros(num_params)
        elif initialization == 'ones':
            return np.ones(num_params)
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

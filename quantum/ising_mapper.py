"""Ising/QUBO mapping for portfolio optimization.

Maps the classical Markowitz optimization problem to an Ising Hamiltonian
that can be solved using VQE or other quantum algorithms.

Encoding Strategy (One-Hot):
- Discretize each asset's weight into bins: {0, Δ, 2Δ, ..., 1}
- For each asset, use one-hot encoding: exactly one bin is active
- Total qubits: num_assets × num_bins
- Budget constraint enforced via penalty term

Mathematical Mapping:
    Classical: minimize (1/2) w^T Σ w - λ μ^T w
               subject to Σw_i = 1
    
    Quantum:   H_ising = H_objective + P * H_constraint
    
    Where:
        H_objective encodes the Markowitz objective
        H_constraint penalizes budget violations
        P is the penalty coefficient
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import sys
sys.path.append('/app')

from finance.markowitz import MarkowitzModel


class OneHotEncoder:
    """One-hot encoding for discrete weight levels.
    
    Maps continuous weights to binary variables using one-hot encoding.
    For each asset, exactly one weight bin is selected.
    """
    
    def __init__(self, num_assets: int, num_bins: int = 5, max_weight: float = 1.0):
        """Initialize one-hot encoder.
        
        Args:
            num_assets: Number of assets in portfolio
            num_bins: Number of discrete weight levels per asset
            max_weight: Maximum weight value (typically 1.0)
        """
        self.num_assets = num_assets
        self.num_bins = num_bins
        self.max_weight = max_weight
        
        # Define weight bins: [0, Δ, 2Δ, ..., max_weight]
        self.weight_bins = np.linspace(0, max_weight, num_bins)
        self.delta = self.weight_bins[1] - self.weight_bins[0] if num_bins > 1 else max_weight
        
        # Total number of binary variables
        self.num_qubits = num_assets * num_bins
    
    def encode_weights(self, weights: np.ndarray) -> np.ndarray:
        """Encode continuous weights as binary one-hot vector.
        
        Args:
            weights: Continuous portfolio weights (length num_assets)
            
        Returns:
            Binary vector (length num_qubits) with one-hot encoding
        """
        binary = np.zeros(self.num_qubits, dtype=int)
        
        for i, w in enumerate(weights):
            # Find closest bin
            bin_idx = np.argmin(np.abs(self.weight_bins - w))
            # Set corresponding binary variable to 1
            binary[i * self.num_bins + bin_idx] = 1
        
        return binary
    
    def decode_binary(self, binary: np.ndarray) -> np.ndarray:
        """Decode binary one-hot vector to continuous weights.
        
        Args:
            binary: Binary vector (length num_qubits)
            
        Returns:
            Continuous portfolio weights (length num_assets)
        """
        weights = np.zeros(self.num_assets)
        
        for i in range(self.num_assets):
            # Extract one-hot block for asset i
            block = binary[i * self.num_bins : (i+1) * self.num_bins]
            
            # If one-hot constraint satisfied, decode
            if np.sum(block) == 1:
                bin_idx = np.argmax(block)
                weights[i] = self.weight_bins[bin_idx]
            elif np.sum(block) > 1:
                # Multiple bins active: average (should be penalized)
                active_indices = np.where(block > 0)[0]
                weights[i] = np.mean(self.weight_bins[active_indices])
            # else: all zeros, weight stays 0
        
        return weights
    
    def get_qubit_map(self) -> Dict[int, Tuple[int, float]]:
        """Get mapping from qubit index to (asset_index, weight_value).
        
        Returns:
            Dictionary: qubit_idx -> (asset_idx, weight_value)
        """
        qubit_map = {}
        for asset_idx in range(self.num_assets):
            for bin_idx in range(self.num_bins):
                qubit_idx = asset_idx * self.num_bins + bin_idx
                weight_value = self.weight_bins[bin_idx]
                qubit_map[qubit_idx] = (asset_idx, weight_value)
        return qubit_map


class IsingMapper:
    """Maps Markowitz portfolio optimization to Ising/QUBO formulation."""
    
    def __init__(self, 
                 model: MarkowitzModel,
                 num_bins: int = 5,
                 penalty_coefficient: float = 10.0):
        """Initialize Ising mapper.
        
        Args:
            model: MarkowitzModel defining the problem
            num_bins: Number of discrete weight levels per asset
            penalty_coefficient: Penalty for constraint violations (P)
        """
        self.model = model
        self.num_bins = num_bins
        self.penalty_coefficient = penalty_coefficient
        
        # Create encoder
        self.encoder = OneHotEncoder(
            num_assets=model.num_assets,
            num_bins=num_bins,
            max_weight=model.constraints.bounds[1]
        )
        
        # Store QUBO matrix (will be computed)
        self.Q: Optional[np.ndarray] = None
        self.offset: float = 0.0
    
    def build_qubo(self) -> Tuple[np.ndarray, float]:
        """Build QUBO matrix for the portfolio optimization problem.
        
        Constructs Q such that the QUBO objective is:
            E(x) = x^T Q x + offset
        
        Where x is the binary one-hot encoded vector.
        
        Key insight: With one-hot encoding, each asset i has weight:
            w_i = Σ_j∈bins(i) weight_j * x_ij
        where exactly one x_ij = 1 and others are 0.
        
        Returns:
            Tuple of (Q_matrix, offset)
        """
        n = self.encoder.num_qubits
        Q = np.zeros((n, n))
        
        # Get weight values for each qubit
        qubit_map = self.encoder.get_qubit_map()
        
        covariance = self.model.objective.covariance_matrix
        expected_returns = self.model.objective.expected_returns
        risk_aversion = self.model.objective.risk_aversion
        budget = self.model.constraints.budget
        P = self.penalty_coefficient
        
        # Part 1: Markowitz objective
        # Objective: (1/2) w^T Σ w - λ μ^T w
        # where w_i = Σ_j∈bins(i) weight_j * x_ij
        
        for i in range(n):
            asset_i, weight_i = qubit_map[i]
            
            for j in range(i, n):
                asset_j, weight_j = qubit_map[j]
                
                # Risk term: (1/2) Σ_ik * w_i * w_j
                cov_term = 0.5 * covariance[asset_i, asset_j] * weight_i * weight_j
                
                if i == j:
                    # Diagonal: single qubit term
                    Q[i, i] += cov_term
                    
                    # Return term: - λ μ_i * w_i (only on diagonal for linear terms)
                    Q[i, i] -= risk_aversion * expected_returns[asset_i] * weight_i
                else:
                    # Off-diagonal: interaction between qubits
                    # Factor of 2 because QUBO counts x_i * x_j once
                    Q[i, j] += 2.0 * cov_term
        
        # Part 2: Budget constraint penalty P * (Σw_i - budget)^2
        # With one-hot: Σw_i = ΣΣ weight_j * x_ij
        # (Σw)^2 = ΣΣ w_i * w_j * x_i * x_j
        
        for i in range(n):
            _, weight_i = qubit_map[i]
            
            for j in range(i, n):
                _, weight_j = qubit_map[j]
                
                if i == j:
                    # x_i^2 = x_i, coefficient: P * weight_i^2
                    Q[i, i] += P * weight_i * weight_i
                else:
                    # x_i * x_j, coefficient: 2P * weight_i * weight_j
                    Q[i, j] += 2.0 * P * weight_i * weight_j
        
        # Linear term from budget: -2P * budget * Σw_i
        for i in range(n):
            _, weight_i = qubit_map[i]
            Q[i, i] -= 2.0 * P * budget * weight_i
        
        # Constant offset from budget: P * budget^2
        offset = P * budget * budget
        
        # Part 3: One-hot constraint penalty
        # For each asset a: P_onehot * (Σ_j∈bins(a) x_aj - 1)^2
        P_onehot = self.penalty_coefficient
        
        for asset_idx in range(self.model.num_assets):
            start_qubit = asset_idx * self.num_bins
            end_qubit = start_qubit + self.num_bins
            
            # (Σx_j - 1)^2 = Σ_j x_j^2 + 2Σ_j<k x_j*x_k - 2Σ_j x_j + 1
            for i in range(start_qubit, end_qubit):
                # Diagonal: x_i^2 = x_i
                Q[i, i] += P_onehot
                # Linear: -2 x_i
                Q[i, i] -= 2.0 * P_onehot
                
                for j in range(i+1, end_qubit):
                    # Off-diagonal: 2 x_i * x_j
                    Q[i, j] += 2.0 * P_onehot
            
            # Constant: +1
            offset += P_onehot
        
        self.Q = Q
        self.offset = offset
        
        return Q, offset
    
    def qubo_to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convert QUBO to Ising formulation.
        
        QUBO: E(x) = Σ_i Q_ii x_i + Σ_i<j Q_ij x_i x_j + offset
              where x ∈ {0,1}
        
        Ising: H(σ) = Σ_i<j J_ij σ_i σ_j + Σ_i h_i σ_i + offset_ising
               where σ ∈ {-1,+1}
        
        Conversion: x_i = (σ_i + 1) / 2
        
        Substituting into QUBO:
            E(σ) = Σ_i Q_ii (σ_i + 1)/2 + Σ_i<j Q_ij (σ_i + 1)/2 (σ_j + 1)/2 + offset
        
        Expanding:
            = Σ_i Q_ii/2 (σ_i + 1) + Σ_i<j Q_ij/4 (σ_i σ_j + σ_i + σ_j + 1) + offset
            = Σ_i<j (Q_ij/4) σ_i σ_j 
              + Σ_i (Q_ii/2 + Σ_j≠i Q_ij/4) σ_i
              + Σ_i Q_ii/2 + Σ_i<j Q_ij/4 + offset
        
        Therefore:
            J_ij = Q_ij / 4 for i < j
            h_i = Q_ii/2 + (1/4)Σ_j≠i Q_ij  (accounting for upper triangular Q)
            offset_ising = Σ_i Q_ii/2 + Σ_i<j Q_ij/4 + offset
        
        Returns:
            Tuple of (J_matrix, h_vector, offset_ising)
        """
        if self.Q is None:
            self.build_qubo()
        
        n = self.Q.shape[0]
        
        # Build symmetric Q_full from upper triangular Q
        Q_full = self.Q + self.Q.T - np.diag(np.diag(self.Q))
        
        # J matrix (interaction terms)
        J = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                J[i, j] = Q_full[i, j] / 4.0
                J[j, i] = J[i, j]  # Symmetric
        
        # h vector (linear terms)
        h = np.zeros(n)
        for i in range(n):
            # h_i = Q_ii/2 + (1/4) Σ_j≠i Q_ij
            h[i] = self.Q[i, i] / 2.0
            for j in range(n):
                if j != i:
                    h[i] += Q_full[i, j] / 4.0
        
        # Offset
        offset_ising = self.offset
        # Add diagonal contribution
        offset_ising += np.sum(self.Q.diagonal()) / 2.0
        # Add off-diagonal contribution
        offset_ising += np.sum(np.triu(Q_full, 1)) / 4.0
        
        return J, h, offset_ising
    
    def evaluate_binary(self, binary: np.ndarray) -> Dict[str, float]:
        """Evaluate QUBO objective for a binary solution.
        
        Args:
            binary: Binary solution vector
            
        Returns:
            Dictionary with energy, objective, constraints
        """
        if self.Q is None:
            self.build_qubo()
        
        # QUBO energy
        qubo_energy = binary @ self.Q @ binary + self.offset
        
        # Decode to weights
        weights = self.encoder.decode_binary(binary)
        
        # Evaluate Markowitz objective
        markowitz_obj = self.model.objective.compute_objective(weights)
        
        # Check constraints
        budget_violation = abs(np.sum(weights) - self.model.constraints.budget)
        
        # Check one-hot violations
        onehot_violations = 0
        for asset_idx in range(self.model.num_assets):
            block = binary[asset_idx * self.num_bins : (asset_idx + 1) * self.num_bins]
            if np.sum(block) != 1:
                onehot_violations += abs(np.sum(block) - 1)
        
        return {
            'qubo_energy': qubo_energy,
            'markowitz_objective': markowitz_obj,
            'weights': weights,
            'budget_violation': budget_violation,
            'onehot_violations': onehot_violations,
            'is_feasible': budget_violation < 1e-6 and onehot_violations == 0
        }
    
    def validate_mapping(self, classical_weights: np.ndarray, 
                        tolerance: float = 1e-3) -> Dict[str, any]:
        """Validate that classical optimum maps to low-energy Ising state.
        
        Args:
            classical_weights: Optimal weights from classical solver
            tolerance: Tolerance for weight discretization
            
        Returns:
            Validation results dictionary
        """
        # Encode classical solution
        binary_classical = self.encoder.encode_weights(classical_weights)
        
        # Evaluate
        result = self.evaluate_binary(binary_classical)
        
        # Decode back
        decoded_weights = result['weights']
        weight_error = np.linalg.norm(decoded_weights - classical_weights)
        
        return {
            'classical_weights': classical_weights,
            'encoded_binary': binary_classical,
            'decoded_weights': decoded_weights,
            'weight_error': weight_error,
            'qubo_energy': result['qubo_energy'],
            'is_feasible': result['is_feasible'],
            'budget_violation': result['budget_violation'],
            'passes_validation': weight_error < tolerance and result['is_feasible']
        }

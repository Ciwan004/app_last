"""Result decoder for VQE solutions.

Decodes quantum measurement results (bitstrings) back to portfolio weights
and evaluates solution quality.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('/app')

from quantum.ising_mapper import IsingMapper


class ResultDecoder:
    """Decodes VQE results to portfolio weights."""
    
    def __init__(self, mapper: IsingMapper):
        """Initialize decoder.
        
        Args:
            mapper: IsingMapper with encoder and model information
        """
        self.mapper = mapper
        self.encoder = mapper.encoder
        self.model = mapper.model
    
    def decode_statevector(self, statevector: np.ndarray, 
                          num_samples: int = 1000) -> Dict:
        """Decode statevector to portfolio weights.
        
        Samples from the statevector probability distribution and
        decodes the most probable feasible solution.
        
        Args:
            statevector: Quantum statevector (amplitudes)
            num_samples: Number of samples to draw
            
        Returns:
            Dictionary with decoded solution
        """
        # Compute probabilities
        probabilities = np.abs(statevector) ** 2
        
        # Sample bitstrings
        num_qubits = int(np.log2(len(statevector)))
        samples = np.random.choice(
            len(statevector),
            size=num_samples,
            p=probabilities
        )
        
        # Count occurrences
        counts = {}
        for sample in samples:
            bitstring = format(sample, f'0{num_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return self.decode_counts(counts)
    
    def decode_counts(self, counts: Dict[str, int]) -> Dict:
        """Decode measurement counts to portfolio solution.
        
        Finds the most probable feasible solution from measurement counts.
        
        Args:
            counts: Dictionary {bitstring: count}
            
        Returns:
            Dictionary with:
                - best_weights: Decoded portfolio weights
                - best_bitstring: Corresponding bitstring
                - best_energy: QUBO energy
                - best_probability: Measurement probability
                - feasible_fraction: Fraction of feasible measurements
                - all_solutions: List of top solutions
        """
        total_counts = sum(counts.values())
        
        # Evaluate all measured states
        solutions = []
        feasible_count = 0
        
        for bitstring, count in counts.items():
            # Convert bitstring to binary array
            binary = np.array([int(b) for b in bitstring])
            
            # Evaluate using mapper
            result = self.mapper.evaluate_binary(binary)
            
            probability = count / total_counts
            
            if result['is_feasible']:
                feasible_count += count
            
            solutions.append({
                'bitstring': bitstring,
                'binary': binary,
                'weights': result['weights'],
                'energy': result['qubo_energy'],
                'markowitz_obj': result['markowitz_objective'],
                'is_feasible': result['is_feasible'],
                'budget_violation': result['budget_violation'],
                'probability': probability,
                'count': count
            })
        
        # Sort by energy (lowest first)
        solutions.sort(key=lambda x: x['energy'])
        
        # Find best feasible solution
        best_feasible = None
        for sol in solutions:
            if sol['is_feasible']:
                best_feasible = sol
                break
        
        # If no feasible solution, take lowest energy (even if infeasible)
        best_overall = solutions[0] if solutions else None
        
        return {
            'best_solution': best_feasible if best_feasible else best_overall,
            'best_weights': best_feasible['weights'] if best_feasible else (best_overall['weights'] if best_overall else None),
            'best_energy': best_feasible['energy'] if best_feasible else (best_overall['energy'] if best_overall else None),
            'best_bitstring': best_feasible['bitstring'] if best_feasible else (best_overall['bitstring'] if best_overall else None),
            'is_feasible': best_feasible is not None,
            'feasible_fraction': feasible_count / total_counts if total_counts > 0 else 0,
            'num_unique_states': len(solutions),
            'all_solutions': solutions[:10]  # Top 10
        }
    
    def compare_with_classical(self, 
                              vqe_weights: np.ndarray,
                              classical_weights: np.ndarray) -> Dict:
        """Compare VQE solution with classical optimum.
        
        Args:
            vqe_weights: Weights from VQE
            classical_weights: Weights from classical solver
            
        Returns:
            Comparison metrics
        """
        # Compute objectives
        vqe_obj = self.model.objective.compute_objective(vqe_weights)
        classical_obj = self.model.objective.compute_objective(classical_weights)
        
        # Compute risk and return
        vqe_risk = self.model.objective.compute_risk(vqe_weights)
        vqe_return = self.model.objective.compute_return(vqe_weights)
        
        classical_risk = self.model.objective.compute_risk(classical_weights)
        classical_return = self.model.objective.compute_return(classical_weights)
        
        # Weight difference
        weight_diff = np.linalg.norm(vqe_weights - classical_weights)
        
        # Approximation ratio
        # For minimization: ratio = vqe_obj / classical_obj
        # (ratio close to 1 is good)
        approx_ratio = vqe_obj / classical_obj if classical_obj != 0 else np.inf
        
        return {
            'vqe_objective': vqe_obj,
            'classical_objective': classical_obj,
            'objective_gap': vqe_obj - classical_obj,
            'approximation_ratio': approx_ratio,
            'vqe_risk': vqe_risk,
            'vqe_return': vqe_return,
            'classical_risk': classical_risk,
            'classical_return': classical_return,
            'weight_l2_distance': weight_diff,
            'weight_max_diff': np.max(np.abs(vqe_weights - classical_weights))
        }

"""Adaptive penalty strategy for constraint enforcement in VQE.

Implements adaptive penalty coefficient adjustment to ensure feasible solutions
while maintaining optimization quality.
"""

import numpy as np
from typing import Dict, Optional, Callable
import sys
sys.path.append('/app')

from quantum.ising_mapper import IsingMapper
from quantum.vqe_optimizer import VQEOptimizer
from quantum.result_decoder import ResultDecoder


class AdaptivePenaltyVQE:
    """VQE with adaptive penalty coefficient adjustment.
    
    Strategy:
    1. Start with initial penalty P₀
    2. Run VQE to convergence
    3. If solution infeasible, increase penalty: P ← α * P
    4. Repeat until feasible or max attempts reached
    """
    
    def __init__(self,
                 mapper: IsingMapper,
                 vqe_optimizer: VQEOptimizer,
                 initial_penalty: float = 100.0,
                 penalty_multiplier: float = 2.0,
                 max_attempts: int = 5,
                 feasibility_tolerance: float = 0.01):
        """Initialize adaptive penalty VQE.
        
        Args:
            mapper: IsingMapper instance
            vqe_optimizer: VQEOptimizer instance
            initial_penalty: Starting penalty coefficient
            penalty_multiplier: Factor to increase penalty (α)
            max_attempts: Maximum penalty adjustment attempts
            feasibility_tolerance: Tolerance for constraint violations
        """
        self.mapper = mapper
        self.vqe_optimizer = vqe_optimizer
        self.initial_penalty = initial_penalty
        self.penalty_multiplier = penalty_multiplier
        self.max_attempts = max_attempts
        self.feasibility_tolerance = feasibility_tolerance
        
        # Track history
        self.penalty_history = []
        self.feasibility_history = []
        self.energy_history = []
        self.solution_history = []
    
    def optimize_with_adaptive_penalty(self, 
                                       initial_parameters: Optional[np.ndarray] = None) -> Dict:
        """Run VQE with adaptive penalty adjustment.
        
        Args:
            initial_parameters: Initial circuit parameters
            
        Returns:
            Dictionary with best feasible solution found
        """
        print("="*60)
        print("ADAPTIVE PENALTY VQE")
        print("="*60)
        print()
        
        current_penalty = self.initial_penalty
        best_feasible_solution = None
        best_feasible_energy = float('inf')
        
        for attempt in range(1, self.max_attempts + 1):
            print(f"--- Attempt {attempt}/{self.max_attempts} ---")
            print(f"Penalty coefficient: {current_penalty:.1f}")
            print()
            
            # Update mapper penalty
            self.mapper.penalty_coefficient = current_penalty
            
            # Rebuild QUBO with new penalty
            self.mapper.build_qubo()
            
            # Rebuild Hamiltonian in VQE optimizer
            self.vqe_optimizer.mapper = self.mapper
            self.vqe_optimizer.J, self.vqe_optimizer.h, self.vqe_optimizer.offset = \
                self.mapper.qubo_to_ising()
            self.vqe_optimizer.hamiltonian = self.vqe_optimizer._build_hamiltonian_operator()
            
            # Run VQE
            vqe_result = self.vqe_optimizer.optimize(initial_parameters=initial_parameters)
            
            # Evaluate final state
            state_result = self.vqe_optimizer.evaluate_final_state(vqe_result['optimal_parameters'])
            counts = state_result['counts']
            
            # Decode results
            decoder = ResultDecoder(self.mapper)
            decoded = decoder.decode_counts(counts)
            
            # Record history
            self.penalty_history.append(current_penalty)
            self.feasibility_history.append(decoded['is_feasible'])
            self.energy_history.append(decoded['best_energy'])
            self.solution_history.append(decoded)
            
            print(f"Results:")
            print(f"  Weights: {decoded['best_weights']}")
            print(f"  Feasible: {decoded['is_feasible']}")
            print(f"  Feasible fraction: {decoded['feasible_fraction']:.2%}")
            
            if decoded['best_weights'] is not None:
                budget_sum = np.sum(decoded['best_weights'])
                budget_violation = abs(budget_sum - self.mapper.model.constraints.budget)
                print(f"  Weight sum: {budget_sum:.4f} (target: {self.mapper.model.constraints.budget})")
                print(f"  Budget violation: {budget_violation:.4e}")
            print()
            
            # Check feasibility
            if decoded['is_feasible']:
                print(f"✓ Feasible solution found at attempt {attempt}!")
                print()
                
                # Check if better than previous feasible
                if decoded['best_energy'] < best_feasible_energy:
                    best_feasible_solution = decoded
                    best_feasible_energy = decoded['best_energy']
                
                # Found feasible, can stop
                break
            else:
                print(f"✗ Solution infeasible, increasing penalty...")
                print()
                
                # Increase penalty for next attempt
                current_penalty *= self.penalty_multiplier
                
                # Use current parameters as warm start for next attempt
                initial_parameters = vqe_result['optimal_parameters']
        
        if best_feasible_solution is None:
            print("⚠ No feasible solution found after all attempts")
            print("   Returning best overall solution (infeasible)")
            print()
            # Return solution with lowest constraint violation
            best_idx = np.argmin([s['best_solution']['budget_violation'] 
                                  for s in self.solution_history 
                                  if s['best_solution'] is not None])
            best_feasible_solution = self.solution_history[best_idx]
        else:
            print(f"✓ Best feasible solution has energy: {best_feasible_energy:.6f}")
            print()
        
        return {
            'best_solution': best_feasible_solution,
            'penalty_history': self.penalty_history,
            'feasibility_history': self.feasibility_history,
            'energy_history': self.energy_history,
            'solution_history': self.solution_history,
            'attempts': len(self.penalty_history)
        }
    
    def get_summary(self) -> Dict:
        """Get summary of adaptive penalty optimization.
        
        Returns:
            Summary statistics
        """
        return {
            'total_attempts': len(self.penalty_history),
            'penalties_tried': self.penalty_history,
            'feasible_count': sum(self.feasibility_history),
            'first_feasible_attempt': (
                self.feasibility_history.index(True) + 1 
                if True in self.feasibility_history 
                else None
            ),
            'optimal_penalty': (
                self.penalty_history[self.feasibility_history.index(True)]
                if True in self.feasibility_history
                else None
            )
        }

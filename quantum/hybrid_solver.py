"""Hybrid classical-quantum post-processing for VQE solutions.

Projects infeasible VQE solutions onto the feasible set using classical optimization.
This ensures all solutions satisfy portfolio constraints while leveraging quantum exploration.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional
import sys
sys.path.append('/app')

from finance.markowitz import MarkowitzModel


class FeasibilityProjection:
    """Projects infeasible solutions to feasible constraint set.
    
    Solves: minimize ||w - w_vqe||²
            subject to: sum(w) = budget
                       w_min <= w_i <= w_max
    
    This is a convex QP with closed-form solution or fast numerical solve.
    """
    
    def __init__(self, model: MarkowitzModel):
        """Initialize projection.
        
        Args:
            model: MarkowitzModel with constraint information
        """
        self.model = model
        self.budget = model.constraints.budget
        self.bounds = model.constraints.bounds
        self.num_assets = model.num_assets
    
    def project_to_feasible(self, weights: np.ndarray, method: str = 'slsqp') -> Dict:
        """Project weights to feasible set.
        
        Args:
            weights: Possibly infeasible weights from VQE
            method: Optimization method ('slsqp', 'trust-constr', 'closed-form')
            
        Returns:
            Dictionary with projected weights and projection info
        """
        # Check if already feasible
        if self._is_feasible(weights):
            return {
                'weights': weights,
                'projection_distance': 0.0,
                'iterations': 0,
                'was_feasible': True,
                'method': 'none'
            }
        
        # Project using specified method
        if method == 'closed-form':
            projected, info = self._project_closed_form(weights)
        else:
            projected, info = self._project_optimization(weights, method)
        
        return {
            'weights': projected,
            'projection_distance': info['distance'],
            'iterations': info.get('iterations', 0),
            'was_feasible': False,
            'method': method,
            'original_weights': weights
        }
    
    def _is_feasible(self, weights: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if weights satisfy all constraints.
        
        Args:
            weights: Portfolio weights
            tolerance: Numerical tolerance
            
        Returns:
            True if feasible
        """
        # Check budget
        if abs(np.sum(weights) - self.budget) > tolerance:
            return False
        
        # Check bounds
        if np.any(weights < self.bounds[0] - tolerance):
            return False
        if np.any(weights > self.bounds[1] + tolerance):
            return False
        
        return True
    
    def _project_optimization(self, weights: np.ndarray, method: str) -> tuple:
        """Project using numerical optimization.
        
        Args:
            weights: Initial weights
            method: Scipy optimization method
            
        Returns:
            Tuple of (projected_weights, info_dict)
        """
        # Objective: minimize ||w - w_vqe||²
        def objective(w):
            return np.sum((w - weights) ** 2)
        
        # Gradient
        def gradient(w):
            return 2 * (w - weights)
        
        # Budget constraint
        budget_constraint = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - self.budget,
            'jac': lambda w: np.ones(self.num_assets)
        }
        
        # Bounds
        bounds = [self.bounds] * self.num_assets
        
        # Initial guess (project to budget first for better starting point)
        w_init = self._simple_budget_projection(weights)
        
        # Solve
        result = minimize(
            fun=objective,
            x0=w_init,
            method=method.upper(),
            jac=gradient,
            bounds=bounds,
            constraints=[budget_constraint],
            options={'disp': False}
        )
        
        projected = result.x
        distance = np.linalg.norm(projected - weights)
        
        return projected, {
            'distance': distance,
            'iterations': result.nit if hasattr(result, 'nit') else 0,
            'success': result.success
        }
    
    def _project_closed_form(self, weights: np.ndarray) -> tuple:
        """Project using closed-form solution for simple cases.
        
        For box constraints + budget, we can use iterative projection:
        1. Project to bounds
        2. Project to budget (preserve bounds)
        3. Repeat until converged
        
        Args:
            weights: Initial weights
            
        Returns:
            Tuple of (projected_weights, info_dict)
        """
        w = weights.copy()
        max_iter = 100
        tolerance = 1e-8
        
        for iteration in range(max_iter):
            w_old = w.copy()
            
            # Step 1: Project to bounds
            w = np.clip(w, self.bounds[0], self.bounds[1])
            
            # Step 2: Project to budget while respecting bounds
            current_sum = np.sum(w)
            deficit = self.budget - current_sum
            
            if abs(deficit) > tolerance:
                # Adjust weights proportionally where possible
                if deficit > 0:
                    # Need to increase: increase weights that are below upper bound
                    room = self.bounds[1] - w
                    total_room = np.sum(room)
                    if total_room > 0:
                        adjustment = deficit * (room / total_room)
                        w = w + adjustment
                else:
                    # Need to decrease: decrease weights that are above lower bound
                    room = w - self.bounds[0]
                    total_room = np.sum(room)
                    if total_room > 0:
                        adjustment = -deficit * (room / total_room)
                        w = w - adjustment
            
            # Check convergence
            if np.linalg.norm(w - w_old) < tolerance:
                break
        
        # Final enforcement
        w = np.clip(w, self.bounds[0], self.bounds[1])
        
        # Scale to exact budget
        current_sum = np.sum(w)
        if current_sum > 0:
            w = w * (self.budget / current_sum)
            w = np.clip(w, self.bounds[0], self.bounds[1])
        
        distance = np.linalg.norm(w - weights)
        
        return w, {
            'distance': distance,
            'iterations': iteration + 1,
            'success': True
        }
    
    def _simple_budget_projection(self, weights: np.ndarray) -> np.ndarray:
        """Simple budget projection for initialization.
        
        Args:
            weights: Initial weights
            
        Returns:
            Weights projected to budget (may violate bounds)
        """
        current_sum = np.sum(weights)
        if current_sum == 0:
            # Equal allocation if all zeros
            return np.full(self.num_assets, self.budget / self.num_assets)
        else:
            # Scale to budget
            return weights * (self.budget / current_sum)


class HybridVQESolver:
    """Hybrid VQE solver with classical post-processing.
    
    Workflow:
    1. Run VQE to find low-energy solution (may be infeasible)
    2. Project to feasible set using classical optimization
    3. Return feasible solution with quantum-inspired initialization
    """
    
    def __init__(self, 
                 vqe_optimizer,
                 mapper,
                 result_decoder):
        """Initialize hybrid solver.
        
        Args:
            vqe_optimizer: VQEOptimizer instance
            mapper: IsingMapper instance
            result_decoder: ResultDecoder instance
        """
        self.vqe_optimizer = vqe_optimizer
        self.mapper = mapper
        self.decoder = result_decoder
        self.projector = FeasibilityProjection(mapper.model)
    
    def solve(self, initial_parameters: Optional[np.ndarray] = None) -> Dict:
        """Solve portfolio optimization using hybrid approach.
        
        Args:
            initial_parameters: Initial VQE parameters
            
        Returns:
            Dictionary with hybrid solution
        """
        print("="*60)
        print("HYBRID CLASSICAL-QUANTUM PORTFOLIO OPTIMIZATION")
        print("="*60)
        print()
        
        # Step 1: Run VQE
        print("[1/3] Running VQE for quantum exploration...")
        vqe_result = self.vqe_optimizer.optimize(initial_parameters=initial_parameters)
        
        # Step 2: Decode VQE solution
        print()
        print("[2/3] Decoding VQE measurements...")
        state_result = self.vqe_optimizer.evaluate_final_state(vqe_result['optimal_parameters'])
        decoded = self.decoder.decode_counts(state_result['counts'])
        
        print(f"  VQE solution: {decoded['best_weights']}")
        print(f"  Feasible: {decoded['is_feasible']}")
        
        if decoded['best_weights'] is not None:
            vqe_sum = np.sum(decoded['best_weights'])
            print(f"  Weight sum: {vqe_sum:.4f} (target: {self.projector.budget})")
        print()
        
        # Step 3: Project to feasible set
        print("[3/3] Projecting to feasible set...")
        
        if decoded['best_weights'] is None:
            print("  ✗ No valid VQE solution, using equal weights")
            vqe_weights = np.full(self.projector.num_assets, 
                                 self.projector.budget / self.projector.num_assets)
        else:
            vqe_weights = decoded['best_weights']
        
        projection = self.projector.project_to_feasible(vqe_weights, method='slsqp')
        
        print(f"  Projection method: {projection['method']}")
        print(f"  Was feasible: {projection['was_feasible']}")
        
        if not projection['was_feasible']:
            print(f"  Projection distance: {projection['projection_distance']:.6f}")
            print(f"  Iterations: {projection['iterations']}")
        
        final_weights = projection['weights']
        print(f"  Final weights: {final_weights}")
        print(f"  Final sum: {np.sum(final_weights):.6f}")
        
        # Verify feasibility
        is_feasible = self.projector._is_feasible(final_weights)
        print(f"  ✓ Feasible: {is_feasible}")
        print()
        
        return {
            'final_weights': final_weights,
            'is_feasible': is_feasible,
            'vqe_result': vqe_result,
            'vqe_decoded': decoded,
            'projection': projection,
            'vqe_weights': vqe_weights
        }

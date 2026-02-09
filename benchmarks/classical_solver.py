"""Classical solver for Markowitz portfolio optimization.

This module solves the Markowitz model defined in finance/markowitz.py
using classical optimization algorithms (SciPy). It does NOT redefine
the model, only solves it.

Serves as:
1. Ground truth for validation of quantum solvers
2. Baseline for performance comparison
3. Fallback when quantum solver fails to converge
"""

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from typing import Optional, Dict, Any
import sys
sys.path.append('/app')

from finance.markowitz import MarkowitzModel


class ClassicalSolver:
    """Solves Markowitz portfolio optimization using classical methods."""
    
    def __init__(self, model: MarkowitzModel):
        """Initialize solver with a Markowitz model.
        
        Args:
            model: MarkowitzModel instance defining the problem
        """
        self.model = model
    
    def solve(self, 
              method: str = 'SLSQP',
              initial_weights: Optional[np.ndarray] = None,
              verbose: bool = False) -> Dict[str, Any]:
        """Solve portfolio optimization problem.
        
        Args:
            method: SciPy optimization method ('SLSQP', 'COBYLA', 'trust-constr')
            initial_weights: Initial guess (if None, uses equal weights)
            verbose: Whether to print optimization progress
            
        Returns:
            Dictionary containing:
                - weights: Optimal portfolio weights
                - objective: Optimal objective value
                - risk: Portfolio variance
                - return: Expected return
                - success: Whether optimization succeeded
                - message: Solver message
                - n_iterations: Number of iterations
        """
        # Get initial guess
        if initial_weights is None:
            initial_weights = self.model.get_initial_guess()
        
        # Define objective function for SciPy
        def objective_func(w):
            return self.model.objective.compute_objective(w)
        
        # Define gradient (for gradient-based methods)
        def gradient_func(w):
            return self.model.objective.compute_gradient(w)
        
        # Define budget constraint
        budget_constraint = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - self.model.constraints.budget
        }
        
        # Define bounds
        bounds = [self.model.constraints.bounds] * self.model.num_assets
        
        # Solve using SciPy
        result: OptimizeResult = minimize(
            fun=objective_func,
            x0=initial_weights,
            method=method,
            jac=gradient_func if method in ['SLSQP', 'trust-constr'] else None,
            bounds=bounds,
            constraints=[budget_constraint],
            options={'disp': verbose}
        )
        
        # Extract results
        optimal_weights = result.x
        objective_value = result.fun
        risk = self.model.objective.compute_risk(optimal_weights)
        expected_return = self.model.objective.compute_return(optimal_weights)
        
        return {
            'weights': optimal_weights,
            'objective': objective_value,
            'risk': risk,
            'return': expected_return,
            'success': result.success,
            'message': result.message,
            'n_iterations': result.nit if hasattr(result, 'nit') else None
        }
    
    def solve_efficient_frontier(self, 
                                 risk_aversion_range: np.ndarray,
                                 method: str = 'SLSQP') -> Dict[str, np.ndarray]:
        """Compute the efficient frontier by varying risk aversion.
        
        Args:
            risk_aversion_range: Array of risk aversion values to try
            method: Optimization method
            
        Returns:
            Dictionary with arrays of risks, returns, and weights for each λ
        """
        n_points = len(risk_aversion_range)
        risks = np.zeros(n_points)
        returns = np.zeros(n_points)
        weights_list = []
        
        for i, lambda_val in enumerate(risk_aversion_range):
            # Update model risk aversion
            self.model.objective.risk_aversion = lambda_val
            
            # Solve
            result = self.solve(method=method, verbose=False)
            
            if result['success']:
                risks[i] = result['risk']
                returns[i] = result['return']
                weights_list.append(result['weights'])
            else:
                # Mark failed optimization
                risks[i] = np.nan
                returns[i] = np.nan
                weights_list.append(np.full(self.model.num_assets, np.nan))
        
        return {
            'risks': risks,
            'returns': returns,
            'weights': np.array(weights_list),
            'risk_aversions': risk_aversion_range
        }

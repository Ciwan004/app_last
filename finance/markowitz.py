"""Markowitz mean-variance portfolio optimization model.

This is the SINGLE SOURCE OF TRUTH for the Markowitz model definition.
Defines the objective function and constraints. Does NOT solve the problem.

Mathematical Formulation:
    Objective: minimize  (1/2) * w^T Σ w - λ * μ^T w
    Subject to:
        - Budget constraint: Σ w_i = budget
        - Bounds: w_min ≤ w_i ≤ w_max
        - (Optional) Cardinality: Σ I(w_i > 0) ≤ K

Where:
    w = portfolio weights
    Σ = covariance matrix (risk)
    μ = expected returns
    λ = risk aversion coefficient (tradeoff parameter)
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class MarkowitzObjective:
    """Encapsulates the Markowitz objective function and its components.
    
    Attributes:
        expected_returns: Expected return for each asset (μ)
        covariance_matrix: Asset covariance matrix (Σ)
        risk_aversion: Risk aversion coefficient (λ)
    """
    expected_returns: np.ndarray
    covariance_matrix: np.ndarray
    risk_aversion: float
    
    def compute_risk(self, weights: np.ndarray) -> float:
        """Compute portfolio variance (risk).
        
        Args:
            weights: Portfolio weights (w)
            
        Returns:
            Portfolio variance: w^T Σ w
        """
        return float(weights.T @ self.covariance_matrix @ weights)
    
    def compute_return(self, weights: np.ndarray) -> float:
        """Compute portfolio expected return.
        
        Args:
            weights: Portfolio weights (w)
            
        Returns:
            Expected return: μ^T w
        """
        return float(self.expected_returns @ weights)
    
    def compute_objective(self, weights: np.ndarray) -> float:
        """Compute Markowitz objective function value.
        
        Objective: (1/2) * w^T Σ w - λ * μ^T w
        
        Interpretation:
        - First term: portfolio risk (variance)
        - Second term: expected return (scaled by risk aversion)
        - Minimizing this balances risk vs. return
        
        Args:
            weights: Portfolio weights (w)
            
        Returns:
            Objective value (to be minimized)
        """
        risk = self.compute_risk(weights)
        expected_return = self.compute_return(weights)
        return 0.5 * risk - self.risk_aversion * expected_return
    
    def compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        """Compute gradient of objective with respect to weights.
        
        Gradient: ∇f(w) = Σw - λμ
        
        Args:
            weights: Portfolio weights (w)
            
        Returns:
            Gradient vector
        """
        return self.covariance_matrix @ weights - self.risk_aversion * self.expected_returns


@dataclass
class MarkowitzConstraints:
    """Encapsulates constraints for portfolio optimization.
    
    Attributes:
        budget: Total budget (typically 1.0 for normalized weights)
        bounds: (min, max) weight bounds for each asset
        num_assets: Number of assets
        cardinality: Maximum number of non-zero weights (None = no limit)
    """
    budget: float
    bounds: Tuple[float, float]
    num_assets: int
    cardinality: Optional[int] = None
    
    def budget_constraint(self, weights: np.ndarray) -> float:
        """Evaluate budget constraint residual.
        
        Constraint: Σ w_i = budget
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Constraint residual (should be 0 at optimum)
        """
        return np.sum(weights) - self.budget
    
    def check_bounds(self, weights: np.ndarray) -> bool:
        """Check if weights satisfy bound constraints.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            True if all bounds satisfied
        """
        return np.all(weights >= self.bounds[0]) and np.all(weights <= self.bounds[1])
    
    def check_cardinality(self, weights: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if cardinality constraint is satisfied.
        
        Args:
            weights: Portfolio weights
            tolerance: Threshold for considering weight as non-zero
            
        Returns:
            True if cardinality constraint satisfied (or not active)
        """
        if self.cardinality is None:
            return True
        num_nonzero = np.sum(np.abs(weights) > tolerance)
        return num_nonzero <= self.cardinality
    
    def is_feasible(self, weights: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if weights satisfy all constraints.
        
        Args:
            weights: Portfolio weights
            tolerance: Numerical tolerance
            
        Returns:
            True if all constraints satisfied
        """
        # Check budget
        if abs(self.budget_constraint(weights)) > tolerance:
            return False
        
        # Check bounds
        if not self.check_bounds(weights):
            return False
        
        # Check cardinality
        if not self.check_cardinality(weights, tolerance):
            return False
        
        return True


class MarkowitzModel:
    """Complete Markowitz portfolio optimization model.
    
    This class defines the optimization problem but does NOT solve it.
    Solvers (classical or quantum) consume this model definition.
    """
    
    def __init__(self,
                 expected_returns: np.ndarray,
                 covariance_matrix: np.ndarray,
                 risk_aversion: float,
                 budget: float = 1.0,
                 bounds: Tuple[float, float] = (0.0, 1.0),
                 cardinality: Optional[int] = None):
        """Initialize Markowitz model.
        
        Args:
            expected_returns: Expected return for each asset
            covariance_matrix: Asset covariance matrix
            risk_aversion: Risk aversion coefficient λ
            budget: Budget constraint
            bounds: Weight bounds (min, max)
            cardinality: Maximum number of non-zero weights
        """
        self.num_assets = len(expected_returns)
        
        # Create objective and constraints
        self.objective = MarkowitzObjective(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            risk_aversion=risk_aversion
        )
        
        self.constraints = MarkowitzConstraints(
            budget=budget,
            bounds=bounds,
            num_assets=self.num_assets,
            cardinality=cardinality
        )
    
    def evaluate(self, weights: np.ndarray) -> Tuple[float, bool]:
        """Evaluate objective and feasibility for given weights.
        
        Args:
            weights: Portfolio weights to evaluate
            
        Returns:
            Tuple of (objective_value, is_feasible)
        """
        obj_value = self.objective.compute_objective(weights)
        is_feasible = self.constraints.is_feasible(weights)
        return obj_value, is_feasible
    
    def get_initial_guess(self) -> np.ndarray:
        """Generate a feasible initial guess for optimization.
        
        Returns:
            Equal-weighted portfolio satisfying budget constraint
        """
        return np.full(self.num_assets, self.constraints.budget / self.num_assets)

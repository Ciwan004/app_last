"""Portfolio data validation utilities.

Enforces data integrity: checks dimensions, validates covariance matrix properties,
ensures no missing values. Fails fast with clear error messages.
"""

import numpy as np
from typing import Tuple


class DataValidator:
    """Validates portfolio data before optimization."""
    
    @staticmethod
    def validate_portfolio_data(expected_returns: np.ndarray, 
                               covariance_matrix: np.ndarray,
                               num_assets: int) -> None:
        """Validate portfolio data for correctness.
        
        Args:
            expected_returns: Expected return vector
            covariance_matrix: Covariance matrix
            num_assets: Expected number of assets
            
        Raises:
            ValueError: If data validation fails
        """
        # Check expected returns
        if expected_returns.ndim != 1:
            raise ValueError(f"expected_returns must be 1D, got shape {expected_returns.shape}")
        
        if len(expected_returns) != num_assets:
            raise ValueError(
                f"expected_returns length {len(expected_returns)} != num_assets {num_assets}"
            )
        
        if np.any(np.isnan(expected_returns)):
            raise ValueError("expected_returns contains NaN values")
        
        if np.any(np.isinf(expected_returns)):
            raise ValueError("expected_returns contains infinite values")
        
        # Check covariance matrix
        if covariance_matrix.ndim != 2:
            raise ValueError(f"covariance_matrix must be 2D, got shape {covariance_matrix.shape}")
        
        if covariance_matrix.shape != (num_assets, num_assets):
            raise ValueError(
                f"covariance_matrix shape {covariance_matrix.shape} != ({num_assets}, {num_assets})"
            )
        
        if np.any(np.isnan(covariance_matrix)):
            raise ValueError("covariance_matrix contains NaN values")
        
        if np.any(np.isinf(covariance_matrix)):
            raise ValueError("covariance_matrix contains infinite values")
        
        # Check symmetry
        if not np.allclose(covariance_matrix, covariance_matrix.T):
            raise ValueError("covariance_matrix must be symmetric")
        
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        if np.any(eigenvalues < -1e-10):  # Allow small numerical errors
            raise ValueError(
                f"covariance_matrix must be positive semi-definite, "
                f"got min eigenvalue {eigenvalues.min():.6e}"
            )
        
    @staticmethod
    def validate_weights(weights: np.ndarray, 
                        num_assets: int,
                        budget: float = 1.0,
                        bounds: Tuple[float, float] = (0.0, 1.0),
                        tolerance: float = 1e-6) -> None:
        """Validate portfolio weights for correctness.
        
        Args:
            weights: Portfolio weights
            num_assets: Expected number of assets
            budget: Expected budget constraint sum
            bounds: (min, max) bounds for each weight
            tolerance: Numerical tolerance for constraint checks
            
        Raises:
            ValueError: If weights validation fails
        """
        if weights.ndim != 1:
            raise ValueError(f"weights must be 1D, got shape {weights.shape}")
        
        if len(weights) != num_assets:
            raise ValueError(f"weights length {len(weights)} != num_assets {num_assets}")
        
        if np.any(np.isnan(weights)):
            raise ValueError("weights contains NaN values")
        
        if np.any(np.isinf(weights)):
            raise ValueError("weights contains infinite values")
        
        # Check budget constraint
        weight_sum = np.sum(weights)
        if abs(weight_sum - budget) > tolerance:
            raise ValueError(
                f"weights sum {weight_sum:.6f} != budget {budget:.6f} "
                f"(tolerance {tolerance})"
            )
        
        # Check bounds
        if np.any(weights < bounds[0] - tolerance):
            raise ValueError(f"Some weights below lower bound {bounds[0]}")
        
        if np.any(weights > bounds[1] + tolerance):
            raise ValueError(f"Some weights above upper bound {bounds[1]}")

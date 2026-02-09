"""Portfolio data loading utilities.

Responsible for ingesting portfolio data from various sources
and outputting cleaned numpy arrays ready for optimization.
"""

import numpy as np
from typing import Tuple


class PortfolioDataLoader:
    """Loads and prepares portfolio optimization data."""
    
    @staticmethod
    def load_toy_data(num_assets: int = 4, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic portfolio data for testing.
        
        Creates a random but realistic covariance matrix and expected returns
        for a toy portfolio optimization problem.
        
        Args:
            num_assets: Number of assets
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (expected_returns, covariance_matrix)
            - expected_returns: shape (num_assets,) with annual returns
            - covariance_matrix: shape (num_assets, num_assets), symmetric positive definite
        """
        np.random.seed(seed)
        
        # Generate expected returns: random values between 5% and 15% annual return
        expected_returns = np.random.uniform(0.05, 0.15, size=num_assets)
        
        # Generate covariance matrix using random correlation + volatilities
        # 1. Generate random volatilities (annual standard deviation)
        volatilities = np.random.uniform(0.1, 0.3, size=num_assets)
        
        # 2. Generate random correlation matrix
        # Start with random matrix, make symmetric, ensure positive definiteness
        A = np.random.uniform(-0.5, 0.5, size=(num_assets, num_assets))
        correlation = A @ A.T  # Guaranteed positive semi-definite
        
        # Normalize to get correlations in [-1, 1]
        std_matrix = np.sqrt(np.diag(correlation))
        correlation = correlation / np.outer(std_matrix, std_matrix)
        
        # 3. Convert correlation to covariance
        covariance_matrix = np.outer(volatilities, volatilities) * correlation
        
        return expected_returns, covariance_matrix
    
    @staticmethod
    def load_from_arrays(expected_returns: np.ndarray, 
                         covariance_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Load portfolio data from numpy arrays.
        
        Args:
            expected_returns: Expected return for each asset
            covariance_matrix: Asset return covariance matrix
            
        Returns:
            Tuple of (expected_returns, covariance_matrix) as copies
        """
        return expected_returns.copy(), covariance_matrix.copy()
    
    @staticmethod
    def get_4asset_example() -> Tuple[np.ndarray, np.ndarray]:
        """Returns a fixed 4-asset example for reproducible testing.
        
        This represents a realistic small portfolio:
        - Asset 0: Low risk, low return (e.g., bonds)
        - Asset 1: Medium risk, medium return (e.g., blue-chip stocks)
        - Asset 2: Higher risk, higher return (e.g., growth stocks)
        - Asset 3: High risk, high return (e.g., emerging markets)
        
        Returns:
            Tuple of (expected_returns, covariance_matrix)
        """
        expected_returns = np.array([0.06, 0.08, 0.12, 0.15])  # Annual returns
        
        # Covariance matrix (annualized)
        covariance_matrix = np.array([
            [0.04, 0.01, 0.02, 0.01],  # Asset 0: σ ≈ 0.20 (low volatility)
            [0.01, 0.09, 0.03, 0.02],  # Asset 1: σ ≈ 0.30
            [0.02, 0.03, 0.16, 0.04],  # Asset 2: σ ≈ 0.40
            [0.01, 0.02, 0.04, 0.25]   # Asset 3: σ ≈ 0.50 (high volatility)
        ])
        
        return expected_returns, covariance_matrix

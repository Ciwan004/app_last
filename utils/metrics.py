"""Portfolio performance metrics.

Computes standard portfolio metrics for evaluation and comparison:
- Sharpe ratio
- Volatility (standard deviation)
- Maximum drawdown
- Information ratio
"""

import numpy as np
from typing import Dict, Optional


class PortfolioMetrics:
    """Computes portfolio performance metrics."""
    
    @staticmethod
    def sharpe_ratio(expected_return: float, 
                    volatility: float, 
                    risk_free_rate: float = 0.02) -> float:
        """Compute Sharpe ratio.
        
        Sharpe ratio = (E[R] - R_f) / σ
        
        Measures risk-adjusted return. Higher is better.
        
        Args:
            expected_return: Expected portfolio return
            volatility: Portfolio standard deviation
            risk_free_rate: Risk-free rate (default 2% annual)
            
        Returns:
            Sharpe ratio
        """
        if volatility == 0:
            return np.inf if expected_return > risk_free_rate else -np.inf
        return (expected_return - risk_free_rate) / volatility
    
    @staticmethod
    def volatility(covariance_matrix: np.ndarray, weights: np.ndarray) -> float:
        """Compute portfolio volatility (standard deviation).
        
        σ = sqrt(w^T Σ w)
        
        Args:
            covariance_matrix: Asset covariance matrix
            weights: Portfolio weights
            
        Returns:
            Portfolio standard deviation (annualized)
        """
        variance = weights.T @ covariance_matrix @ weights
        return float(np.sqrt(variance))
    
    @staticmethod
    def expected_return(returns: np.ndarray, weights: np.ndarray) -> float:
        """Compute portfolio expected return.
        
        E[R] = μ^T w
        
        Args:
            returns: Expected returns for each asset
            weights: Portfolio weights
            
        Returns:
            Expected portfolio return (annualized)
        """
        return float(returns @ weights)
    
    @staticmethod
    def compute_all_metrics(weights: np.ndarray,
                           expected_returns: np.ndarray,
                           covariance_matrix: np.ndarray,
                           risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Compute all standard portfolio metrics.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            risk_free_rate: Risk-free rate for Sharpe calculation
            
        Returns:
            Dictionary with all computed metrics
        """
        portfolio_return = PortfolioMetrics.expected_return(expected_returns, weights)
        portfolio_vol = PortfolioMetrics.volatility(covariance_matrix, weights)
        sharpe = PortfolioMetrics.sharpe_ratio(portfolio_return, portfolio_vol, risk_free_rate)
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'variance': portfolio_vol ** 2,
            'sharpe_ratio': sharpe,
            'risk_free_rate': risk_free_rate
        }
    
    @staticmethod
    def compare_solutions(weights1: np.ndarray,
                         weights2: np.ndarray,
                         expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray,
                         labels: tuple = ('Solution 1', 'Solution 2')) -> Dict[str, Dict[str, float]]:
        """Compare metrics between two portfolio solutions.
        
        Args:
            weights1: First portfolio weights
            weights2: Second portfolio weights
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            labels: Names for the two solutions
            
        Returns:
            Dictionary with metrics for both solutions and differences
        """
        metrics1 = PortfolioMetrics.compute_all_metrics(weights1, expected_returns, covariance_matrix)
        metrics2 = PortfolioMetrics.compute_all_metrics(weights2, expected_returns, covariance_matrix)
        
        # Compute differences
        differences = {
            key: metrics2[key] - metrics1[key]
            for key in metrics1.keys()
        }
        
        # Add weight difference metrics
        differences['weight_l1_distance'] = float(np.sum(np.abs(weights2 - weights1)))
        differences['weight_l2_distance'] = float(np.linalg.norm(weights2 - weights1))
        differences['weight_max_diff'] = float(np.max(np.abs(weights2 - weights1)))
        
        return {
            labels[0]: metrics1,
            labels[1]: metrics2,
            'differences': differences
        }

"""Portfolio optimization configuration.

Defines risk parameters, constraints, and asset universe settings.
Serves as the single source of truth for problem parameters.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization problem.
    
    Attributes:
        num_assets: Number of assets in the portfolio
        risk_aversion: Risk aversion coefficient λ (higher = prefer lower risk)
        budget: Total budget constraint (typically 1.0 for normalized weights)
        bounds: (min, max) weight bounds for each asset
        cardinality: Maximum number of assets to select (None = no limit)
        allow_short: Whether to allow short positions (negative weights)
    """
    num_assets: int
    risk_aversion: float = 0.5
    budget: float = 1.0
    bounds: tuple[float, float] = (0.0, 1.0)
    cardinality: Optional[int] = None
    allow_short: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_assets <= 0:
            raise ValueError("num_assets must be positive")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.bounds[0] > self.bounds[1]:
            raise ValueError("bounds[0] must be <= bounds[1]")
        if self.cardinality is not None and self.cardinality > self.num_assets:
            raise ValueError("cardinality cannot exceed num_assets")


def get_toy_problem_config() -> PortfolioConfig:
    """Returns a 4-asset toy problem configuration for testing.
    
    Returns:
        PortfolioConfig with 4 assets, moderate risk aversion, no short selling
    """
    return PortfolioConfig(
        num_assets=4,
        risk_aversion=0.5,  # Balanced risk-return tradeoff
        budget=1.0,
        bounds=(0.0, 1.0),  # Long-only portfolio
        cardinality=None,   # No cardinality constraint for initial testing
        allow_short=False
    )

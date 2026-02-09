"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np


class OptimizationRequest(BaseModel):
    """Request schema for portfolio optimization."""
    
    expected_returns: List[float] = Field(
        ...,
        description="Expected returns for each asset",
        example=[0.06, 0.08, 0.12, 0.15]
    )
    
    covariance_matrix: List[List[float]] = Field(
        ...,
        description="Covariance matrix (must be symmetric and positive definite)",
        example=[
            [0.04, 0.01, 0.02, 0.01],
            [0.01, 0.09, 0.03, 0.02],
            [0.02, 0.03, 0.16, 0.04],
            [0.01, 0.02, 0.04, 0.25]
        ]
    )
    
    risk_aversion: float = Field(
        default=0.5,
        ge=0.0,
        description="Risk aversion coefficient (λ). Higher = prefer lower risk",
        example=0.5
    )
    
    budget: float = Field(
        default=1.0,
        gt=0.0,
        description="Total budget constraint (typically 1.0)",
        example=1.0
    )
    
    bounds: List[float] = Field(
        default=[0.0, 1.0],
        description="[min, max] weight bounds for each asset",
        example=[0.0, 1.0]
    )
    
    # Quantum configuration
    num_bins: int = Field(
        default=5,
        ge=3,
        le=11,
        description="Number of discrete weight bins (3-11). More bins = better precision but more qubits",
        example=5
    )
    
    penalty_coefficient: float = Field(
        default=1000.0,
        ge=100.0,
        description="Penalty coefficient for constraint violations",
        example=1000.0
    )
    
    ansatz_type: str = Field(
        default="RealAmplitudes",
        description="Variational ansatz type",
        example="RealAmplitudes"
    )
    
    ansatz_reps: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Ansatz repetitions (circuit depth)",
        example=2
    )
    
    max_vqe_iterations: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum VQE optimization iterations",
        example=100
    )
    
    optimizer: str = Field(
        default="COBYLA",
        description="Classical optimizer for VQE",
        example="COBYLA"
    )
    
    @validator('covariance_matrix')
    def validate_covariance(cls, v, values):
        """Validate covariance matrix dimensions."""
        if 'expected_returns' not in values:
            return v
        
        n = len(values['expected_returns'])
        if len(v) != n:
            raise ValueError(f"Covariance matrix must be {n}x{n}")
        
        for row in v:
            if len(row) != n:
                raise ValueError(f"Covariance matrix must be square ({n}x{n})")
        
        return v
    
    @validator('bounds')
    def validate_bounds(cls, v):
        """Validate bounds."""
        if len(v) != 2:
            raise ValueError("Bounds must be [min, max]")
        if v[0] > v[1]:
            raise ValueError("bounds[0] must be <= bounds[1]")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "expected_returns": [0.06, 0.08, 0.12, 0.15],
                "covariance_matrix": [
                    [0.04, 0.01, 0.02, 0.01],
                    [0.01, 0.09, 0.03, 0.02],
                    [0.02, 0.03, 0.16, 0.04],
                    [0.01, 0.02, 0.04, 0.25]
                ],
                "risk_aversion": 0.5,
                "budget": 1.0,
                "num_bins": 5,
                "max_vqe_iterations": 100
            }
        }


class PortfolioMetrics(BaseModel):
    """Portfolio performance metrics."""
    expected_return: float = Field(description="Expected portfolio return")
    volatility: float = Field(description="Portfolio volatility (standard deviation)")
    variance: float = Field(description="Portfolio variance")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    risk_free_rate: float = Field(description="Risk-free rate used")


class OptimizationResult(BaseModel):
    """Response schema for optimization results."""
    
    # Solution
    weights: List[float] = Field(description="Optimal portfolio weights")
    is_feasible: bool = Field(description="Whether solution is feasible")
    
    # Portfolio metrics
    metrics: PortfolioMetrics = Field(description="Portfolio performance metrics")
    
    # Optimization info
    vqe_iterations: int = Field(description="Number of VQE iterations")
    vqe_convergence_energy: float = Field(description="Final VQE energy")
    vqe_execution_time: float = Field(description="VQE execution time (seconds)")
    
    # Hybrid solver info
    projection_distance: float = Field(description="Distance between VQE and projected solution")
    vqe_weights: List[float] = Field(description="Raw VQE weights before projection")
    vqe_was_feasible: bool = Field(description="Whether VQE solution was feasible")
    
    # Classical comparison
    classical_weights: List[float] = Field(description="Classical optimal weights")
    classical_objective: float = Field(description="Classical objective value")
    hybrid_objective: float = Field(description="Hybrid objective value")
    objective_gap: float = Field(description="Difference: hybrid - classical")
    approximation_ratio: float = Field(description="Ratio: hybrid / classical")
    
    # Quantum config
    num_qubits: int = Field(description="Number of qubits used")
    num_bins: int = Field(description="Number of discretization bins")
    
    class Config:
        schema_extra = {
            "example": {
                "weights": [0.417, 0.0, 0.542, 0.042],
                "is_feasible": True,
                "metrics": {
                    "expected_return": 0.0962,
                    "volatility": 0.2560,
                    "variance": 0.0655,
                    "sharpe_ratio": 0.2979,
                    "risk_free_rate": 0.02
                },
                "vqe_iterations": 100,
                "vqe_convergence_energy": 12870.4,
                "vqe_execution_time": 28.98,
                "projection_distance": 0.361,
                "vqe_was_feasible": False,
                "classical_objective": -0.027032,
                "hybrid_objective": -0.024156,
                "objective_gap": 0.002876,
                "approximation_ratio": 0.894,
                "num_qubits": 20,
                "num_bins": 5
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    backend: str = Field(description="Quantum backend")
    backend_available: bool = Field(description="Whether backend is available")
    components: Dict[str, str] = Field(description="Status of each component")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "backend": "aer_simulator",
                "backend_available": True,
                "components": {
                    "classical_solver": "ok",
                    "quantum_mapper": "ok",
                    "vqe_optimizer": "ok",
                    "hybrid_solver": "ok"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid covariance matrix: must be symmetric",
                "details": {"row": 0, "col": 1}
            }
        }

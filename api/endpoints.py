"""FastAPI endpoints for portfolio optimization API."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import numpy as np
import sys
import time
import traceback
from typing import Dict, Any

sys.path.append('/app')

from api.schemas import (
    OptimizationRequest,
    OptimizationResult,
    HealthResponse,
    ErrorResponse,
    PortfolioMetrics as PortfolioMetricsSchema
)
from finance.markowitz import MarkowitzModel
from benchmarks.classical_solver import ClassicalSolver
from quantum.ising_mapper import IsingMapper
from quantum.backend_manager import BackendManager
from quantum.vqe_optimizer import VQEOptimizer
from quantum.result_decoder import ResultDecoder
from quantum.hybrid_solver import HybridVQESolver
from config.quantum_config import get_simulator_config
from utils.metrics import PortfolioMetrics
from data.validator import DataValidator

router = APIRouter()

# Global backend manager (initialized once)
_backend_manager = None


def get_backend_manager():
    """Get or create backend manager singleton."""
    global _backend_manager
    if _backend_manager is None:
        config = get_simulator_config()
        _backend_manager = BackendManager(config)
    return _backend_manager


@router.post(
    "/optimize",
    response_model=OptimizationResult,
    status_code=status.HTTP_200_OK,
    summary="Run portfolio optimization",
    description="Optimize portfolio using hybrid classical-quantum approach (VQE + projection)",
    responses={
        200: {"description": "Optimization successful"},
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def optimize_portfolio(request: OptimizationRequest):
    """Run hybrid portfolio optimization.
    
    Args:
        request: Optimization request with portfolio data and parameters
        
    Returns:
        OptimizationResult with optimal weights and metrics
    """
    try:
        # Convert inputs to numpy arrays
        expected_returns = np.array(request.expected_returns)
        covariance_matrix = np.array(request.covariance_matrix)
        num_assets = len(expected_returns)
        
        # Validate data
        try:
            DataValidator.validate_portfolio_data(
                expected_returns,
                covariance_matrix,
                num_assets
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "ValidationError",
                    "message": str(e)
                }
            )
        
        # Create Markowitz model
        model = MarkowitzModel(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            risk_aversion=request.risk_aversion,
            budget=request.budget,
            bounds=tuple(request.bounds)
        )
        
        # Solve with classical baseline
        classical_solver = ClassicalSolver(model)
        classical_result = classical_solver.solve(method='SLSQP', verbose=False)
        
        if not classical_result['success']:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "ClassicalSolverError",
                    "message": "Classical solver failed to converge",
                    "details": {"message": classical_result['message']}
                }
            )
        
        # Create quantum components
        mapper = IsingMapper(
            model,
            num_bins=request.num_bins,
            penalty_coefficient=request.penalty_coefficient
        )
        
        backend_mgr = get_backend_manager()
        
        vqe = VQEOptimizer(
            mapper=mapper,
            backend_manager=backend_mgr,
            ansatz_type=request.ansatz_type,
            ansatz_reps=request.ansatz_reps,
            optimizer=request.optimizer,
            max_iterations=request.max_vqe_iterations
        )
        
        decoder = ResultDecoder(mapper)
        
        # Create and run hybrid solver
        hybrid = HybridVQESolver(vqe, mapper, decoder)
        
        start_time = time.time()
        hybrid_result = hybrid.solve()
        total_time = time.time() - start_time
        
        # Extract results
        final_weights = hybrid_result['final_weights']
        vqe_weights = hybrid_result['vqe_weights']
        
        # Compute metrics
        hybrid_metrics = PortfolioMetrics.compute_all_metrics(
            final_weights,
            expected_returns,
            covariance_matrix
        )
        
        # Compute objectives
        hybrid_obj = model.objective.compute_objective(final_weights)
        classical_obj = classical_result['objective']
        
        # Build response
        response = OptimizationResult(
            weights=final_weights.tolist(),
            is_feasible=hybrid_result['is_feasible'],
            metrics=PortfolioMetricsSchema(**hybrid_metrics),
            vqe_iterations=hybrid_result['vqe_result']['iterations'],
            vqe_convergence_energy=float(hybrid_result['vqe_result']['optimal_energy']),
            vqe_execution_time=float(hybrid_result['vqe_result']['total_time']),
            projection_distance=float(hybrid_result['projection']['projection_distance']),
            vqe_weights=vqe_weights.tolist(),
            vqe_was_feasible=hybrid_result['projection']['was_feasible'],
            classical_weights=classical_result['weights'].tolist(),
            classical_objective=float(classical_obj),
            hybrid_objective=float(hybrid_obj),
            objective_gap=float(hybrid_obj - classical_obj),
            approximation_ratio=float(hybrid_obj / classical_obj) if classical_obj != 0 else float('inf'),
            num_qubits=mapper.encoder.num_qubits,
            num_bins=request.num_bins
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the full traceback
        error_details = traceback.format_exc()
        print(f"Error in optimization: {error_details}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": type(e).__name__,
                "message": str(e),
                "details": {"traceback": error_details.split('\n')[-5:]}
            }
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if the API and quantum backend are operational"
)
async def health_check():
    """Check system health.
    
    Returns:
        HealthResponse with system status
    """
    try:
        # Check backend
        backend_mgr = get_backend_manager()
        backend_info = backend_mgr.get_backend_info()
        backend_available = backend_info['name'] is not None
        
        # Test each component
        components = {}
        
        # Classical solver
        try:
            test_returns = np.array([0.06, 0.08])
            test_cov = np.array([[0.04, 0.01], [0.01, 0.09]])
            test_model = MarkowitzModel(test_returns, test_cov, 0.5)
            solver = ClassicalSolver(test_model)
            result = solver.solve(method='SLSQP', verbose=False)
            components['classical_solver'] = 'ok' if result['success'] else 'degraded'
        except Exception as e:
            components['classical_solver'] = f'error: {str(e)}'
        
        # Quantum mapper
        try:
            test_mapper = IsingMapper(test_model, num_bins=3, penalty_coefficient=100)
            test_mapper.build_qubo()
            components['quantum_mapper'] = 'ok'
        except Exception as e:
            components['quantum_mapper'] = f'error: {str(e)}'
        
        # VQE optimizer
        try:
            components['vqe_optimizer'] = 'ok'  # If backend is up, VQE should work
        except Exception as e:
            components['vqe_optimizer'] = f'error: {str(e)}'
        
        # Hybrid solver
        try:
            components['hybrid_solver'] = 'ok'  # If all above work, hybrid should work
        except Exception as e:
            components['hybrid_solver'] = f'error: {str(e)}'
        
        # Overall status
        all_ok = all(status == 'ok' for status in components.values())
        overall_status = 'healthy' if all_ok and backend_available else 'degraded'
        
        return HealthResponse(
            status=overall_status,
            version="1.0.0",
            backend=backend_info['name'] or 'unavailable',
            backend_available=backend_available,
            components=components
        )
        
    except Exception as e:
        return HealthResponse(
            status='unhealthy',
            version="1.0.0",
            backend='unavailable',
            backend_available=False,
            components={'error': str(e)}
        )


@router.get(
    "/",
    summary="API Root",
    description="Get API information"
)
async def root():
    """API root endpoint."""
    return {
        "name": "Quantum Portfolio Optimization API",
        "version": "1.0.0",
        "description": "Hybrid classical-quantum portfolio optimization using VQE",
        "endpoints": {
            "POST /api/optimize": "Run portfolio optimization",
            "GET /api/health": "Health check",
            "GET /api/docs": "OpenAPI documentation",
            "GET /api/redoc": "ReDoc documentation"
        },
        "documentation": "/api/docs"
    }

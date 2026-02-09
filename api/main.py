"""FastAPI application for quantum portfolio optimization."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys

sys.path.append('/app')

from api.endpoints import router

# Create FastAPI app
app = FastAPI(
    title="Quantum Portfolio Optimization API",
    description="""
    Hybrid classical-quantum portfolio optimization system using VQE (Variational Quantum Eigensolver)
    combined with classical projection for guaranteed feasible solutions.
    
    ## Features
    
    * **Quantum Exploration**: VQE explores quantum state space for portfolio optimization
    * **Classical Refinement**: Projection ensures feasibility (budget constraints)
    * **Production Ready**: Fast, reliable, with comprehensive error handling
    * **Flexible Configuration**: Tune quantum parameters (bins, ansatz, iterations)
    
    ## Workflow
    
    1. **Input**: Portfolio data (returns, covariance, risk aversion)
    2. **Classical Baseline**: Solve with classical optimizer (ground truth)
    3. **Quantum Optimization**: Run VQE to explore solution space
    4. **Projection**: Project VQE result to feasible set
    5. **Output**: Guaranteed feasible portfolio with metrics
    
    ## Example Usage
    
    ```python
    import requests
    
    data = {
        "expected_returns": [0.06, 0.08, 0.12, 0.15],
        "covariance_matrix": [
            [0.04, 0.01, 0.02, 0.01],
            [0.01, 0.09, 0.03, 0.02],
            [0.02, 0.03, 0.16, 0.04],
            [0.01, 0.02, 0.04, 0.25]
        ],
        "risk_aversion": 0.5,
        "num_bins": 5,
        "max_vqe_iterations": 100
    }
    
    response = requests.post("http://localhost:8001/api/optimize", json=data)
    result = response.json()
    print(f"Optimal weights: {result['weights']}")
    print(f"Sharpe ratio: {result['metrics']['sharpe_ratio']}")
    ```
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api", tags=["optimization"])


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print("Starting Quantum Portfolio Optimization API...")
    print("Backend: Aer Simulator")
    print("Solver: Hybrid VQE with Classical Projection")
    print("Documentation: http://localhost:8001/api/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

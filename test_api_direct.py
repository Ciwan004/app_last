"""Quick API validation test without running server.

Tests API components directly without HTTP.
"""

import sys
sys.path.append('/app')

import numpy as np
from api.schemas import OptimizationRequest, HealthResponse
from api.endpoints import get_backend_manager
from finance.markowitz import MarkowitzModel
from benchmarks.classical_solver import ClassicalSolver
from quantum.ising_mapper import IsingMapper
from quantum.vqe_optimizer import VQEOptimizer
from quantum.result_decoder import ResultDecoder
from quantum.hybrid_solver import HybridVQESolver


def test_api_schemas():
    """Test Pydantic schemas."""
    print("="*60)
    print("TEST 1: API Schemas")
    print("="*60)
    print()
    
    # Test OptimizationRequest
    request_data = {
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
    
    try:
        request = OptimizationRequest(**request_data)
        print("✓ OptimizationRequest schema valid")
        print(f"  Assets: {len(request.expected_returns)}")
        print(f"  Risk aversion: {request.risk_aversion}")
        print(f"  Bins: {request.num_bins}")
    except Exception as e:
        print(f"✗ Schema validation failed: {e}")
        return False
    
    print()
    return True


def test_backend_manager():
    """Test backend manager singleton."""
    print("="*60)
    print("TEST 2: Backend Manager")
    print("="*60)
    print()
    
    try:
        backend_mgr = get_backend_manager()
        info = backend_mgr.get_backend_info()
        
        print(f"✓ Backend initialized: {info['name']}")
        print(f"  Type: {info['type']}")
        print(f"  Shots: {info['shots']}")
    except Exception as e:
        print(f"✗ Backend initialization failed: {e}")
        return False
    
    print()
    return True


def test_optimization_flow():
    """Test full optimization flow (API logic without HTTP)."""
    print("="*60)
    print("TEST 3: Optimization Flow (Direct Call)")
    print("="*60)
    print()
    
    # Create request
    expected_returns = np.array([0.06, 0.08, 0.12, 0.15])
    covariance_matrix = np.array([
        [0.04, 0.01, 0.02, 0.01],
        [0.01, 0.09, 0.03, 0.02],
        [0.02, 0.03, 0.16, 0.04],
        [0.01, 0.02, 0.04, 0.25]
    ])
    
    print("Running optimization (may take 10-20 seconds)...")
    
    try:
        # Create model
        model = MarkowitzModel(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            risk_aversion=0.5,
            budget=1.0,
            bounds=(0.0, 1.0)
        )
        
        # Classical baseline
        classical_solver = ClassicalSolver(model)
        classical_result = classical_solver.solve(method='SLSQP', verbose=False)
        
        print(f"  Classical: {classical_result['weights']}")
        
        # Quantum optimization
        mapper = IsingMapper(model, num_bins=3, penalty_coefficient=1000.0)
        backend_mgr = get_backend_manager()
        
        vqe = VQEOptimizer(
            mapper=mapper,
            backend_manager=backend_mgr,
            ansatz_type='RealAmplitudes',
            ansatz_reps=2,
            optimizer='COBYLA',
            max_iterations=50
        )
        
        decoder = ResultDecoder(mapper)
        hybrid = HybridVQESolver(vqe, mapper, decoder)
        
        result = hybrid.solve()
        
        print(f"  Hybrid: {result['final_weights']}")
        print(f"  Feasible: {result['is_feasible']}")
        print(f"  Sum: {np.sum(result['final_weights']):.10f}")
        print()
        
        print("✓ Optimization flow complete")
        return True
        
    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all API validation tests."""
    print("\n")
    print("#"*60)
    print("API COMPONENT VALIDATION (Direct Testing)")
    print("#"*60)
    print("\n")
    
    results = {}
    
    results['schemas'] = test_api_schemas()
    results['backend'] = test_backend_manager()
    
    if results['schemas'] and results['backend']:
        results['optimization'] = test_optimization_flow()
    else:
        print("⚠ Skipping optimization test (prerequisites failed)")
        results['optimization'] = False
    
    # Summary
    print()
    print("#"*60)
    print("VALIDATION SUMMARY")
    print("#"*60)
    print()
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name.capitalize()}: {status}")
    
    print()
    
    if all(results.values()):
        print("="*60)
        print("✓ ALL API COMPONENTS VALIDATED")
        print("="*60)
        print()
        print("API is ready to run!")
        print()
        print("Start with:")
        print("  cd /app")
        print("  uvicorn api.main:app --host 0.0.0.0 --port 8002")
        print()
        print("Then access:")
        print("  API: http://localhost:8002/api/")
        print("  Docs: http://localhost:8002/api/docs")
        print("  Health: http://localhost:8002/api/health")
    else:
        print("⚠ SOME COMPONENTS FAILED")
    
    print()


if __name__ == "__main__":
    main()

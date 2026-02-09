"""Test script for FastAPI endpoints.

Tests:
1. Health check endpoint
2. Optimization endpoint with valid data
3. Error handling with invalid data
"""

import sys
sys.path.append('/app')

import requests
import json
import time


BASE_URL = "http://localhost:8001/api"


def test_health_endpoint():
    """Test health check endpoint."""
    print("="*60)
    print("TEST 1: Health Check Endpoint")
    print("="*60)
    print()
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response:")
            print(json.dumps(data, indent=2))
            print()
            
            if data['status'] == 'healthy':
                print("✓ API is healthy")
            else:
                print(f"⚠ API status: {data['status']}")
                print(f"  Components: {data['components']}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
        
        print()
        return response.status_code == 200
        
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API (is it running?)")
        print("  Start with: uvicorn api.main:app --host 0.0.0.0 --port 8001")
        print()
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        return False


def test_optimize_endpoint():
    """Test optimization endpoint with valid data."""
    print("="*60)
    print("TEST 2: Optimization Endpoint (Valid Data)")
    print("="*60)
    print()
    
    # 4-asset portfolio (same as our tests)
    request_data = {
        "expected_returns": [0.06, 0.08, 0.12, 0.15],
        "covariance_matrix": [
            [0.04, 0.01, 0.02, 0.01],
            [0.01, 0.09, 0.03, 0.02],
            [0.02, 0.03, 0.16, 0.04],
            [0.01, 0.02, 0.04, 0.25]
        ],
        "risk_aversion": 0.5,
        "budget": 1.0,
        "num_bins": 3,  # Small for faster test
        "penalty_coefficient": 1000.0,
        "ansatz_reps": 2,
        "max_vqe_iterations": 50  # Fewer iterations for speed
    }
    
    print("Request:")
    print(json.dumps(request_data, indent=2))
    print()
    
    try:
        print("Sending request (this may take ~10-30 seconds)...")
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/optimize",
            json=request_data,
            timeout=120  # 2 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {elapsed:.2f}s")
        print()
        
        if response.status_code == 200:
            data = response.json()
            
            print("✓ Optimization successful!")
            print()
            print(f"Results:")
            print(f"  Weights: {data['weights']}")
            print(f"  Feasible: {data['is_feasible']}")
            print(f"  Weight sum: {sum(data['weights']):.10f}")
            print()
            
            print(f"Portfolio Metrics:")
            print(f"  Return: {data['metrics']['expected_return']:.4f} ({data['metrics']['expected_return']*100:.2f}%)")
            print(f"  Volatility: {data['metrics']['volatility']:.4f} ({data['metrics']['volatility']*100:.2f}%)")
            print(f"  Sharpe: {data['metrics']['sharpe_ratio']:.4f}")
            print()
            
            print(f"VQE Info:")
            print(f"  Iterations: {data['vqe_iterations']}")
            print(f"  Execution time: {data['vqe_execution_time']:.2f}s")
            print(f"  Convergence energy: {data['vqe_convergence_energy']:.2f}")
            print()
            
            print(f"Hybrid Solver:")
            print(f"  VQE weights: {data['vqe_weights']}")
            print(f"  VQE feasible: {data['vqe_was_feasible']}")
            print(f"  Projection distance: {data['projection_distance']:.4f}")
            print()
            
            print(f"Comparison with Classical:")
            print(f"  Classical weights: {data['classical_weights']}")
            print(f"  Hybrid objective: {data['hybrid_objective']:.6f}")
            print(f"  Classical objective: {data['classical_objective']:.6f}")
            print(f"  Approximation ratio: {data['approximation_ratio']:.4f}")
            print()
            
            print(f"Quantum Config:")
            print(f"  Qubits: {data['num_qubits']}")
            print(f"  Bins: {data['num_bins']}")
            print()
            
            # Validation
            weight_sum = sum(data['weights'])
            if abs(weight_sum - 1.0) < 1e-6:
                print("✓ Budget constraint satisfied")
            else:
                print(f"✗ Budget violation: sum = {weight_sum}")
            
            if all(0 <= w <= 1.0 for w in data['weights']):
                print("✓ Bound constraints satisfied")
            else:
                print("✗ Bounds violated")
            
            print()
            return True
            
        else:
            print(f"✗ Optimization failed: {response.status_code}")
            print(f"Response: {response.text}")
            print()
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Request timeout (>120s)")
        print()
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        return False


def test_invalid_data():
    """Test error handling with invalid data."""
    print("="*60)
    print("TEST 3: Error Handling (Invalid Data)")
    print("="*60)
    print()
    
    test_cases = [
        {
            "name": "Invalid covariance (dimension mismatch)",
            "data": {
                "expected_returns": [0.06, 0.08, 0.12],
                "covariance_matrix": [
                    [0.04, 0.01],
                    [0.01, 0.09]
                ],
                "risk_aversion": 0.5
            },
            "expected_status": 400
        },
        {
            "name": "Negative risk aversion",
            "data": {
                "expected_returns": [0.06, 0.08],
                "covariance_matrix": [
                    [0.04, 0.01],
                    [0.01, 0.09]
                ],
                "risk_aversion": -0.5
            },
            "expected_status": 422  # Validation error
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test case {i}: {test['name']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/optimize",
                json=test['data'],
                timeout=10
            )
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == test['expected_status']:
                print(f"  ✓ Correctly returned {test['expected_status']}")
            else:
                print(f"  ✗ Expected {test['expected_status']}, got {response.status_code}")
            
            error_data = response.json()
            if 'detail' in error_data:
                print(f"  Error: {error_data['detail']}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
    
    print("✓ Error handling test complete")
    print()


def test_root_endpoint():
    """Test root endpoint."""
    print("="*60)
    print("TEST 4: Root Endpoint")
    print("="*60)
    print()
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("API Info:")
            print(json.dumps(data, indent=2))
            print()
            print("✓ Root endpoint working")
        else:
            print(f"✗ Root endpoint failed: {response.status_code}")
        
        print()
        return response.status_code == 200
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        return False


def main():
    """Run all API tests."""
    print("\n")
    print("#"*60)
    print("FASTAPI ENDPOINT TESTING")
    print("#"*60)
    print("\n")
    
    print("Note: API must be running on http://localhost:8001")
    print("Start with: uvicorn api.main:app --host 0.0.0.0 --port 8001")
    print()
    input("Press Enter when API is running...")
    print()
    
    # Run tests
    results = {}
    
    results['root'] = test_root_endpoint()
    results['health'] = test_health_endpoint()
    
    if results['health']:
        results['optimize'] = test_optimize_endpoint()
        results['errors'] = test_invalid_data()
    else:
        print("⚠ Skipping optimization tests (API not healthy)")
        results['optimize'] = False
        results['errors'] = False
    
    # Summary
    print("#"*60)
    print("TEST SUMMARY")
    print("#"*60)
    print()
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name.capitalize()}: {status}")
    
    print()
    
    all_passed = all(results.values())
    if all_passed:
        print("="*60)
        print("✓ ALL TESTS PASSED - API IS FULLY OPERATIONAL")
        print("="*60)
    else:
        print("="*60)
        print("⚠ SOME TESTS FAILED")
        print("="*60)
    
    print()
    print("API Documentation: http://localhost:8001/api/docs")
    print("ReDoc: http://localhost:8001/api/redoc")
    print()


if __name__ == "__main__":
    main()

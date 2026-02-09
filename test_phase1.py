"""Test script for Phase 1: Classical Markowitz optimization.

Validates:
1. Data loading and validation
2. Markowitz model definition
3. Classical solver
4. Portfolio metrics computation
"""

import sys
sys.path.append('/app')

import numpy as np
from config.portfolio_config import get_toy_problem_config
from data.loader import PortfolioDataLoader
from data.validator import DataValidator
from finance.markowitz import MarkowitzModel
from benchmarks.classical_solver import ClassicalSolver
from utils.metrics import PortfolioMetrics


def test_phase1():
    """Run Phase 1 validation tests."""
    print("="*60)
    print("PHASE 1: Classical Markowitz Portfolio Optimization")
    print("="*60)
    print()
    
    # Step 1: Load configuration
    print("[1] Loading configuration...")
    config = get_toy_problem_config()
    print(f"    Number of assets: {config.num_assets}")
    print(f"    Risk aversion (λ): {config.risk_aversion}")
    print(f"    Budget: {config.budget}")
    print(f"    Bounds: {config.bounds}")
    print()
    
    # Step 2: Load data
    print("[2] Loading 4-asset example data...")
    expected_returns, covariance_matrix = PortfolioDataLoader.get_4asset_example()
    print(f"    Expected returns: {expected_returns}")
    print(f"    Covariance matrix shape: {covariance_matrix.shape}")
    print(f"    Asset volatilities: {np.sqrt(np.diag(covariance_matrix))}")
    print()
    
    # Step 3: Validate data
    print("[3] Validating data...")
    try:
        DataValidator.validate_portfolio_data(
            expected_returns, 
            covariance_matrix, 
            config.num_assets
        )
        print("    ✓ Data validation passed")
    except ValueError as e:
        print(f"    ✗ Data validation failed: {e}")
        return
    print()
    
    # Step 4: Create Markowitz model
    print("[4] Creating Markowitz model...")
    model = MarkowitzModel(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_aversion=config.risk_aversion,
        budget=config.budget,
        bounds=config.bounds
    )
    print(f"    Model created with {model.num_assets} assets")
    
    # Test initial guess
    initial_weights = model.get_initial_guess()
    print(f"    Initial weights (equal): {initial_weights}")
    obj_init, feasible_init = model.evaluate(initial_weights)
    print(f"    Initial objective: {obj_init:.6f}")
    print(f"    Initial feasibility: {feasible_init}")
    print()
    
    # Step 5: Solve using classical optimizer
    print("[5] Solving with classical optimizer (SLSQP)...")
    solver = ClassicalSolver(model)
    result = solver.solve(method='SLSQP', verbose=False)
    
    print(f"    Success: {result['success']}")
    print(f"    Message: {result['message']}")
    print(f"    Iterations: {result['n_iterations']}")
    print()
    
    if result['success']:
        weights = result['weights']
        print(f"    Optimal weights: {weights}")
        print(f"    Weights sum: {np.sum(weights):.6f}")
        print(f"    Objective value: {result['objective']:.6f}")
        print(f"    Portfolio risk (variance): {result['risk']:.6f}")
        print(f"    Portfolio return: {result['return']:.6f}")
        print()
        
        # Step 6: Compute metrics
        print("[6] Computing portfolio metrics...")
        metrics = PortfolioMetrics.compute_all_metrics(
            weights,
            expected_returns,
            covariance_matrix,
            risk_free_rate=0.02
        )
        
        print(f"    Expected return: {metrics['expected_return']:.4f} ({metrics['expected_return']*100:.2f}%)")
        print(f"    Volatility (σ): {metrics['volatility']:.4f} ({metrics['volatility']*100:.2f}%)")
        print(f"    Variance (σ²): {metrics['variance']:.6f}")
        print(f"    Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        print()
        
        # Step 7: Validate solution
        print("[7] Validating solution...")
        try:
            DataValidator.validate_weights(
                weights,
                config.num_assets,
                config.budget,
                config.bounds
            )
            print("    ✓ Solution validation passed")
        except ValueError as e:
            print(f"    ✗ Solution validation failed: {e}")
        print()
        
        # Step 8: Compare with equal-weight portfolio
        print("[8] Comparing with equal-weight baseline...")
        comparison = PortfolioMetrics.compare_solutions(
            initial_weights,
            weights,
            expected_returns,
            covariance_matrix,
            labels=('Equal-weight', 'Optimized')
        )
        
        print("    Equal-weight portfolio:")
        print(f"      Return: {comparison['Equal-weight']['expected_return']:.4f}")
        print(f"      Volatility: {comparison['Equal-weight']['volatility']:.4f}")
        print(f"      Sharpe: {comparison['Equal-weight']['sharpe_ratio']:.4f}")
        print()
        
        print("    Optimized portfolio:")
        print(f"      Return: {comparison['Optimized']['expected_return']:.4f}")
        print(f"      Volatility: {comparison['Optimized']['volatility']:.4f}")
        print(f"      Sharpe: {comparison['Optimized']['sharpe_ratio']:.4f}")
        print()
        
        print("    Improvement:")
        diff = comparison['differences']
        print(f"      Δ Return: {diff['expected_return']:.4f} ({diff['expected_return']*100:.2f}%)")
        print(f"      Δ Volatility: {diff['volatility']:.4f} ({diff['volatility']*100:.2f}%)")
        print(f"      Δ Sharpe: {diff['sharpe_ratio']:.4f}")
        print()
        
        print("="*60)
        print("✓ PHASE 1 COMPLETE: Classical baseline validated")
        print("="*60)
        
    else:
        print(f"    ✗ Optimization failed: {result['message']}")
        print()
        print("="*60)
        print("✗ PHASE 1 FAILED")
        print("="*60)


if __name__ == "__main__":
    test_phase1()

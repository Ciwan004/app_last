"""Test penalty tuning and adaptive penalty schedule.

Validates:
1. High penalty (P=1000) produces feasible solutions
2. Adaptive penalty converges to feasibility
3. Solution quality comparison across penalties
"""

import sys
sys.path.append('/app')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config.portfolio_config import get_toy_problem_config
from config.quantum_config import get_simulator_config
from data.loader import PortfolioDataLoader
from finance.markowitz import MarkowitzModel
from benchmarks.classical_solver import ClassicalSolver
from quantum.ising_mapper import IsingMapper
from quantum.backend_manager import BackendManager
from quantum.vqe_optimizer import VQEOptimizer
from quantum.result_decoder import ResultDecoder
from quantum.adaptive_penalty import AdaptivePenaltyVQE
from utils.metrics import PortfolioMetrics


def test_fixed_penalty(penalty: float, num_bins: int = 5):
    """Test VQE with fixed penalty coefficient.
    
    Args:
        penalty: Penalty coefficient
        num_bins: Number of bins for discretization
    """
    print("="*60)
    print(f"FIXED PENALTY TEST (P = {penalty})")
    print("="*60)
    print()
    
    # Load problem
    config = get_toy_problem_config()
    returns, cov = PortfolioDataLoader.get_4asset_example()
    
    model = MarkowitzModel(
        expected_returns=returns,
        covariance_matrix=cov,
        risk_aversion=config.risk_aversion,
        budget=config.budget,
        bounds=config.bounds
    )
    
    # Classical solution
    print("Classical baseline:")
    solver = ClassicalSolver(model)
    classical_result = solver.solve(method='SLSQP', verbose=False)
    print(f"  Weights: {classical_result['weights']}")
    print(f"  Objective: {classical_result['objective']:.6f}")
    print()
    
    # Create mapper with specified penalty
    mapper = IsingMapper(model, num_bins=num_bins, penalty_coefficient=penalty)
    print(f"Configuration:")
    print(f"  Bins: {num_bins}")
    print(f"  Qubits: {mapper.encoder.num_qubits}")
    print(f"  Penalty: {penalty}")
    print()
    
    # Create backend
    quantum_config = get_simulator_config()
    backend_mgr = BackendManager(quantum_config)
    
    # Create VQE optimizer
    vqe = VQEOptimizer(
        mapper=mapper,
        backend_manager=backend_mgr,
        ansatz_type='RealAmplitudes',
        ansatz_reps=2,
        optimizer='COBYLA',
        max_iterations=100
    )
    
    # Run VQE
    vqe_result = vqe.optimize()
    
    # Evaluate and decode
    print("Evaluating solution...")
    state_result = vqe.evaluate_final_state(vqe_result['optimal_parameters'])
    decoder = ResultDecoder(mapper)
    decoded = decoder.decode_counts(state_result['counts'])
    
    print(f"VQE Solution:")
    print(f"  Weights: {decoded['best_weights']}")
    print(f"  Energy: {decoded['best_energy']:.6f}")
    print(f"  Feasible: {decoded['is_feasible']}")
    print(f"  Feasible fraction: {decoded['feasible_fraction']:.2%}")
    
    if decoded['best_weights'] is not None:
        weight_sum = np.sum(decoded['best_weights'])
        budget_violation = abs(weight_sum - model.constraints.budget)
        print(f"  Weight sum: {weight_sum:.6f} (target: {model.constraints.budget})")
        print(f"  Budget violation: {budget_violation:.6e}")
        
        # Compute portfolio metrics
        if decoded['is_feasible']:
            vqe_metrics = PortfolioMetrics.compute_all_metrics(
                decoded['best_weights'], returns, cov
            )
            classical_metrics = PortfolioMetrics.compute_all_metrics(
                classical_result['weights'], returns, cov
            )
            
            print()
            print("Portfolio Comparison:")
            print(f"  VQE:")
            print(f"    Return: {vqe_metrics['expected_return']:.4f} ({vqe_metrics['expected_return']*100:.2f}%)")
            print(f"    Volatility: {vqe_metrics['volatility']:.4f} ({vqe_metrics['volatility']*100:.2f}%)")
            print(f"    Sharpe: {vqe_metrics['sharpe_ratio']:.4f}")
            print(f"  Classical:")
            print(f"    Return: {classical_metrics['expected_return']:.4f} ({classical_metrics['expected_return']*100:.2f}%)")
            print(f"    Volatility: {classical_metrics['volatility']:.4f} ({classical_metrics['volatility']*100:.2f}%)")
            print(f"    Sharpe: {classical_metrics['sharpe_ratio']:.4f}")
            
            # Compare with classical
            comparison = decoder.compare_with_classical(
                decoded['best_weights'],
                classical_result['weights']
            )
            print()
            print(f"  Objective gap: {comparison['objective_gap']:.6f}")
            print(f"  Approximation ratio: {comparison['approximation_ratio']:.4f}")
            print(f"  Weight L2 distance: {comparison['weight_l2_distance']:.4f}")
    
    print()
    print("-"*60)
    print()
    
    return {
        'penalty': penalty,
        'decoded': decoded,
        'vqe_result': vqe_result,
        'classical_result': classical_result,
        'feasible': decoded['is_feasible']
    }


def test_penalty_sweep():
    """Test multiple penalty values to find optimal."""
    print("="*60)
    print("PENALTY SWEEP TEST")
    print("="*60)
    print()
    
    penalties = [50, 100, 250, 500, 1000]
    results = []
    
    for penalty in penalties:
        result = test_fixed_penalty(penalty, num_bins=3)  # Use 3 bins for speed
        results.append(result)
    
    # Summary
    print("="*60)
    print("PENALTY SWEEP SUMMARY")
    print("="*60)
    print()
    
    print(f"{'Penalty':<10} {'Feasible':<12} {'Weight Sum':<12} {'Budget Viol.':<15}")
    print("-"*60)
    
    for res in results:
        feasible = "✓" if res['feasible'] else "✗"
        weights = res['decoded']['best_weights']
        if weights is not None:
            weight_sum = np.sum(weights)
            budget_viol = abs(weight_sum - 1.0)
            print(f"{res['penalty']:<10} {feasible:<12} {weight_sum:<12.6f} {budget_viol:<15.6e}")
        else:
            print(f"{res['penalty']:<10} {feasible:<12} {'N/A':<12} {'N/A':<15}")
    
    print()
    
    # Find first feasible
    feasible_results = [r for r in results if r['feasible']]
    if feasible_results:
        min_penalty = min(r['penalty'] for r in feasible_results)
        print(f"✓ Minimum penalty for feasibility: {min_penalty}")
    else:
        print("✗ No feasible solutions found in sweep")
    
    print()
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Feasibility vs Penalty
    plt.subplot(1, 2, 1)
    feasibility_vals = [1 if r['feasible'] else 0 for r in results]
    plt.plot(penalties, feasibility_vals, 'o-', markersize=10, linewidth=2)
    plt.xlabel('Penalty Coefficient')
    plt.ylabel('Feasible (1=Yes, 0=No)')
    plt.title('Feasibility vs Penalty')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Plot 2: Budget Violation vs Penalty
    plt.subplot(1, 2, 2)
    violations = []
    for r in results:
        if r['decoded']['best_weights'] is not None:
            viol = abs(np.sum(r['decoded']['best_weights']) - 1.0)
            violations.append(viol)
        else:
            violations.append(np.nan)
    
    plt.plot(penalties, violations, 's-', markersize=8, linewidth=2, color='red')
    plt.axhline(y=0.01, color='green', linestyle='--', label='Tolerance')
    plt.xlabel('Penalty Coefficient')
    plt.ylabel('Budget Violation')
    plt.title('Budget Violation vs Penalty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('/app/penalty_sweep.png', dpi=100, bbox_inches='tight')
    print("✓ Penalty sweep plot saved: /app/penalty_sweep.png")
    print()
    
    return results


def test_adaptive_penalty():
    """Test adaptive penalty schedule."""
    print("="*60)
    print("ADAPTIVE PENALTY TEST")
    print("="*60)
    print()
    
    # Load problem
    config = get_toy_problem_config()
    returns, cov = PortfolioDataLoader.get_4asset_example()
    
    model = MarkowitzModel(
        expected_returns=returns,
        covariance_matrix=cov,
        risk_aversion=config.risk_aversion,
        budget=config.budget,
        bounds=config.bounds
    )
    
    # Classical solution
    solver = ClassicalSolver(model)
    classical_result = solver.solve(method='SLSQP', verbose=False)
    print(f"Classical baseline:")
    print(f"  Weights: {classical_result['weights']}")
    print(f"  Objective: {classical_result['objective']:.6f}")
    print()
    
    # Create mapper (will be updated with adaptive penalty)
    num_bins = 3  # Use 3 bins for speed
    mapper = IsingMapper(model, num_bins=num_bins, penalty_coefficient=100.0)
    
    # Create backend
    quantum_config = get_simulator_config()
    backend_mgr = BackendManager(quantum_config)
    
    # Create VQE optimizer
    vqe = VQEOptimizer(
        mapper=mapper,
        backend_manager=backend_mgr,
        ansatz_type='RealAmplitudes',
        ansatz_reps=1,  # Shallow for speed
        optimizer='COBYLA',
        max_iterations=50  # Fewer iterations per attempt
    )
    
    # Create adaptive penalty VQE
    adaptive_vqe = AdaptivePenaltyVQE(
        mapper=mapper,
        vqe_optimizer=vqe,
        initial_penalty=100.0,
        penalty_multiplier=2.5,
        max_attempts=5
    )
    
    # Run adaptive optimization
    result = adaptive_vqe.optimize_with_adaptive_penalty()
    
    # Get summary
    summary = adaptive_vqe.get_summary()
    
    print("="*60)
    print("ADAPTIVE PENALTY SUMMARY")
    print("="*60)
    print()
    print(f"Total attempts: {summary['total_attempts']}")
    print(f"Penalties tried: {summary['penalties_tried']}")
    print(f"Feasible solutions: {summary['feasible_count']}")
    
    if summary['first_feasible_attempt']:
        print(f"First feasible at attempt: {summary['first_feasible_attempt']}")
        print(f"Optimal penalty: {summary['optimal_penalty']:.1f}")
    else:
        print("No feasible solution found")
    
    print()
    
    # Compare best solution with classical
    best_sol = result['best_solution']
    if best_sol['best_weights'] is not None and best_sol['is_feasible']:
        print("Best Feasible Solution:")
        print(f"  Weights: {best_sol['best_weights']}")
        
        decoder = ResultDecoder(mapper)
        comparison = decoder.compare_with_classical(
            best_sol['best_weights'],
            classical_result['weights']
        )
        
        print(f"  VQE objective: {comparison['vqe_objective']:.6f}")
        print(f"  Classical objective: {comparison['classical_objective']:.6f}")
        print(f"  Approximation ratio: {comparison['approximation_ratio']:.4f}")
        
        # Portfolio metrics
        vqe_metrics = PortfolioMetrics.compute_all_metrics(
            best_sol['best_weights'], returns, cov
        )
        print()
        print(f"  Return: {vqe_metrics['expected_return']:.4f} ({vqe_metrics['expected_return']*100:.2f}%)")
        print(f"  Volatility: {vqe_metrics['volatility']:.4f}")
        print(f"  Sharpe: {vqe_metrics['sharpe_ratio']:.4f}")
    
    print()
    
    # Plot adaptive progress
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Penalty progression
    plt.subplot(1, 3, 1)
    attempts = list(range(1, len(result['penalty_history']) + 1))
    plt.plot(attempts, result['penalty_history'], 'o-', markersize=10, linewidth=2)
    plt.xlabel('Attempt')
    plt.ylabel('Penalty Coefficient')
    plt.title('Penalty Progression')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Feasibility progression
    plt.subplot(1, 3, 2)
    feasibility_int = [1 if f else 0 for f in result['feasibility_history']]
    plt.plot(attempts, feasibility_int, 'o-', markersize=10, linewidth=2, color='green')
    plt.xlabel('Attempt')
    plt.ylabel('Feasible (1=Yes, 0=No)')
    plt.title('Feasibility Progression')
    plt.ylim([-0.1, 1.1])
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Energy progression
    plt.subplot(1, 3, 3)
    plt.plot(attempts, result['energy_history'], 's-', markersize=8, linewidth=2, color='blue')
    plt.xlabel('Attempt')
    plt.ylabel('QUBO Energy')
    plt.title('Energy Progression')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/app/adaptive_penalty.png', dpi=100, bbox_inches='tight')
    print("✓ Adaptive penalty plot saved: /app/adaptive_penalty.png")
    print()
    
    return result


def main():
    """Run all penalty tuning tests."""
    print("\n")
    print("#"*60)
    print("PENALTY TUNING AND ADAPTIVE SCHEDULE VALIDATION")
    print("#"*60)
    print("\n")
    
    # Test 1: High fixed penalty (P=1000)
    print("TEST 1: High Fixed Penalty (P=1000)")
    high_penalty_result = test_fixed_penalty(penalty=1000, num_bins=3)
    
    # Test 2: Penalty sweep
    print("\nTEST 2: Penalty Sweep")
    sweep_results = test_penalty_sweep()
    
    # Test 3: Adaptive penalty
    print("\nTEST 3: Adaptive Penalty Schedule")
    adaptive_result = test_adaptive_penalty()
    
    # Final summary
    print("#"*60)
    print("FINAL SUMMARY")
    print("#"*60)
    print()
    
    print("✓ High penalty (P=1000) test:", "PASS" if high_penalty_result['feasible'] else "FAIL")
    
    sweep_feasible = any(r['feasible'] for r in sweep_results)
    print("✓ Penalty sweep:", "PASS" if sweep_feasible else "FAIL")
    
    adaptive_feasible = adaptive_result['best_solution']['is_feasible']
    print("✓ Adaptive penalty:", "PASS" if adaptive_feasible else "FAIL")
    
    print()
    print("Key Findings:")
    if high_penalty_result['feasible']:
        print(f"  • P=1000 produces feasible solutions ✓")
    
    feasible_sweep = [r for r in sweep_results if r['feasible']]
    if feasible_sweep:
        min_p = min(r['penalty'] for r in feasible_sweep)
        print(f"  • Minimum feasible penalty: {min_p}")
    
    if adaptive_result['best_solution']['is_feasible']:
        summary = adaptive_result
        first_attempt = None
        for i, f in enumerate(summary['feasibility_history']):
            if f:
                first_attempt = i + 1
                break
        if first_attempt:
            print(f"  • Adaptive schedule converges at attempt {first_attempt}")
    
    print()
    print("="*60)
    print("✓ PENALTY TUNING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

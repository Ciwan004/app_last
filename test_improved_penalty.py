"""Improved penalty tuning test with better parameters.

Key improvements:
1. Use 5 bins (more feasible allocations)
2. More VQE iterations (150-200)
3. Deeper ansatz (reps=2)
4. Test specific penalty values that should work
"""

import sys
sys.path.append('/app')

import numpy as np

from config.portfolio_config import get_toy_problem_config
from config.quantum_config import get_simulator_config
from data.loader import PortfolioDataLoader
from finance.markowitz import MarkowitzModel
from benchmarks.classical_solver import ClassicalSolver
from quantum.ising_mapper import IsingMapper
from quantum.backend_manager import BackendManager
from quantum.vqe_optimizer import VQEOptimizer
from quantum.result_decoder import ResultDecoder
from utils.metrics import PortfolioMetrics


def test_improved_penalty(penalty: float = 1000):
    """Test VQE with improved configuration."""
    print("="*60)
    print(f"IMPROVED VQE TEST (P = {penalty})")
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
    
    # Create mapper with 5 bins (better feasible space)
    num_bins = 5
    mapper = IsingMapper(model, num_bins=num_bins, penalty_coefficient=penalty)
    
    print(f"Configuration:")
    print(f"  Bins: {num_bins} → {mapper.encoder.weight_bins}")
    print(f"  Qubits: {mapper.encoder.num_qubits}")
    print(f"  Penalty: {penalty}")
    print()
    
    # Check how many feasible allocations exist
    from itertools import product
    feasible_count = 0
    for config_tuple in product(mapper.encoder.weight_bins, repeat=4):
        if abs(sum(config_tuple) - 1.0) < 0.01:
            feasible_count += 1
    print(f"  Feasible allocations in discrete space: {feasible_count}")
    print()
    
    # Create backend
    quantum_config = get_simulator_config()
    backend_mgr = BackendManager(quantum_config)
    
    # Create VQE with better configuration
    vqe = VQEOptimizer(
        mapper=mapper,
        backend_manager=backend_mgr,
        ansatz_type='RealAmplitudes',
        ansatz_reps=3,  # Deeper ansatz for better expressivity
        optimizer='COBYLA',
        max_iterations=150  # More iterations
    )
    
    print(f"VQE Configuration:")
    print(f"  Ansatz: RealAmplitudes")
    print(f"  Depth: {vqe.ansatz.depth()}")
    print(f"  Parameters: {vqe.ansatz.num_parameters}")
    print(f"  Max iterations: 150")
    print()
    
    # Run VQE
    vqe_result = vqe.optimize()
    
    # Evaluate and decode
    print("Evaluating solution...")
    state_result = vqe.evaluate_final_state(vqe_result['optimal_parameters'])
    decoder = ResultDecoder(mapper)
    decoded = decoder.decode_counts(state_result['counts'])
    
    print()
    print(f"VQE Solution:")
    print(f"  Weights: {decoded['best_weights']}")
    print(f"  Energy: {decoded['best_energy']:.6f}")
    print(f"  Feasible: {decoded['is_feasible']}")
    print(f"  Feasible fraction: {decoded['feasible_fraction']:.2%}")
    print(f"  Unique states measured: {decoded['num_unique_states']}")
    
    if decoded['best_weights'] is not None:
        weight_sum = np.sum(decoded['best_weights'])
        budget_violation = abs(weight_sum - model.constraints.budget)
        print(f"  Weight sum: {weight_sum:.6f} (target: {model.constraints.budget})")
        print(f"  Budget violation: {budget_violation:.6e}")
        
        # Check one-hot violations
        best_sol = decoded['best_solution']
        if best_sol:
            print(f"  One-hot violations: {best_sol['onehot_violations']}")
        
        # If feasible, compute metrics
        if decoded['is_feasible']:
            print()
            print("✓ FEASIBLE SOLUTION FOUND!")
            print()
            
            vqe_metrics = PortfolioMetrics.compute_all_metrics(
                decoded['best_weights'], returns, cov
            )
            classical_metrics = PortfolioMetrics.compute_all_metrics(
                classical_result['weights'], returns, cov
            )
            
            print("Portfolio Comparison:")
            print(f"  VQE:")
            print(f"    Weights: {decoded['best_weights']}")
            print(f"    Return: {vqe_metrics['expected_return']:.4f} ({vqe_metrics['expected_return']*100:.2f}%)")
            print(f"    Volatility: {vqe_metrics['volatility']:.4f} ({vqe_metrics['volatility']*100:.2f}%)")
            print(f"    Sharpe: {vqe_metrics['sharpe_ratio']:.4f}")
            print()
            print(f"  Classical:")
            print(f"    Weights: {classical_result['weights']}")
            print(f"    Return: {classical_metrics['expected_return']:.4f} ({classical_metrics['expected_return']*100:.2f}%)")
            print(f"    Volatility: {classical_metrics['volatility']:.4f} ({classical_metrics['volatility']*100:.2f}%)")
            print(f"    Sharpe: {classical_metrics['sharpe_ratio']:.4f}")
            print()
            
            # Comparison
            comparison = decoder.compare_with_classical(
                decoded['best_weights'],
                classical_result['weights']
            )
            
            print(f"Comparison Metrics:")
            print(f"  Objective gap: {comparison['objective_gap']:.6f}")
            print(f"  Approximation ratio: {comparison['approximation_ratio']:.4f}")
            print(f"  Weight L2 distance: {comparison['weight_l2_distance']:.4f}")
            print()
            
            # Assess quality
            if comparison['approximation_ratio'] < 1.5:
                print("  Quality: EXCELLENT (within 50% of classical)")
            elif comparison['approximation_ratio'] < 2.0:
                print("  Quality: GOOD (within 2x of classical)")
            else:
                print("  Quality: ACCEPTABLE (>2x of classical)")
    
    print()
    print("="*60)
    
    return {
        'penalty': penalty,
        'decoded': decoded,
        'feasible': decoded['is_feasible'],
        'vqe_result': vqe_result,
        'classical_result': classical_result
    }


def main():
    """Run improved penalty tuning test."""
    print("\n")
    print("#"*60)
    print("IMPROVED PENALTY TUNING WITH OPTIMAL CONFIGURATION")
    print("#"*60)
    print("\n")
    
    # Test with penalty = 1000
    result = test_improved_penalty(penalty=1000)
    
    print()
    print("#"*60)
    print("SUMMARY")
    print("#"*60)
    print()
    
    if result['feasible']:
        print("✓ SUCCESS: Feasible solution found with P=1000")
        print()
        print("Key Improvements:")
        print("  • Used 5 bins (0, 0.25, 0.5, 0.75, 1.0)")
        print("  • Increased VQE iterations to 150")
        print("  • Deeper ansatz (reps=3) for better expressivity")
        print("  • High penalty (P=1000) for constraint enforcement")
    else:
        print("✗ FAIL: Solution still infeasible")
        print()
        print("Possible causes:")
        print("  • VQE converged to local minimum")
        print("  • Penalty still insufficient (try P=2000-5000)")
        print("  • Ansatz not expressive enough (try reps=4-5)")
        print("  • More iterations needed (try 200-300)")
        print()
        print("Recommendations:")
        print("  • Run multiple times with different random seeds")
        print("  • Use SPSA optimizer for better exploration")
        print("  • Consider hybrid classical-quantum post-processing")


if __name__ == "__main__":
    main()

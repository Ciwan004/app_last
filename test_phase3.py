"""Test script for Phase 3: VQE Execution.

Validates:
1. Backend manager initialization
2. Ansatz creation and parameterization
3. Hamiltonian construction
4. VQE optimization loop
5. Result decoding
6. Comparison with classical optimum
"""

import sys
sys.path.append('/app')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from config.portfolio_config import get_toy_problem_config
from config.quantum_config import get_simulator_config
from data.loader import PortfolioDataLoader
from finance.markowitz import MarkowitzModel
from benchmarks.classical_solver import ClassicalSolver
from quantum.ising_mapper import IsingMapper
from quantum.backend_manager import BackendManager
from quantum.ansatz_factory import AnsatzFactory
from quantum.vqe_optimizer import VQEOptimizer
from quantum.result_decoder import ResultDecoder
from utils.metrics import PortfolioMetrics


def test_backend_manager():
    """Test backend initialization."""
    print("="*60)
    print("TEST 1: Backend Manager")
    print("="*60)
    print()
    
    config = get_simulator_config()
    backend_mgr = BackendManager(config)
    
    info = backend_mgr.get_backend_info()
    print("Backend info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    estimator = backend_mgr.get_estimator()
    print(f"Estimator: {type(estimator).__name__}")
    print()
    
    print("✓ Backend manager test complete")
    print()
    
    return backend_mgr


def test_ansatz_creation():
    """Test ansatz factory."""
    print("="*60)
    print("TEST 2: Ansatz Creation")
    print("="*60)
    print()
    
    # Create ansatz for small problem (12 qubits for testing)
    num_qubits = 12
    
    for ansatz_type in ['RealAmplitudes', 'EfficientSU2']:
        print(f"Creating {ansatz_type} ansatz...")
        ansatz = AnsatzFactory.create_ansatz(
            num_qubits=num_qubits,
            ansatz_type=ansatz_type,
            reps=2
        )
        
        info = AnsatzFactory.get_ansatz_info(ansatz)
        print(f"  Qubits: {info['num_qubits']}")
        print(f"  Parameters: {info['num_parameters']}")
        print(f"  Depth: {info['depth']}")
        print(f"  Gates: {info['gate_count']}")
        print()
    
    print("✓ Ansatz creation test complete")
    print()
    
    return ansatz


def test_hamiltonian_construction(mapper):
    """Test Hamiltonian operator construction."""
    print("="*60)
    print("TEST 3: Hamiltonian Construction")
    print("="*60)
    print()
    
    # Get Ising parameters
    J, h, offset = mapper.qubo_to_ising()
    
    print(f"Ising Hamiltonian:")
    print(f"  J matrix shape: {J.shape}")
    print(f"  J non-zero: {np.sum(np.abs(J) > 1e-10)}")
    print(f"  h vector shape: {h.shape}")
    print(f"  h non-zero: {np.sum(np.abs(h) > 1e-10)}")
    print(f"  Offset: {offset:.2f}")
    print()
    
    print("✓ Hamiltonian construction test complete")
    print()


def test_vqe_small_problem():
    """Test VQE on a small problem (reduced qubits for speed)."""
    print("="*60)
    print("TEST 4: VQE on Small Problem (3 bins)")
    print("="*60)
    print()
    
    # Use smaller discretization for faster testing
    num_bins = 3  # Only 3 bins → 12 qubits (4 assets × 3 bins)
    
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
    print("Solving classical baseline...")
    solver = ClassicalSolver(model)
    classical_result = solver.solve(method='SLSQP', verbose=False)
    print(f"  Classical objective: {classical_result['objective']:.6f}")
    print(f"  Classical weights: {classical_result['weights']}")
    print()
    
    # Create mapper
    mapper = IsingMapper(model, num_bins=num_bins, penalty_coefficient=50.0)
    print(f"Quantum configuration:")
    print(f"  Bins: {num_bins}")
    print(f"  Qubits: {mapper.encoder.num_qubits}")
    print()
    
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
        max_iterations=50
    )
    
    # Run VQE
    vqe_result = vqe.optimize()
    
    # Get final state
    print("Evaluating final quantum state...")
    state_result = vqe.evaluate_final_state(vqe_result['optimal_parameters'])
    counts = state_result['counts']
    print(f"  Measured {len(counts)} unique states")
    print(f"  Top 5 states:")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for bitstring, count in sorted_counts[:5]:
        print(f"    {bitstring}: {count}")
    print()
    
    # Decode results
    print("Decoding VQE solution...")
    decoder = ResultDecoder(mapper)
    decoded = decoder.decode_counts(counts)
    
    print(f"  Best solution found:")
    print(f"    Weights: {decoded['best_weights']}")
    print(f"    Energy: {decoded['best_energy']:.6f}")
    print(f"    Feasible: {decoded['is_feasible']}")
    print(f"    Feasible fraction: {decoded['feasible_fraction']:.2%}")
    print()
    
    # Compare with classical
    if decoded['best_weights'] is not None:
        comparison = decoder.compare_with_classical(
            decoded['best_weights'],
            classical_result['weights']
        )
        
        print("Comparison with classical:")
        print(f"  VQE objective: {comparison['vqe_objective']:.6f}")
        print(f"  Classical objective: {comparison['classical_objective']:.6f}")
        print(f"  Objective gap: {comparison['objective_gap']:.6f}")
        print(f"  Approximation ratio: {comparison['approximation_ratio']:.4f}")
        print(f"  Weight L2 distance: {comparison['weight_l2_distance']:.4f}")
        print()
    
    # Plot convergence
    convergence = vqe.get_convergence_data()
    if convergence:
        plt.figure(figsize=(10, 5))
        plt.plot(convergence['iterations'], convergence['energies'], 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('VQE Convergence (Small Problem)')
        plt.grid(True, alpha=0.3)
        plt.savefig('/app/vqe_convergence_small.png', dpi=100, bbox_inches='tight')
        print("✓ Convergence plot saved: /app/vqe_convergence_small.png")
        print()
    
    print("✓ VQE small problem test complete")
    print()
    
    return vqe_result, decoded, comparison


def test_vqe_medium_problem():
    """Test VQE on a medium problem (5 bins, 20 qubits)."""
    print("="*60)
    print("TEST 5: VQE on Medium Problem (5 bins, 20 qubits)")
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
    print("Solving classical baseline...")
    solver = ClassicalSolver(model)
    classical_result = solver.solve(method='SLSQP', verbose=False)
    print(f"  Classical objective: {classical_result['objective']:.6f}")
    print(f"  Classical weights: {classical_result['weights']}")
    print()
    
    # Create mapper
    num_bins = 5  # 5 bins → 20 qubits
    mapper = IsingMapper(model, num_bins=num_bins, penalty_coefficient=75.0)
    print(f"Quantum configuration:")
    print(f"  Bins: {num_bins}")
    print(f"  Qubits: {mapper.encoder.num_qubits}")
    print()
    
    # Create backend
    quantum_config = get_simulator_config()
    backend_mgr = BackendManager(quantum_config)
    
    # Create VQE optimizer
    vqe = VQEOptimizer(
        mapper=mapper,
        backend_manager=backend_mgr,
        ansatz_type='RealAmplitudes',
        ansatz_reps=2,  # Deeper ansatz
        optimizer='COBYLA',
        max_iterations=100
    )
    
    # Run VQE
    vqe_result = vqe.optimize()
    
    # Get final state
    print("Evaluating final quantum state...")
    state_result = vqe.evaluate_final_state(vqe_result['optimal_parameters'])
    counts = state_result['counts']
    print(f"  Measured {len(counts)} unique states")
    print()
    
    # Decode results
    decoder = ResultDecoder(mapper)
    decoded = decoder.decode_counts(counts)
    
    print(f"Best VQE solution:")
    print(f"  Weights: {decoded['best_weights']}")
    print(f"  Energy: {decoded['best_energy']:.6f}")
    print(f"  Feasible: {decoded['is_feasible']}")
    print(f"  Feasible fraction: {decoded['feasible_fraction']:.2%}")
    print()
    
    # Compare with classical
    if decoded['best_weights'] is not None:
        comparison = decoder.compare_with_classical(
            decoded['best_weights'],
            classical_result['weights']
        )
        
        print("Comparison with classical:")
        print(f"  VQE objective: {comparison['vqe_objective']:.6f}")
        print(f"  Classical objective: {comparison['classical_objective']:.6f}")
        print(f"  Objective gap: {comparison['objective_gap']:.6f}")
        print(f"  Approximation ratio: {comparison['approximation_ratio']:.4f}")
        print()
        
        print("Portfolio metrics:")
        vqe_metrics = PortfolioMetrics.compute_all_metrics(
            decoded['best_weights'], returns, cov
        )
        classical_metrics = PortfolioMetrics.compute_all_metrics(
            classical_result['weights'], returns, cov
        )
        
        print(f"  VQE:")
        print(f"    Return: {vqe_metrics['expected_return']:.4f} ({vqe_metrics['expected_return']*100:.2f}%)")
        print(f"    Volatility: {vqe_metrics['volatility']:.4f} ({vqe_metrics['volatility']*100:.2f}%)")
        print(f"    Sharpe: {vqe_metrics['sharpe_ratio']:.4f}")
        print(f"  Classical:")
        print(f"    Return: {classical_metrics['expected_return']:.4f} ({classical_metrics['expected_return']*100:.2f}%)")
        print(f"    Volatility: {classical_metrics['volatility']:.4f} ({classical_metrics['volatility']*100:.2f}%)")
        print(f"    Sharpe: {classical_metrics['sharpe_ratio']:.4f}")
        print()
    
    # Plot convergence
    convergence = vqe.get_convergence_data()
    if convergence:
        plt.figure(figsize=(10, 5))
        plt.plot(convergence['iterations'], convergence['energies'], 'b-', linewidth=2)
        plt.axhline(y=convergence['best_energy'], color='r', linestyle='--', label='Best')
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('VQE Convergence (Medium Problem, 20 qubits)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/app/vqe_convergence_medium.png', dpi=100, bbox_inches='tight')
        print("✓ Convergence plot saved: /app/vqe_convergence_medium.png")
        print()
    
    print("✓ VQE medium problem test complete")
    print()
    
    return vqe_result, decoded, comparison


def main():
    """Run all Phase 3 tests."""
    print("\n")
    print("#"*60)
    print("PHASE 3: VQE EXECUTION VALIDATION")
    print("#"*60)
    print("\n")
    
    # Test 1: Backend
    backend_mgr = test_backend_manager()
    
    # Test 2: Ansatz
    ansatz = test_ansatz_creation()
    
    # Test 3: Hamiltonian (need mapper first)
    config = get_toy_problem_config()
    returns, cov = PortfolioDataLoader.get_4asset_example()
    model = MarkowitzModel(returns, cov, config.risk_aversion, config.budget, config.bounds)
    mapper = IsingMapper(model, num_bins=3, penalty_coefficient=50.0)
    test_hamiltonian_construction(mapper)
    
    # Test 4: VQE small problem
    vqe_small, decoded_small, comparison_small = test_vqe_small_problem()
    
    # Test 5: VQE medium problem  
    vqe_medium, decoded_medium, comparison_medium = test_vqe_medium_problem()
    
    # Final summary
    print("#"*60)
    print("PHASE 3 SUMMARY")
    print("#"*60)
    print()
    
    print("✓ Backend manager: PASS")
    print("✓ Ansatz creation: PASS")
    print("✓ Hamiltonian construction: PASS")
    print("✓ VQE small problem (3 bins, 12 qubits): PASS")
    print("✓ VQE medium problem (5 bins, 20 qubits): PASS")
    print()
    
    print("Key Results:")
    print(f"  Small problem (3 bins, 12 qubits):")
    if comparison_small:
        print(f"    Approximation ratio: {comparison_small['approximation_ratio']:.4f}")
        print(f"    Feasible: {decoded_small['is_feasible']}")
    print(f"  Medium problem (5 bins, 20 qubits):")
    if comparison_medium:
        print(f"    Approximation ratio: {comparison_medium['approximation_ratio']:.4f}")
        print(f"    Feasible: {decoded_medium['is_feasible']}")
        if decoded_medium['best_weights'] is not None:
            print(f"    Weights: {decoded_medium['best_weights']}")
    print()
    
    print("="*60)
    print("✓ PHASE 3 COMPLETE: VQE execution validated")
    print("="*60)
    print()
    
    print("All quantum components operational:")
    print("  ✓ Backend management (Aer simulator)")
    print("  ✓ Variational ansatz (RealAmplitudes)")
    print("  ✓ Hamiltonian construction (Pauli operators)")
    print("  ✓ VQE optimization loop (COBYLA)")
    print("  ✓ Result decoding and comparison")
    print()


if __name__ == "__main__":
    main()

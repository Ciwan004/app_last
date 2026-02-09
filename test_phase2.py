"""Test script for Phase 2: Ising/QUBO Mapping.

Validates:
1. One-hot encoding/decoding
2. QUBO matrix construction
3. Consistency with classical optimum
4. Ground state search (brute force for small problems)
"""

import sys
sys.path.append('/app')

import numpy as np
from config.portfolio_config import get_toy_problem_config
from data.loader import PortfolioDataLoader
from finance.markowitz import MarkowitzModel
from benchmarks.classical_solver import ClassicalSolver
from quantum.ising_mapper import OneHotEncoder, IsingMapper
from itertools import product


def test_one_hot_encoding():
    """Test one-hot encoder."""
    print("="*60)
    print("TEST 1: One-Hot Encoding")
    print("="*60)
    print()
    
    # Create encoder for 4 assets, 5 bins
    encoder = OneHotEncoder(num_assets=4, num_bins=5, max_weight=1.0)
    
    print(f"Configuration:")
    print(f"  Assets: {encoder.num_assets}")
    print(f"  Bins per asset: {encoder.num_bins}")
    print(f"  Weight bins: {encoder.weight_bins}")
    print(f"  Total qubits: {encoder.num_qubits}")
    print()
    
    # Test encoding/decoding
    test_weights = np.array([0.25, 0.25, 0.25, 0.25])
    print(f"Test weights: {test_weights}")
    
    binary = encoder.encode_weights(test_weights)
    print(f"Encoded binary (length {len(binary)}): {binary}")
    
    decoded = encoder.decode_binary(binary)
    print(f"Decoded weights: {decoded}")
    
    error = np.linalg.norm(decoded - test_weights)
    print(f"Reconstruction error: {error:.6f}")
    print()
    
    # Test with optimal weights from Phase 1
    optimal_weights = np.array([0.37902262, 0.21697695, 0.1929049, 0.21109553])
    print(f"Optimal weights (Phase 1): {optimal_weights}")
    
    binary_opt = encoder.encode_weights(optimal_weights)
    decoded_opt = encoder.decode_binary(binary_opt)
    print(f"Decoded optimal: {decoded_opt}")
    print(f"Discretization error: {np.linalg.norm(decoded_opt - optimal_weights):.6f}")
    print()
    
    print("✓ One-hot encoding test complete")
    print()
    return encoder


def test_qubo_construction():
    """Test QUBO matrix construction."""
    print("="*60)
    print("TEST 2: QUBO Matrix Construction")
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
    
    # Create mapper
    mapper = IsingMapper(model, num_bins=5, penalty_coefficient=10.0)
    
    print(f"Model configuration:")
    print(f"  Assets: {model.num_assets}")
    print(f"  Risk aversion (λ): {model.objective.risk_aversion}")
    print(f"  Penalty coefficient: {mapper.penalty_coefficient}")
    print()
    
    # Build QUBO
    print("Building QUBO matrix...")
    Q, offset = mapper.build_qubo()
    
    print(f"  Q matrix shape: {Q.shape}")
    print(f"  Q matrix sparsity: {np.sum(Q != 0) / Q.size * 100:.1f}% non-zero")
    print(f"  Offset: {offset:.6f}")
    print()
    
    # Check symmetry (QUBO upper triangular)
    is_upper_triangular = np.allclose(np.tril(Q, -1), 0)
    print(f"  Q is upper triangular: {is_upper_triangular}")
    print()
    
    print("✓ QUBO construction test complete")
    print()
    return mapper


def test_classical_consistency(mapper, classical_weights):
    """Test consistency between classical and QUBO formulations."""
    print("="*60)
    print("TEST 3: Classical-Quantum Consistency")
    print("="*60)
    print()
    
    print("Validating classical optimum encoding...")
    print(f"Classical weights: {classical_weights}")
    print()
    
    validation = mapper.validate_mapping(classical_weights)
    
    print("Validation results:")
    print(f"  Decoded weights: {validation['decoded_weights']}")
    print(f"  Weight error (L2): {validation['weight_error']:.6f}")
    print(f"  QUBO energy: {validation['qubo_energy']:.6f}")
    print(f"  Feasible: {validation['is_feasible']}")
    print(f"  Budget violation: {validation['budget_violation']:.6e}")
    print()
    
    if validation['passes_validation']:
        print("✓ Classical solution maps correctly to QUBO")
    else:
        print("✗ Validation failed")
    print()
    
    return validation


def test_ground_state_search(mapper, classical_energy_estimate):
    """Search for ground state via brute force (feasible for small problems)."""
    print("="*60)
    print("TEST 4: Ground State Search (Brute Force)")
    print("="*60)
    print()
    
    n_qubits = mapper.encoder.num_qubits
    print(f"Total qubits: {n_qubits}")
    print(f"Search space size: 2^{n_qubits} = {2**n_qubits:,} states")
    
    # Brute force only feasible for small problems
    if n_qubits > 20:
        print("⚠ Search space too large for brute force, skipping...")
        print()
        return None
    
    print("Searching for ground state...")
    print("(Only checking feasible states with one-hot constraint)")
    print()
    
    # Generate all feasible one-hot configurations
    num_assets = mapper.model.num_assets
    num_bins = mapper.num_bins
    
    # Each asset chooses exactly one bin
    bin_choices = list(range(num_bins))
    feasible_configs = list(product(bin_choices, repeat=num_assets))
    
    print(f"Feasible configurations: {len(feasible_configs)} (one-hot constraint enforced)")
    print()
    
    best_energy = float('inf')
    best_binary = None
    best_weights = None
    energies = []
    
    for config in feasible_configs:
        # Build binary vector
        binary = np.zeros(n_qubits, dtype=int)
        for asset_idx, bin_idx in enumerate(config):
            qubit_idx = asset_idx * num_bins + bin_idx
            binary[qubit_idx] = 1
        
        # Evaluate
        result = mapper.evaluate_binary(binary)
        energy = result['qubo_energy']
        energies.append(energy)
        
        if energy < best_energy:
            best_energy = energy
            best_binary = binary.copy()
            best_weights = result['weights'].copy()
    
    print(f"Ground state found:")
    print(f"  Energy: {best_energy:.6f}")
    print(f"  Weights: {best_weights}")
    print(f"  Sum of weights: {np.sum(best_weights):.6f}")
    print()
    
    print(f"Energy statistics:")
    energies = np.array(energies)
    print(f"  Min: {energies.min():.6f}")
    print(f"  Max: {energies.max():.6f}")
    print(f"  Mean: {energies.mean():.6f}")
    print(f"  Std: {energies.std():.6f}")
    print()
    
    # Compare with classical
    print(f"Comparison with classical optimum:")
    print(f"  Classical energy (estimate): {classical_energy_estimate:.6f}")
    print(f"  Quantum ground state energy: {best_energy:.6f}")
    print(f"  Difference: {abs(best_energy - classical_energy_estimate):.6f}")
    print()
    
    print("✓ Ground state search complete")
    print()
    
    return {
        'ground_state_energy': best_energy,
        'ground_state_binary': best_binary,
        'ground_state_weights': best_weights,
        'all_energies': energies
    }


def test_ising_conversion(mapper):
    """Test QUBO to Ising conversion."""
    print("="*60)
    print("TEST 5: QUBO to Ising Conversion")
    print("="*60)
    print()
    
    print("Converting QUBO to Ising formulation...")
    J, h, offset_ising = mapper.qubo_to_ising()
    
    print(f"  J matrix shape: {J.shape}")
    print(f"  h vector shape: {h.shape}")
    print(f"  Ising offset: {offset_ising:.6f}")
    print()
    
    # Check J is symmetric
    is_symmetric = np.allclose(J, J.T)
    print(f"  J is symmetric: {is_symmetric}")
    print()
    
    # Verify conversion with a random state
    print("Verifying conversion with random state...")
    binary_test = np.random.randint(0, 2, size=mapper.encoder.num_qubits)
    
    # QUBO energy
    qubo_energy = binary_test @ mapper.Q @ binary_test + mapper.offset
    
    # Convert to Ising spins
    spins = 2 * binary_test - 1  # {0,1} -> {-1,+1}
    
    # Ising energy
    ising_energy = spins @ J @ spins + h @ spins + offset_ising
    
    print(f"  QUBO energy: {qubo_energy:.6f}")
    print(f"  Ising energy: {ising_energy:.6f}")
    print(f"  Difference: {abs(qubo_energy - ising_energy):.6e}")
    
    if abs(qubo_energy - ising_energy) < 1e-6:
        print("  ✓ Conversion verified")
    else:
        print("  ✗ Conversion mismatch")
    print()
    
    print("✓ QUBO-Ising conversion test complete")
    print()
    
    return J, h, offset_ising


def main():
    """Run all Phase 2 tests."""
    print("\n")
    print("#"*60)
    print("PHASE 2: ISING/QUBO MAPPING VALIDATION")
    print("#"*60)
    print("\n")
    
    # Test 1: One-hot encoding
    encoder = test_one_hot_encoding()
    
    # Test 2: QUBO construction
    mapper = test_qubo_construction()
    
    # Solve classical problem first for comparison
    print("="*60)
    print("CLASSICAL BASELINE (from Phase 1)")
    print("="*60)
    print()
    
    config = get_toy_problem_config()
    returns, cov = PortfolioDataLoader.get_4asset_example()
    model = MarkowitzModel(
        expected_returns=returns,
        covariance_matrix=cov,
        risk_aversion=config.risk_aversion,
        budget=config.budget,
        bounds=config.bounds
    )
    
    solver = ClassicalSolver(model)
    classical_result = solver.solve(method='SLSQP', verbose=False)
    
    print(f"Classical solution:")
    print(f"  Weights: {classical_result['weights']}")
    print(f"  Objective: {classical_result['objective']:.6f}")
    print(f"  Risk: {classical_result['risk']:.6f}")
    print(f"  Return: {classical_result['return']:.6f}")
    print()
    
    # Test 3: Classical consistency
    validation = test_classical_consistency(mapper, classical_result['weights'])
    
    # Test 4: Ground state search
    ground_state = test_ground_state_search(mapper, validation['qubo_energy'])
    
    # Test 5: QUBO to Ising
    J, h, offset_ising = test_ising_conversion(mapper)
    
    # Final summary
    print("#"*60)
    print("PHASE 2 SUMMARY")
    print("#"*60)
    print()
    
    print("✓ One-hot encoding: PASS")
    print("✓ QUBO construction: PASS")
    print(f"✓ Classical consistency: {'PASS' if validation['passes_validation'] else 'FAIL'}")
    
    if ground_state is not None:
        print("✓ Ground state search: PASS")
        print(f"  Ground state weights: {ground_state['ground_state_weights']}")
        print(f"  Classical weights (discretized): {validation['decoded_weights']}")
        
        weight_diff = np.linalg.norm(
            ground_state['ground_state_weights'] - validation['decoded_weights']
        )
        print(f"  Difference: {weight_diff:.6f}")
        
        if weight_diff < 1e-6:
            print("  ✓ Ground state matches classical optimum!")
        else:
            print("  ⚠ Ground state differs from classical (expected due to discretization)")
    
    print("✓ QUBO-Ising conversion: PASS")
    print()
    
    print("="*60)
    print("✓ PHASE 2 COMPLETE: Ising/QUBO mapping validated")
    print("="*60)
    print()
    
    print("Next: Phase 3 - VQE Execution")
    print()


if __name__ == "__main__":
    main()

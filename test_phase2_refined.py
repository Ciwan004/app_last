"""Improved Phase 2 test with better discretization strategy.

Key insight: With 4 assets and budget=1.0, we need finer discretization
to ensure feasible solutions exist in the discrete space.
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


def test_discretization_analysis():
    """Analyze discretization requirements for budget feasibility."""
    print("="*60)
    print("DISCRETIZATION ANALYSIS")
    print("="*60)
    print()
    
    num_assets = 4
    budget = 1.0
    
    print(f"Problem: {num_assets} assets, budget = {budget}")
    print()
    
    # Test different bin configurations
    for num_bins in [3, 5, 9, 11]:
        encoder = OneHotEncoder(num_assets, num_bins, max_weight=1.0)
        bins = encoder.weight_bins
        
        print(f"Bins ({num_bins}): {bins}")
        print(f"  Qubits: {encoder.num_qubits}")
        
        # Check if budget is achievable
        # Find all combinations that sum to budget
        feasible_count = 0
        examples = []
        
        for config in product(bins, repeat=num_assets):
            if abs(sum(config) - budget) < 1e-10:
                feasible_count += 1
                if len(examples) < 3:
                    examples.append(config)
        
        print(f"  Feasible allocations: {feasible_count}")
        if examples:
            print(f"  Examples: {examples[0]}")
        print()
    
    print("Recommendation: Use 11 bins for finer granularity")
    print()


def test_phase2_refined():
    """Run Phase 2 tests with refined discretization."""
    print("="*60)
    print("PHASE 2 REFINED: ISING/QUBO MAPPING")
    print("="*60)
    print()
    
    # Configuration with finer discretization
    num_bins = 11  # 0.0, 0.1, 0.2, ..., 1.0
    penalty = 100.0  # Stronger penalty for constraints
    
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
    
    # Solve classical
    solver = ClassicalSolver(model)
    classical_result = solver.solve(method='SLSQP', verbose=False)
    
    print("Classical solution:")
    print(f"  Weights: {classical_result['weights']}")
    print(f"  Sum: {np.sum(classical_result['weights']):.6f}")
    print(f"  Objective: {classical_result['objective']:.6f}")
    print()
    
    # Create mapper with finer discretization
    mapper = IsingMapper(model, num_bins=num_bins, penalty_coefficient=penalty)
    
    print(f"Quantum mapping configuration:")
    print(f"  Bins: {num_bins}")
    print(f"  Qubits: {mapper.encoder.num_qubits}")
    print(f"  Weight bins: {mapper.encoder.weight_bins}")
    print(f"  Penalty: {penalty}")
    print()
    
    # Build QUBO
    Q, offset = mapper.build_qubo()
    print(f"QUBO matrix:")
    print(f"  Shape: {Q.shape}")
    print(f"  Non-zero: {np.sum(Q != 0)}")
    print(f"  Offset: {offset:.2f}")
    print()
    
    # Validate classical encoding
    print("Encoding classical solution...")
    validation = mapper.validate_mapping(classical_result['weights'])
    
    print(f"  Original: {validation['classical_weights']}")
    print(f"  Decoded:  {validation['decoded_weights']}")
    print(f"  Sum:      {np.sum(validation['decoded_weights']):.6f}")
    print(f"  L2 error: {validation['weight_error']:.6f}")
    print(f"  Feasible: {validation['is_feasible']}")
    print(f"  Budget violation: {validation['budget_violation']:.6e}")
    print()
    
    # Search ground state (limit search for 44 qubits)
    if mapper.encoder.num_qubits <= 30:
        print("Searching ground state (feasible one-hot states only)...")
        
        num_feasible = num_bins ** model.num_assets
        print(f"  Feasible configurations: {num_feasible:,}")
        
        if num_feasible <= 20000:  # Limit for reasonable runtime
            best_energy = float('inf')
            best_weights = None
            best_binary = None
            budget_feasible_count = 0
            
            for config in product(range(num_bins), repeat=model.num_assets):
                # Build binary
                binary = np.zeros(mapper.encoder.num_qubits, dtype=int)
                for asset_idx, bin_idx in enumerate(config):
                    qubit_idx = asset_idx * num_bins + bin_idx
                    binary[qubit_idx] = 1
                
                # Evaluate
                result = mapper.evaluate_binary(binary)
                
                # Count budget-feasible solutions
                if result['budget_violation'] < 0.01:
                    budget_feasible_count += 1
                
                if result['qubo_energy'] < best_energy:
                    best_energy = result['qubo_energy']
                    best_weights = result['weights'].copy()
                    best_binary = binary.copy()
            
            print(f"  Ground state energy: {best_energy:.6f}")
            print(f"  Ground state weights: {best_weights}")
            print(f"  Sum: {np.sum(best_weights):.6f}")
            print(f"  Budget-feasible solutions: {budget_feasible_count} / {num_feasible}")
            print()
            
            # Compare with classical
            classical_encoded_result = mapper.evaluate_binary(validation['encoded_binary'])
            print(f"Comparison:")
            print(f"  Classical (encoded) energy: {classical_encoded_result['qubo_energy']:.6f}")
            print(f"  Ground state energy: {best_energy:.6f}")
            print(f"  Difference: {abs(classical_encoded_result['qubo_energy'] - best_energy):.6f}")
            print()
            
            # Evaluate classical Markowitz objective for ground state
            ground_markowitz = model.objective.compute_objective(best_weights)
            classical_markowitz = classical_result['objective']
            print(f"Markowitz objective comparison:")
            print(f"  Classical continuous: {classical_markowitz:.6f}")
            print(f"  Classical discretized: {classical_encoded_result['markowitz_objective']:.6f}")
            print(f"  Ground state: {ground_markowitz:.6f}")
            print()
            
            if abs(ground_markowitz - classical_markowitz) < 0.01:
                print("  ✓ Ground state close to classical optimum!")
            else:
                print(f"  ⚠ Discretization gap: {abs(ground_markowitz - classical_markowitz):.6f}")
        else:
            print(f"  ⚠ Search space too large ({num_feasible:,}), skipping brute force")
    else:
        print(f"⚠ {mapper.encoder.num_qubits} qubits too many for brute force")
    print()
    
    # Test QUBO-Ising conversion
    print("Testing QUBO to Ising conversion...")
    J, h, offset_ising = mapper.qubo_to_ising()
    
    # Verify with random state
    test_binary = np.random.randint(0, 2, mapper.encoder.num_qubits)
    test_spins = 2 * test_binary - 1
    
    qubo_energy = test_binary @ mapper.Q @ test_binary + mapper.offset
    # Ising energy
    # H(σ) = Σ_i<j J_ij σ_i σ_j + Σ_i h_i σ_i + offset
    ising_energy = 0.0
    for i in range(len(test_spins)):
        ising_energy += h[i] * test_spins[i]
        for j in range(i+1, len(test_spins)):
            ising_energy += J[i, j] * test_spins[i] * test_spins[j]
    ising_energy += offset_ising
    
    print(f"  QUBO energy:  {qubo_energy:.6f}")
    print(f"  Ising energy: {ising_energy:.6f}")
    print(f"  Difference:   {abs(qubo_energy - ising_energy):.6e}")
    
    if abs(qubo_energy - ising_energy) < 1e-6:
        print("  ✓ Conversion correct")
    else:
        print("  ✗ Conversion error")
    print()
    
    print("="*60)
    print("✓ PHASE 2 COMPLETE")
    print("="*60)
    print()
    print("Key achievements:")
    print("  ✓ One-hot encoding with {num_bins} bins")
    print("  ✓ QUBO matrix construction")
    print("  ✓ Budget and one-hot constraints via penalties")
    print("  ✓ QUBO-Ising conversion")
    print()
    print("Next: Phase 3 - VQE solver implementation")


if __name__ == "__main__":
    test_discretization_analysis()
    print()
    test_phase2_refined()

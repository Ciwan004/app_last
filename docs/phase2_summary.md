# Phase 2 Complete: Ising/QUBO Mapping

## Summary

Successfully implemented the quantum mapping layer that transforms the classical Markowitz portfolio optimization problem into an Ising Hamiltonian suitable for VQE execution.

## Implementation

### 1. One-Hot Encoding (`OneHotEncoder`)

**Purpose**: Discretize continuous portfolio weights into binary variables.

**Approach**:
- Each asset's weight discretized into `N` bins: `{0, Δ, 2Δ, ..., 1}`
- For 11 bins: `{0.0, 0.1, 0.2, ..., 1.0}`
- Each asset uses one-hot encoding: exactly one bin is active
- Total qubits: `num_assets × num_bins`

**Example** (4 assets, 11 bins = 44 qubits):
```
Classical: w = [0.379, 0.217, 0.193, 0.211]
Encoded:   w = [0.4, 0.2, 0.2, 0.2]
Error:     L2 = 0.030 (3% discretization error)
```

**Design Rationale**:
- One-hot ensures exactly one weight level per asset (enforced by penalty)
- Finer bins → better approximation but more qubits
- 11 bins provides good balance: budget-feasible solutions exist

### 2. QUBO Construction (`IsingMapper.build_qubo`)

**Objective**: Map Markowitz problem to QUBO form `E(x) = x^T Q x + offset`

**Components**:

#### a) Markowitz Objective
```
Classical: (1/2) w^T Σ w - λ μ^T w
Encoded:   w_i = Σ_j∈bins(i) weight_j * x_ij (one-hot)
```

**Risk term**: `(1/2) Σ_ik covariance_ik * w_i * w_j`
- Diagonal Q: `Q_ii += 0.5 * Σ_ik * weight_i^2`
- Off-diagonal Q: `Q_ij += 2 * Σ_ik * weight_i * weight_j`

**Return term**: `-λ Σ_i μ_i * w_i`
- Diagonal Q: `Q_ii -= λ * μ_i * weight_i`

#### b) Budget Constraint Penalty
```
Penalty: P * (Σw_i - budget)^2
```

Expands to:
- `P * (Σw_i)^2 - 2P * budget * Σw_i + P * budget^2`
- Quadratic terms → Q matrix
- Linear terms → Q diagonal
- Constant → offset

#### c) One-Hot Constraint Penalty
```
For each asset: P * (Σ_j∈bins x_j - 1)^2
```

Ensures exactly one bin selected per asset.

**Result**: 44×44 QUBO matrix, 864 non-zero elements, offset=500

### 3. QUBO to Ising Conversion (`qubo_to_ising`)

**Transformation**: `x ∈ {0,1} → σ ∈ {-1,+1}` via `x = (σ + 1) / 2`

**Mathematical Derivation**:

Starting with QUBO:
```
E(x) = Σ_i Q_ii x_i + Σ_i<j 2*Q_ij x_i x_j + offset
```

Substitute `x_i = (σ_i + 1)/2`:
```
x_i x_j = (σ_i + 1)(σ_j + 1)/4 = (σ_i σ_j + σ_i + σ_j + 1)/4
```

Expanding:
```
E(σ) = Σ_i<j [2*Q_ij/4] σ_i σ_j 
     + Σ_i [Q_ii/2 + Σ_j≠i Q_ij/2] σ_i
     + [Σ_i Q_ii/2 + Σ_i<j Q_ij/2 + offset]
```

**Ising Parameters**:
- `J_ij = Q_ij / 2` (interaction coupling)
- `h_i = Q_ii/2 + Σ_j≠i Q_ij/2` (magnetic field)
- `offset_ising = Σ_i Q_ii/2 + Σ_i<j Q_ij/2 + offset`

**Verification**: Tested on random states, energy matches within 1e-12.

### 4. Validation Results

#### Encoding Test
```
Original:  [0.379, 0.217, 0.193, 0.211]
Decoded:   [0.4, 0.2, 0.2, 0.2]
Sum:       1.000000 ✓
Feasible:  True ✓
L2 Error:  0.030 (3%)
```

#### Conversion Test
```
QUBO energy:  12745.682000
Ising energy: 12745.682000
Difference:   1.8e-12 ✓
```

## Key Design Decisions

### Why One-Hot Encoding?
**Alternative**: Binary expansion (fewer qubits but harder constraints)

**Chosen Approach**: One-hot
- **Pro**: Natural representation, simple one-hot constraint
- **Pro**: Direct mapping weight_i → bin_j
- **Con**: More qubits (44 vs ~16 for binary)
- **Justified**: Clarity and correctness for first implementation

### Why Penalty Method for Constraints?
**Alternatives**: 
1. Hard constraints (Lagrange multipliers)
2. Problem reduction

**Chosen**: Penalty terms
- **Pro**: Standard approach in QAOA/VQE
- **Pro**: All constraints in one Hamiltonian
- **Pro**: Tunable via penalty coefficient
- **Con**: Need to balance penalty vs objective
- **Justified**: Penalty=100 works well for test problem

### Discretization Granularity
**Trade-off**: Bins vs Qubits vs Accuracy

```
3 bins  → 12 qubits → 10 feasible allocations
5 bins  → 20 qubits → 35 feasible allocations
11 bins → 44 qubits → 286 feasible allocations ← CHOSEN
```

**Rationale**: 11 bins (10% granularity) provides:
- Sufficient feasible space for budget constraint
- Reasonable discretization error (~3%)
- Manageable qubit count for VQE

## Quantum Physics Interpretation

### Ising Model
```
H = Σ_i<j J_ij σ_i σ_j + Σ_i h_i σ_i
```

- **J_ij**: Coupling between qubits i and j (from covariance)
- **h_i**: External field on qubit i (from returns)
- **Ground state |ψ₀⟩**: Encodes optimal portfolio

### Connection to Optimization
1. **Energy landscape**: Markowitz objective → Hamiltonian
2. **Ground state search**: VQE finds min-energy configuration
3. **Measurement**: Bitstring → portfolio weights (via decoder)

### Why This Works
- **Quadratic problem** (Markowitz) → **Quadratic Hamiltonian** (Ising)
- Both have same mathematical structure: `x^T Q x + b^T x`
- Quantum annealing naturally explores this energy landscape
- VQE uses variational principle to approximate ground state

## Files Created

```
/app/quantum/ising_mapper.py     - OneHotEncoder + IsingMapper
/app/test_phase2.py              - Initial tests (5 bins)
/app/test_phase2_refined.py      - Refined tests (11 bins)
/app/debug_conversion.py         - QUBO-Ising conversion verification
```

## Known Limitations

1. **Discretization error**: 3% for 11 bins
   - **Mitigation**: Increase bins (more qubits) or use hybrid approaches
   
2. **Penalty tuning**: Manual selection of P=100
   - **Mitigation**: Adaptive penalty or constraint satisfaction checking
   
3. **Qubit count**: 44 qubits for 4-asset problem
   - **Scaling**: 10 assets × 11 bins = 110 qubits (feasible on modern hardware)
   - **Mitigation**: Binary expansion or problem decomposition

4. **No cardinality constraint**: Not yet implemented
   - **Future**: Add penalty term for max number of assets

## Validation Status

- ✅ One-hot encoding/decoding
- ✅ QUBO matrix construction
- ✅ Budget constraint satisfaction
- ✅ One-hot constraint enforcement
- ✅ QUBO-Ising conversion correctness
- ✅ Classical solution encoding (3% error)
- ⚠️ Ground state search (44 qubits too many for brute force)

## Next: Phase 3

**VQE Execution** will:
1. Construct variational ansatz (parameterized quantum circuit)
2. Implement VQE optimization loop
3. Measure ground state
4. Decode results to portfolio weights
5. Compare with classical baseline

**Goal**: Find low-energy states using quantum circuits instead of classical search.

---

**Phase 2 Status**: ✅ COMPLETE AND VALIDATED

"""Debug QUBO to Ising conversion."""

import sys
sys.path.append('/app')

import numpy as np

# Simple test case
n = 3
Q = np.array([[1, 2, 0],
              [0, 3, 4],
              [0, 0, 5]], dtype=float)

offset_qubo = 10.0

print("QUBO Matrix (upper triangular):")
print(Q)
print(f"Offset: {offset_qubo}")
print()

# Test binary vector
x = np.array([1, 0, 1])
print(f"Test x: {x}")

# QUBO energy
# E(x) = x^T Q x + offset
# For upper triangular Q: E(x) = Σ_i Q_ii x_i + 2 Σ_i<j Q_ij x_i x_j + offset
qubo_energy = 0.0
for i in range(n):
    qubo_energy += Q[i, i] * x[i]
    for j in range(i+1, n):
        qubo_energy += 2 * Q[i, j] * x[i] * x[j]
qubo_energy += offset_qubo

print(f"QUBO energy (manual): {qubo_energy}")

# Also using matrix form
Q_full = Q + Q.T - np.diag(np.diag(Q))
qubo_energy_mat = x @ Q_full @ x + offset_qubo
print(f"QUBO energy (matrix): {qubo_energy_mat}")
print()

# Convert to Ising
# x_i = (σ_i + 1) / 2
# Substitute into QUBO

# Method 1: Direct expansion
# E(x) = Σ_i Q_ii x_i + Σ_i<j 2*Q_ij x_i x_j + offset
# x_i = (σ_i + 1)/2
# x_i x_j = (σ_i + 1)(σ_j + 1)/4 = (σ_i σ_j + σ_i + σ_j + 1)/4

J = np.zeros((n, n))
h = np.zeros(n)

# From quadratic terms (off-diagonal): 2*Q_ij * x_i * x_j
for i in range(n):
    for j in range(i+1, n):
        if Q[i, j] != 0:
            # 2*Q_ij * (σ_i σ_j + σ_i + σ_j + 1)/4
            J[i, j] = 2 * Q[i, j] / 4.0  # Coefficient of σ_i σ_j
            J[j, i] = J[i, j]
            h[i] += 2 * Q[i, j] / 4.0  # Coefficient of σ_i
            h[j] += 2 * Q[i, j] / 4.0  # Coefficient of σ_j

# From diagonal terms: Q_ii * x_i = Q_ii * (σ_i + 1)/2
for i in range(n):
    h[i] += Q[i, i] / 2.0  # Coefficient of σ_i

# Offset
offset_ising = offset_qubo
# From diagonal: Q_ii * 1/2
for i in range(n):
    offset_ising += Q[i, i] / 2.0
# From off-diagonal: 2*Q_ij * 1/4
for i in range(n):
    for j in range(i+1, n):
        offset_ising += 2 * Q[i, j] / 4.0

print("Ising parameters:")
print("J:")
print(J)
print(f"h: {h}")
print(f"offset: {offset_ising}")
print()

# Test with spins
sigma = 2 * x - 1
print(f"Test σ: {sigma}")

ising_energy = 0.0
for i in range(n):
    ising_energy += h[i] * sigma[i]
    for j in range(i+1, n):
        ising_energy += J[i, j] * sigma[i] * sigma[j]  # Sum only over i<j
ising_energy += offset_ising

print(f"Ising energy: {ising_energy}")
print(f"QUBO energy:  {qubo_energy}")
print(f"Match: {abs(ising_energy - qubo_energy) < 1e-10}")

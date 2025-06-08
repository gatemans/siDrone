from gwo import run_GWO
from pso import run_PSO
from spso import run_SPSO
from ga import run_GA
from benchmark_functions import sphere
import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
n_var = 4
lb = -10 * np.ones(n_var)
ub = 10 * np.ones(n_var)
max_iter = 100
pop_size = 10

# Run algorithms
gwo_best, _, gwo_curve = run_GWO(sphere, n_var, lb, ub, max_iter, pop_size)
pso_best, _, pso_curve = run_PSO(sphere, n_var, lb, ub, max_iter, pop_size)
spso_best, _, spso_curve = run_SPSO(sphere, n_var, lb, ub, max_iter, pop_size)
ga_best, _, ga_curve = run_GA(sphere, n_var, lb, ub, max_iter, pop_size)

# Print scores
print("GWO Best Score:", gwo_best)
print("PSO Best Score:", pso_best)
print("SPSO Best Score:", spso_best)
print("GA Best Score:", ga_best)

# Plot results
plt.figure(figsize=(10,6))
plt.plot(gwo_curve, label="GWO")
plt.plot(pso_curve, label="PSO")
plt.plot(spso_curve, label="SPSO")
plt.plot(ga_curve, label="GA")
plt.title("Convergence Curves")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()

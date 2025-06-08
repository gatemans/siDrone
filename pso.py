import numpy as np

def run_PSO(fobj, n_var, lb, ub, max_iter, pop_size):
    """
    Particle Swarm Optimization (PSO)

    Parameters:
    - fobj: objective function to minimize
    - n_var: number of variables
    - lb: lower bounds (array-like)
    - ub: upper bounds (array-like)
    - max_iter: maximum iterations
    - pop_size: swarm size

    Returns:
    - best_score: best fitness value found
    - best_pos: position of best solution
    - convergence: array with best fitness per iteration
    """

    w = 0.5      # inertia weight
    c1 = 2.0     # cognitive coefficient
    c2 = 2.0     # social coefficient

    # Initialize particle positions uniformly between lb and ub
    pos = np.random.uniform(lb, ub, (pop_size, n_var))
    # Initialize velocities to zero
    vel = np.zeros((pop_size, n_var))
    # Personal best positions start as initial positions
    pbest = np.copy(pos)

    # Evaluate initial personal best fitnesses
    pbest_val = np.array([fobj(p) for p in pbest])

    # Find global best position and its fitness
    gbest_idx = np.argmin(pbest_val)
    gbest = np.copy(pbest[gbest_idx])
    gbest_val = pbest_val[gbest_idx]

    convergence = np.zeros(max_iter)

    for t in range(max_iter):
        for i in range(pop_size):
            r1 = np.random.rand(n_var)
            r2 = np.random.rand(n_var)

            # Update velocity
            vel[i] = (w * vel[i] + 
                      c1 * r1 * (pbest[i] - pos[i]) + 
                      c2 * r2 * (gbest - pos[i]))

            # Update position
            pos[i] = pos[i] + vel[i]

            # Enforce bounds
            pos[i] = np.maximum(pos[i], lb)
            pos[i] = np.minimum(pos[i], ub)

            # Evaluate fitness
            fitness = fobj(pos[i])

            # Update personal best if current is better
            if fitness < pbest_val[i]:
                pbest[i] = pos[i]
                pbest_val[i] = fitness

                # Update global best if current is better
                if fitness < gbest_val:
                    gbest = pos[i]
                    gbest_val = fitness

        convergence[t] = gbest_val

    return gbest_val, gbest, convergence

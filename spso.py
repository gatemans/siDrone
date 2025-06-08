import numpy as np

def init(pop_size, n_var, lb, ub):
    """
    Initialize the population randomly within the lower and upper bounds.
    """
    return np.random.uniform(low=lb, high=ub, size=(pop_size, n_var))

def run_SPSO(fobj, n_var, lb, ub, max_iter, pop_size):
    # Inertia and acceleration constants
    w = 0.5      # Inertia weight
    c1 = 2       # Personal attraction
    c2 = 2       # Global attraction

    # Initialize particle positions and velocities
    pos = init(pop_size, n_var, lb, ub)
    vel = np.zeros((pop_size, n_var))

    # Initialize personal bests
    pbest = pos.copy()
    pbest_val = np.array([fobj(p) for p in pos])

    # Initialize global best
    gbest_idx = np.argmin(pbest_val)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    # Convergence curve
    convergence = np.zeros(max_iter)

    # Main loop
    for t in range(max_iter):
        for i in range(pop_size):
            # Vector differences to pbest and gbest (sphere vectors)
            dist_pbest = pbest[i] - pos[i]
            dist_gbest = gbest - pos[i]

            # Velocity update rule (Sphere-PSO)
            vel[i] = (w * vel[i] +
                      c1 * np.random.rand() * dist_pbest +
                      c2 * np.random.rand() * dist_gbest)

            # Update position
            pos[i] += vel[i]
            pos[i] = np.clip(pos[i], lb, ub)

            # Evaluate objective function
            val = fobj(pos[i])
            if val < pbest_val[i]:
                pbest[i] = pos[i].copy()
                pbest_val[i] = val

                # Update global best if needed
                if val < gbest_val:
                    gbest = pos[i].copy()
                    gbest_val = val

        # Save convergence info
        convergence[t] = gbest_val

    # Return best score, position, and convergence curve
    return gbest_val, gbest, convergence

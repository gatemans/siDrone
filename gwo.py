import numpy as np

def init(pop_size, n_var, lb, ub):
    """
    Initialize the population within bounds.
    """
    return np.random.uniform(low=lb, high=ub, size=(pop_size, n_var))

def run_GWO(fobj, n_var, lb, ub, max_iter, pop_size):
    # Initialize the population
    position = init(pop_size, n_var, lb, ub)

    # Initialize Alpha, Beta, and Delta wolves (best, second-best, third-best)
    alpha_pos = np.zeros(n_var)
    beta_pos = np.zeros(n_var)
    delta_pos = np.zeros(n_var)

    alpha = np.inf   # Best fitness
    beta = np.inf    # Second-best fitness
    delta = np.inf   # Third-best fitness

    # To store the convergence curve
    convergence = np.zeros(max_iter)

    # Main optimization loop
    for t in range(max_iter):
        for i in range(pop_size):
            # Ensure positions are within bounds
            position[i, :] = np.clip(position[i, :], lb, ub)

            # Evaluate fitness of the current position
            fit = fobj(position[i, :])

            # Update Alpha, Beta, Delta wolves
            if fit < alpha:
                delta = beta
                delta_pos = beta_pos.copy()
                beta = alpha
                beta_pos = alpha_pos.copy()
                alpha = fit
                alpha_pos = position[i, :].copy()
            elif fit < beta:
                delta = beta
                delta_pos = beta_pos.copy()
                beta = fit
                beta_pos = position[i, :].copy()
            elif fit < delta:
                delta = fit
                delta_pos = position[i, :].copy()

        # Linearly decrease parameter 'a' from 2 to 0
        a = 2 - t * (2 / max_iter)

        # Update the position of each search agent
        for i in range(pop_size):
            for j in range(n_var):
                # Coefficients for Alpha
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                # Coefficients for Beta
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                # Coefficients for Delta
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                # Distance vectors
                D1 = abs(C1 * alpha_pos[j] - position[i, j])
                D2 = abs(C2 * beta_pos[j]  - position[i, j])
                D3 = abs(C3 * delta_pos[j] - position[i, j])

                # Position updates
                X1 = alpha_pos[j] - A1 * D1
                X2 = beta_pos[j]  - A2 * D2
                X3 = delta_pos[j] - A3 * D3

                # New position is the average of X1, X2, and X3
                position[i, j] = (X1 + X2 + X3) / 3

            # Apply bounds after position update
            position[i, :] = np.clip(position[i, :], lb, ub)

        # Save the best fitness at this iteration
        convergence[t] = alpha

    # Return the best solution and convergence curve
    best_score = alpha
    best_pos = alpha_pos
    return best_score, best_pos, convergence

import numpy as np

def init(pop_size, n_var, lb, ub):
    """
    Initialize the population with random solutions within bounds.
    """
    return np.random.uniform(lb, ub, (pop_size, n_var))

def run_GA(fobj, n_var, lb, ub, max_iter, pop_size):
    # Genetic Algorithm parameters
    crossover_rate = 0.8
    mutation_rate = 0.1
    tournament_size = 3

    # Initialize population
    population = init(pop_size, n_var, lb, ub)
    fitness = np.array([fobj(ind) for ind in population])

    # Track the best solution
    best_idx = np.argmin(fitness)
    best_score = fitness[best_idx]
    best_pos = population[best_idx].copy()

    convergence = np.zeros(max_iter)

    for t in range(max_iter):
        new_population = []

        while len(new_population) < pop_size:
            # Tournament selection
            def select_parent():
                idxs = np.random.choice(pop_size, tournament_size, replace=False)
                best = idxs[np.argmin(fitness[idxs])]
                return population[best].copy()

            parent1 = select_parent()
            parent2 = select_parent()

            # Crossover (blend crossover)
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand(n_var)
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = alpha * parent2 + (1 - alpha) * parent1
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            def mutate(child):
                for j in range(n_var):
                    if np.random.rand() < mutation_rate:
                        mutation_amount = np.random.uniform(-1, 1)
                        child[j] += mutation_amount * (ub[j] - lb[j]) * 0.1
                return np.clip(child, lb, ub)

            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        # Evaluate new population
        population = np.array(new_population)
        fitness = np.array([fobj(ind) for ind in population])

        # Update best solution
        current_best_idx = np.argmin(fitness)
        current_best_score = fitness[current_best_idx]
        if current_best_score < best_score:
            best_score = current_best_score
            best_pos = population[current_best_idx].copy()

        convergence[t] = best_score

    return best_score, best_pos, convergence

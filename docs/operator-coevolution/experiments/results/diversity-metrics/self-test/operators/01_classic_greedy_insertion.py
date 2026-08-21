def repair_operator(problem, routes, removed, rng):
    order = removed.copy()
    rng.shuffle(order)
    return sequential_insertion(problem, routes, order, rng)

def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    unplaced = list(removed)
    while unplaced:
        pool = []
        for cust in unplaced:
            cands = insertion_candidates(problem, result, cust)
            for cand in cands:
                pool.append((cand[0], cust, cand[1], cand[2]))
        if not pool:
            break
        costs = np.array([item[0] for item in pool])
        mask = np.isfinite(costs)
        finite_mask_indices = np.where(mask)[0]
        if len(finite_mask_indices) == 0:
            weights = np.ones(len(pool))
        else:
            finite_costs = costs[mask]
            min_cost = np.min(finite_costs)
            shifted_costs = finite_costs - min_cost
            weights = np.exp(-shifted_costs)
            weights = weights / np.sum(weights)
        full_weights = np.zeros(len(pool))
        full_weights[mask] = weights
        idx = rng.choice(len(pool), p=full_weights)
        _, cust, r_idx, pos = pool[idx]
        candidate = (pool[idx][0], pool[idx][2], pool[idx][3])
        apply_insertion(result, cust, candidate)
        unplaced.remove(cust)
    return result

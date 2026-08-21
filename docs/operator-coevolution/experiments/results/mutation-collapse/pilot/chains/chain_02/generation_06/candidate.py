def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    if not removed:
        return result
    active_removed = list(removed)
    regret_scores = {}
    for cust in active_removed:
        cands = insertion_candidates(problem, result, cust)
        feasible_costs = []
        for delta, r_idx, pos in cands:
            if 0 <= pos <= len(result[r_idx]):
                temp_route = result[r_idx][:pos] + [cust] + result[r_idx][pos:]
                if route_feasible(problem, temp_route):
                    feasible_costs.append(delta)
            if len(feasible_costs) >= 2:
                break
        if len(feasible_costs) < 2:
            regret_scores[cust] = float('inf')
        else:
            regret_scores[cust] = feasible_costs[1] - feasible_costs[0]
    while active_removed:
        weights = []
        for cust in active_removed:
            r = regret_scores[cust]
            if np.isinf(r):
                r = 1e+100
            weights.append(r)
        total_weight = sum(weights)
        if total_weight == 0:
            probs = [1.0 / len(active_removed)] * len(active_removed)
        else:
            probs = [w / total_weight for w in weights]
        idx = rng.choice(len(active_removed), p=probs)
        cust = active_removed.pop(idx)
        cands = insertion_candidates(problem, result, cust)
        best_candidate = None
        for delta, r_idx, pos in cands:
            if 0 <= pos <= len(result[r_idx]):
                temp_route = result[r_idx][:pos] + [cust] + result[r_idx][pos:]
                if route_feasible(problem, temp_route):
                    best_candidate = (delta, r_idx, pos)
                    break
        if best_candidate:
            apply_insertion(result, cust, best_candidate)
        elif cands:
            apply_insertion(result, cust, cands[0])
    return result

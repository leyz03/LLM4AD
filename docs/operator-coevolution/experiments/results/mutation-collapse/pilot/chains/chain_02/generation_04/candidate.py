def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    if not removed:
        return result
    work_order = sorted(removed, key=lambda c: problem.customers[c].demand, reverse=True)
    INSERTION_BUDGET = min(5, len(removed))
    for customer in work_order:
        candidates = insertion_candidates(problem, result, customer)
        if not candidates:
            continue
        evaluated = []
        for i, (delta, route_idx, pos) in enumerate(candidates[:INSERTION_BUDGET]):
            if route_feasible(problem, result[route_idx][:pos] + [customer] + result[route_idx][pos:]):
                test_routes = clone_routes(result)
                apply_insertion(test_routes, customer, (delta, route_idx, pos))
                global_dist = solution_distance(problem, test_routes)
                evaluated.append((global_dist, delta, route_idx, pos))
        if evaluated:
            evaluated.sort(key=lambda x: x[0])
            best = evaluated[0]
            apply_insertion(result, customer, (best[1], best[2], best[3]))
        else:
            apply_insertion(result, customer, candidates[0])
    return result

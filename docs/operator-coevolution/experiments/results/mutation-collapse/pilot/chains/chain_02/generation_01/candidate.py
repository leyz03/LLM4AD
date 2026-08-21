def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    if not removed:
        return result
    work_order = list(removed)
    rng.shuffle(work_order)
    for customer in work_order:
        candidates = insertion_candidates(problem, result, customer)
        best_feasible = None
        for delta, route_idx, pos in candidates:
            current_route = result[route_idx]
            temp_route = current_route[:pos] + [customer] + current_route[pos:]
            if route_feasible(problem, temp_route):
                best_feasible = (delta, route_idx, pos)
                break
        if best_feasible:
            apply_insertion(result, customer, best_feasible)
        elif candidates:
            apply_insertion(result, customer, candidates[0])
    return result

def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    if not removed:
        return result
    work_order = sorted(removed, key=lambda c: problem.customers[c].deadline)
    for customer in work_order:
        candidates = insertion_candidates(problem, result, customer)
        if not candidates:
            continue
        best_feasible = None
        best_cost_increase = float('inf')
        for delta, route_idx, pos in candidates:
            current_route = result[route_idx]
            temp_route = current_route[:pos] + [customer] + current_route[pos:]
            if route_feasible(problem, temp_route):
                best_feasible = (delta, route_idx, pos)
                break
            if delta < best_cost_increase:
                best_cost_increase = delta
        if best_feasible:
            apply_insertion(result, customer, best_feasible)
        else:
            min_idx = candidates[0][1]
            min_pos = candidates[0][2]
            apply_insertion(result, customer, (candidates[0][0], min_idx, min_pos))
    return result

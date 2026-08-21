def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    remaining = list(removed)
    feasible_insertions = []
    for cust in remaining:
        cands = insertion_candidates(problem, result, cust)
        if not cands:
            continue
        for delta, route_idx, pos in cands[:3]:
            temp_routes = clone_routes(result)
            apply_insertion(temp_routes, cust, (delta, route_idx, pos))
            if route_feasible(problem, temp_routes[route_idx]):
                feasible_insertions.append((cust, delta, route_idx, pos))
    feasible_insertions.sort(key=lambda x: x[1])
    applied_customers = set()
    for cust, delta, route_idx, pos in feasible_insertions:
        if cust in applied_customers:
            continue
        if cust in remaining:
            apply_insertion(result, cust, (delta, route_idx, pos))
            remaining.remove(cust)
            applied_customers.add(cust)
    for cust in remaining[:]:
        cands = insertion_candidates(problem, result, cust)
        if cands:
            apply_insertion(result, cust, cands[0])
            remaining.remove(cust)
    return result

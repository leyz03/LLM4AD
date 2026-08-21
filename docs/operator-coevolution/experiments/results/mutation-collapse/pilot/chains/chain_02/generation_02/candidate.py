def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    if not removed:
        return result

    def get_constraint_score(customer):
        tw_start, tw_end = problem.tw[customer]
        tw_width = tw_end - tw_start
        demand = problem.demands[customer]
        max_capacity = problem.capacity
        return tw_width / max(1, max_capacity - demand)
    work_order = sorted(removed, key=get_constraint_score)
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
    for route_idx in range(len(result)):
        route = result[route_idx]
        if len(route) < 2:
            continue
        improved = True
        iterations = min(3, len(route))
        for _ in range(iterations):
            if not improved:
                break
            improved = False
            for i in range(len(route) - 2):
                j = i + 1
                temp_route = route[:i] + [route[j], route[i]] + route[i + 2:]
                if route_feasible(problem, temp_route):
                    old_dist = route_distance(problem, route)
                    new_dist = route_distance(problem, temp_route)
                    if new_dist < old_dist:
                        result[route_idx] = temp_route
                        improved = True
    return result

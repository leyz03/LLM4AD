def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    if not removed:
        return result
    customer_priorities = []
    for cust in removed:
        cands = insertion_candidates(problem, result, cust)
        feasible_costs = []
        check_count = 0
        for delta, r_idx, pos in cands:
            if check_count >= 2:
                break
            current_route = result[r_idx]
            if 0 <= pos <= len(current_route):
                temp_route = current_route[:pos] + [cust] + current_route[pos:]
                if route_feasible(problem, temp_route):
                    feasible_costs.append(delta)
            check_count += 1
        if len(feasible_costs) == 0:
            regret = float('inf')
        elif len(feasible_costs) == 1:
            regret = float('inf')
        else:
            regret = feasible_costs[1] - feasible_costs[0]
        customer_priorities.append((regret, cust))
    customer_priorities.sort(key=lambda x: x[0], reverse=True)
    work_order = [item[1] for item in customer_priorities]
    for cust in work_order:
        cands = insertion_candidates(problem, result, cust)
        best_candidate = None
        for delta, r_idx, pos in cands:
            current_route = result[r_idx]
            if 0 <= pos <= len(current_route):
                temp_route = current_route[:pos] + [cust] + current_route[pos:]
                if route_feasible(problem, temp_route):
                    best_candidate = (delta, r_idx, pos)
                    break
        if best_candidate:
            apply_insertion(result, cust, best_candidate)
        elif cands:
            apply_insertion(result, cust, cands[0])
    return result

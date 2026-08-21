def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)

    def get_regret_score(customer):
        candidates = insertion_candidates(problem, result, customer)
        feasible_opts = []
        for delta, r_idx, pos in candidates:
            test_route = result[r_idx][:]
            test_route.insert(pos, customer)
            if route_feasible(problem, test_route):
                feasible_opts.append(delta)
        if not feasible_opts:
            return float('inf')
        best = feasible_opts[0]
        if len(feasible_opts) > 1:
            second_best = feasible_opts[1]
        else:
            second_best = best + 1000000
        return second_best - best
    ordered_customers = sorted(removed, key=get_regret_score)
    for customer in ordered_customers:
        candidates = insertion_candidates(problem, result, customer)
        feasible_opts = []
        for delta, r_idx, pos in candidates:
            test_route = result[r_idx][:]
            test_route.insert(pos, customer)
            if route_feasible(problem, test_route):
                feasible_opts.append((delta, r_idx, pos))
        if not feasible_opts:
            result.append([customer])
        else:
            k = min(5, len(feasible_opts))
            idx = rng.integers(0, k)
            best_choice = feasible_opts[idx]
            apply_insertion(result, customer, best_choice)
    return result

def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)

    def estimate_difficulty(customer):
        candidates = insertion_candidates(problem, result, customer)
        feasible_count = 0
        for delta, r_idx, pos in candidates:
            test_route = result[r_idx][:]
            test_route.insert(pos, customer)
            if route_feasible(problem, test_route):
                feasible_count += 1
        return feasible_count
    sorted_removed = sorted(removed, key=estimate_difficulty)
    for customer in sorted_removed:
        candidates = insertion_candidates(problem, result, customer)
        feasible_opts = []
        for delta, r_idx, pos in candidates:
            test_route = result[r_idx][:]
            test_route.insert(pos, customer)
            if route_feasible(problem, test_route):
                feasible_opts.append((delta, r_idx, pos))
        if not feasible_opts:
            result.append([customer])
            continue
        feasible_opts.sort(key=lambda x: x[0])
        epsilon = 0.1
        if rng.random() < epsilon:
            idx = rng.choice(len(feasible_opts))
        else:
            k = min(3, len(feasible_opts))
            idx = rng.choice(k)
        apply_insertion(result, customer, feasible_opts[idx])
    return [route for route in result if route]

def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)

    def feasibility_score(customer):
        candidates = insertion_candidates(problem, result, customer)
        feasible_count = sum((1 for d, r_idx, pos in candidates if route_feasible(problem, result[r_idx][:])))
        return -feasible_count if feasible_count > 0 else float('inf')
    ordered_removed = sorted(removed, key=feasibility_score)
    for customer in ordered_removed:
        candidates = insertion_candidates(problem, result, customer)
        inserted = False
        for delta, r_idx, pos in candidates:
            temp_route = result[r_idx][:]
            temp_route.insert(pos, customer)
            if route_feasible(problem, temp_route):
                apply_insertion(result, customer, (delta, r_idx, pos))
                inserted = True
                break
        if not inserted:
            result.append([customer])
    return result

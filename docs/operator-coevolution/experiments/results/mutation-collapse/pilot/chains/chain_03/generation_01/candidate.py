def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    for i in rng.permutation(len(removed)):
        customer = removed[i]
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

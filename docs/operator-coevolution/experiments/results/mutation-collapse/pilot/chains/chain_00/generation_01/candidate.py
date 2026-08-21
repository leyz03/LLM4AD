def repair_operator(problem, routes, removed, rng):
    del rng
    result = clone_routes(routes)
    sorted_removed = sorted(removed, key=lambda c: problem.l[c])
    for customer in sorted_removed:
        candidates = insertion_candidates(problem, result, customer)
        if candidates:
            apply_insertion(result, customer, candidates[0])
    return result

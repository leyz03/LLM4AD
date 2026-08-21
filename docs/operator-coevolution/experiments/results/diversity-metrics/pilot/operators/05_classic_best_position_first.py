def repair_operator(problem, routes, removed, rng):
    del rng
    result = clone_routes(routes)
    pending = removed.copy()
    while pending:
        choices = [(insertion_candidates(problem, result, c)[0][0], c, insertion_candidates(problem, result, c)[0]) for c in pending]
        _, customer, candidate = min(choices, key=lambda item: (item[0], item[1]))
        apply_insertion(result, customer, candidate)
        pending.remove(customer)
    return result

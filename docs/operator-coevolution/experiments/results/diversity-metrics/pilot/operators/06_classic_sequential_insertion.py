def repair_operator(problem, routes, removed, rng):
    del rng
    result = clone_routes(routes)
    for customer in removed:
        apply_insertion(result, customer, insertion_candidates(problem, result, customer)[0])
    return result

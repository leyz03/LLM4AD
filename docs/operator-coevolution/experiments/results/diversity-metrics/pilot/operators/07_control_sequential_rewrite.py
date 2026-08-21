def repair_operator(problem, routes, removed, rng):
    del rng
    result = clone_routes(routes)
    pending = tuple(removed)
    index = 0
    while index < len(pending):
        customer = pending[index]
        options = insertion_candidates(problem, result, customer)
        apply_insertion(result, customer, min(options, key=lambda item: item[0]))
        index += 1
    return result

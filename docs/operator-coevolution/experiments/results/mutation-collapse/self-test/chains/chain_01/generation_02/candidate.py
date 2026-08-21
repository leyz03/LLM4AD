def repair_operator(problem, routes, removed, rng):
    del problem, rng
    result = clone_routes(routes)
    for customer in removed:
        result.append([customer])
    return result

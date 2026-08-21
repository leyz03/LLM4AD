def repair_operator(problem, routes, removed, rng):
    working_routes = clone_routes(routes)
    pending = list(removed)
    while pending:
        constraints = []
        for cust in pending:
            cands = insertion_candidates(problem, working_routes, cust)
            constraints.append((len(cands), cust))
        constraints.sort(key=lambda x: x[0])
        cust = constraints[0][1]
        pending.remove(cust)
        cands = insertion_candidates(problem, working_routes, cust)
        if cands:
            best_cand = min(cands, key=lambda x: x[0])
            apply_insertion(working_routes, cust, best_cand)
    return working_routes

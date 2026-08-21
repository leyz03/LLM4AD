def repair_operator(problem, routes, removed, rng):
    working_routes = clone_routes(routes)
    pending = list(removed)
    while pending:
        scores = []
        for cust in pending:
            cand = insertion_candidates(problem, working_routes, cust)
            count = len(cand)
            p_count = count if count > 0 else float('inf')
            best_cost = cand[0][0] if count > 0 else float('inf')
            jitter = rng.random()
            scores.append((p_count, best_cost, jitter, cust))
        scores.sort(key=lambda x: (x[0], x[1], x[2]))
        selected_cust = scores[0][3]
        try:
            idx = pending.index(selected_cust)
            pending.pop(idx)
        except ValueError:
            continue
        cand = insertion_candidates(problem, working_routes, selected_cust)
        if cand:
            apply_insertion(working_routes, selected_cust, cand[0])
    return working_routes

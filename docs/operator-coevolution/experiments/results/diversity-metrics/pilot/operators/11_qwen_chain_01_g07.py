def repair_operator(problem, routes, removed, rng):
    working_routes = clone_routes(routes)
    pending = list(removed)
    while pending:
        evaluations = []
        for cust in pending:
            cands = insertion_candidates(problem, working_routes, cust)
            if not cands:
                evaluations.append((-1, float('inf'), cust))
            else:
                best_cost = cands[0][0]
                count = len(cands)
                evaluations.append((count, best_cost, cust))
        evaluations.sort(key=lambda x: (x[0], x[1]))
        min_count = evaluations[0][0]
        tied_indices = [i for i, (cnt, _, _) in enumerate(evaluations) if cnt == min_count]
        k = min(len(tied_indices), max(1, rng.integers(2, 5)))
        idx = rng.integers(0, len(tied_indices))
        target_idx = tied_indices[idx % len(tied_indices)]
        target_cust = evaluations[target_idx][2]
        pending.remove(target_cust)
        cands = insertion_candidates(problem, working_routes, target_cust)
        if cands:
            apply_insertion(working_routes, target_cust, cands[0])
    return working_routes

def repair_operator(problem, routes, removed, rng):
    working_routes = clone_routes(routes)
    pending = list(removed)
    while pending:
        regrets = []
        for cust in pending:
            cand = insertion_candidates(problem, working_routes, cust)
            if len(cand) < 2:
                r = 0.0
            else:
                r = cand[1][0] - cand[0][0]
            regrets.append(r)
        weights = [max(r, 0.01) for r in regrets]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        idx = rng.choice(len(pending), p=probs)
        cust = pending[idx]
        pending.pop(idx)
        cand = insertion_candidates(problem, working_routes, cust)
        K = min(3, len(cand))
        spot_idx = rng.integers(0, K)
        insertion_point = cand[spot_idx]
        apply_insertion(working_routes, cust, insertion_point)
    return working_routes

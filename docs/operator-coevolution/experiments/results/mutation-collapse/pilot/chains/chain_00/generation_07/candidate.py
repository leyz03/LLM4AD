def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    to_repair = list(removed)
    priority_queue = []
    for cust in to_repair:
        cands = insertion_candidates(problem, result, cust)
        if not cands:
            continue
        if len(cands) > 1:
            regret = cands[1][0] - cands[0][0]
        else:
            regret = float('inf')
        priority_queue.append((regret, cust))
    priority_queue.sort(key=lambda x: x[0], reverse=True)
    for _, cust in priority_queue:
        cands = insertion_candidates(problem, result, cust)
        if not cands:
            continue
        num_options = len(cands)
        k = min(num_options, 5)
        idx = rng.integers(k)
        apply_insertion(result, cust, cands[idx])
    return result

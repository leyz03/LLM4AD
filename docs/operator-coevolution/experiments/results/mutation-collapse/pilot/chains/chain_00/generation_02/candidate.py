def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    scored_customers = []
    for customer in removed:
        candidates = insertion_candidates(problem, result, customer)
        if candidates:
            best_delta = candidates[0][0]
            top_k = min(len(candidates), 4)
            scored_customers.append((customer, best_delta, candidates[:top_k]))
        else:
            scored_customers.append((customer, float('inf'), []))
    scored_customers.sort(key=lambda x: x[1], reverse=True)
    for customer, _, candidates in scored_customers:
        if not candidates:
            continue
        deltas = [c[0] for c in candidates]
        weights = np.array([1.0 / max(1e-09, d) for d in deltas])
        weights /= weights.sum()
        idx = rng.choice(len(candidates), p=weights)
        apply_insertion(result, customer, candidates[idx])
    return result

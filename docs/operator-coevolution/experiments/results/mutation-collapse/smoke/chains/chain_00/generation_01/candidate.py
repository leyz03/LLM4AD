def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)

    def get_candidate_count(customer):
        candidates = insertion_candidates(problem, result, customer)
        return len(candidates) if candidates else 0
    ordered_customers = sorted(removed, key=lambda c: (get_candidate_count(c), rng.random()))
    for customer in ordered_customers:
        candidates = insertion_candidates(problem, result, customer)
        if not candidates:
            continue
        best_delta = candidates[0][0]
        if len(candidates) >= 2:
            regrets = [(best_delta - c[0], c) for c in candidates]
            regrets.sort(key=lambda x: (-x[0], rng.random()))
            apply_insertion(result, customer, regrets[0][1])
        else:
            apply_insertion(result, customer, candidates[0])
    return result

def repair_operator(problem, routes, removed, rng):
    new_routes = clone_routes(routes)
    route_slack = []
    for r_idx, route in enumerate(new_routes):
        slack = problem.capacity - sum((problem.demand[c] for c in route))
        route_slack.append(slack)
    num_removed = len(removed)
    base_regret = min(max(int(num_removed * 0.5 + 1), 1), 3)
    regret_values = [base_regret, base_regret + 1]
    sorted_removed = sorted(removed, key=lambda c: problem.demand[c], reverse=True)
    for customer in sorted_removed:
        candidates = insertion_candidates(problem, new_routes, customer)
        if not candidates:
            continue
        top_k = min(len(candidates), max(regret_values))
        top_candidates = candidates[:top_k]
        regrets = []
        for i in range(min(top_k, len(top_candidates))):
            delta = top_candidates[i][0]
            regrets.append((i, delta))
        if len(top_candidates) >= 2:
            best_cost = top_candidates[0][0]
            worst_cost = top_candidates[-1][0]
            regret_score = worst_cost - best_cost
            selection_probs = [1.0 / (1.0 + abs(r[0] - top_k // 2)) for r in range(len(top_candidates))]
            total_prob = sum(selection_probs)
            selection_probs = [p / total_prob for p in selection_probs]
            idx_in_top = rng.choice(len(top_candidates), p=selection_probs)
            selected_candidate = top_candidates[idx_in_top]
        else:
            selected_candidate = top_candidates[0]
        apply_insertion(new_routes, customer, selected_candidate)
    return new_routes

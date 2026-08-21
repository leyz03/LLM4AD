def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    remaining = list(removed)
    while remaining:
        best_idx = -1
        max_regret = -1.0
        selected_cands = None
        for i, customer in enumerate(remaining):
            cands = insertion_candidates(problem, result, customer)
            if not cands:
                continue
            if len(cands) >= 2:
                regret = cands[1][0] - cands[0][0]
            else:
                regret = float('inf')
            is_better = False
            if regret > max_regret:
                is_better = True
            elif regret == float('inf') and max_regret == float('inf'):
                if rng.random() < 0.5:
                    is_better = True
            elif abs(regret - max_regret) < 1e-09:
                if rng.random() < 0.5:
                    is_better = True
            if is_better:
                max_regret = regret
                best_idx = i
                selected_cands = cands
        if selected_cands:
            customer = remaining.pop(best_idx)
            apply_insertion(result, customer, selected_cands[0])
        else:
            break
    return result

def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    remaining = list(removed)
    while remaining:
        regrets = []
        valid_indices = []
        for i, cust in enumerate(remaining):
            cands = insertion_candidates(problem, result, cust)
            if not cands:
                continue
            if len(cands) >= 2:
                reg = cands[1][0] - cands[0][0]
            else:
                reg = float('inf')
            regrets.append(reg)
            valid_indices.append(i)
        if not valid_indices:
            break
        inf_indices = [idx for idx, reg in zip(valid_indices, regrets) if reg == float('inf')]
        if inf_indices:
            local_idx = rng.choice(len(inf_indices))
            selected_remaining_idx = inf_indices[local_idx]
        else:
            total_regret = sum(regrets)
            if total_regret == 0:
                probs = [1.0 / len(regrets)] * len(regrets)
            else:
                probs = [r / total_regret for r in regrets]
            local_idx = rng.choice(len(regrets), p=probs)
            selected_remaining_idx = valid_indices[local_idx]
        customer = remaining.pop(selected_remaining_idx)
        cands = insertion_candidates(problem, result, customer)
        if cands:
            apply_insertion(result, customer, cands[0])
    return result

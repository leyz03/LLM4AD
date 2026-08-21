def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    remaining = list(removed)
    while remaining:
        candidates_info = []
        valid_indices = []
        for i, cust in enumerate(remaining):
            cands = insertion_candidates(problem, result, cust)
            if not cands:
                continue
            best_delta = cands[0][0]
            if len(cands) >= 2:
                reg = cands[1][0] - cands[0][0]
            else:
                reg = float('inf')
            if reg == float('inf'):
                score = -best_delta
            else:
                score = reg - best_delta * 0.3
            candidates_info.append({'score': score, 'customer': cust, 'valid_idx': i})
            valid_indices.append(i)
        if not candidates_info:
            break
        best_candidate = max(candidates_info, key=lambda x: x['score'])
        selected_remaining_idx = best_candidate['valid_idx']
        customer = remaining.pop(selected_remaining_idx)
        cands = insertion_candidates(problem, result, customer)
        if cands:
            apply_insertion(result, customer, cands[0])
    return result

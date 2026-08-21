def repair_operator(problem, routes, removed, rng):
    result = clone_routes(routes)
    pending = removed.copy()
    noise_scale = 0.2 * float(np.mean(problem.distance))
    while pending:
        choices = []
        for customer in pending:
            for candidate in insertion_candidates(problem, result, customer):
                noisy_delta = candidate[0] + rng.uniform(-noise_scale, noise_scale)
                choices.append((float(noisy_delta), customer, candidate))
        _, customer, candidate = min(choices, key=lambda item: item[0])
        apply_insertion(result, customer, candidate)
        pending.remove(customer)
    return result

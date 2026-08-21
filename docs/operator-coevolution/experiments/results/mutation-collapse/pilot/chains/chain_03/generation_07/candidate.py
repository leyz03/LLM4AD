def repair_operator(problem, routes, removed, rng):
    res = clone_routes(routes)
    fitting_customers = []
    pending_customers = []
    customer_info = {}
    for cust in removed:
        candidates = insertion_candidates(problem, res, cust)
        feasible_opts = []
        for delta, r_idx, pos in candidates:
            test_route = res[r_idx][:]
            test_route.insert(pos, cust)
            if route_feasible(problem, test_route):
                feasible_opts.append((delta, r_idx, pos))
        if not feasible_opts:
            pending_customers.append(cust)
        else:
            best_delta = feasible_opts[0][0]
            second_delta = feasible_opts[1][0] if len(feasible_opts) > 1 else float('inf')
            regret = second_delta - best_delta
            fitting_customers.append(cust)
            customer_info[cust] = {'regret': regret, 'opts': feasible_opts}
    fitting_customers.sort(key=lambda c: customer_info[c]['regret'], reverse=True)
    for cust in fitting_customers:
        apply_insertion(res, cust, customer_info[cust]['opts'][0])
    rng.shuffle(pending_customers)
    new_routes_buffer = []
    for cust in pending_customers:
        candidates_in_new = insertion_candidates(problem, new_routes_buffer, cust)
        best_fit = None
        for delta, r_idx, pos in candidates_in_new:
            test_route = new_routes_buffer[r_idx][:]
            test_route.insert(pos, cust)
            if route_feasible(problem, test_route):
                best_fit = (delta, r_idx, pos)
                break
        if best_fit:
            apply_insertion(new_routes_buffer, cust, best_fit)
        else:
            new_routes_buffer.append([cust])
    res.extend(new_routes_buffer)
    return [route for route in res if route]

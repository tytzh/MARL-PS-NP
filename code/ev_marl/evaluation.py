"""Departure metrics: currency/EV, currency/EV, fraction, SoC fraction."""
import json
import random
import numpy as np


def indicators(cost, profit, soc, target, is_cs):
    cost, profit, soc, target = [np.asarray(x, dtype=float) for x in (cost, profit, soc, target)]
    if not all(np.isfinite(x).all() for x in (cost, profit, soc, target)):
        raise ValueError("Non-finite evaluation data")
    is_cs = np.asarray(is_cs, dtype=bool)
    success = is_cs & (soc >= target - 1e-9)
    failure = is_cs & ~success
    success_all = soc >= target - 1e-9
    failure_all = ~success_all
    conditional_dsoc = float((target[failure] - soc[failure]).mean()) if failure.any() else None
    all_cs_dsoc = float(np.maximum(target[is_cs] - soc[is_cs], 0.0).mean()) if is_cs.any() else None
    return dict(costb=float(cost[success].mean()) if success.any() else None,
                cost_all=float(cost.mean()) if len(cost) else None,
                profitu=float(profit.mean()) if len(profit) else None,
                psoc=float(success.sum() / is_cs.sum()) if is_cs.any() else None,
                psoc_all=float(success_all.mean()) if len(soc) else None,
                # Legacy table convention: a fully successful scenario has zero shortfall.
                dsoc=conditional_dsoc if conditional_dsoc is not None else 0.0,
                dsoc_conditional=conditional_dsoc,
                dsoc_failed_cs=conditional_dsoc,
                dsoc_all_cs=all_cs_dsoc,
                dsoc_all_ev=float(np.maximum(target - soc, 0.0).mean()) if len(soc) else None,
                n_evaluated=len(soc), n_cs_evaluated=int(is_cs.sum()),
                n_demand_evaluated=int(is_cs.sum()),
                n_success=int(success.sum()), n_failure=int(failure.sum()),
                n_success_all=int(success_all.sum()), n_failure_all=int(failure_all.sum()))


def evaluate(trained, n_evs, seed, cap_lower=0.1, cap_upper=0.9,
             infrastructure_scenario="hybrid", transformer_scope="cs_only",
             transformer_definition="total_grid", infrastructure_assignment="exclusive"):
    import training_core as core
    import training_utils as ft
    from constants import rate_CS_b, rate_CS_c, rate_PT_b
    random.seed(seed)
    np.random.seed(seed)
    centers = np.asarray(trained["centers"])

    def assign(features, evs, n_clusters):
        # Assign fresh EVs to the training centers, without refitting masks.
        for i, (feature, ev) in enumerate(zip(features, evs)):
            if centers.shape[1] == len(feature) + 1:
                feature = [*feature, ev.cap_c / 6.0]
            distance = np.linalg.norm(centers - feature, axis=1)
            exact = distance == 0
            if exact.any():
                weights = exact / exact.sum()
            else:
                weights = (distance.min() / distance) ** 2
                weights /= weights.sum()
            ev.weight_mask = weights
            ev.id_mask = int(weights.argmax())
            ev.idx_ev_near = range(len(evs))
        return evs, centers

    original = core.mask_list_EVs
    previous_assignment = getattr(core, "INFRASTRUCTURE_ASSIGNMENT", "exclusive")
    core.mask_list_EVs = assign
    core.INFRASTRUCTURE_ASSIGNMENT = infrastructure_assignment
    try:
        data = core.initialization(trained["n_group"])
    finally:
        core.mask_list_EVs = original
        core.INFRASTRUCTURE_ASSIGNMENT = previous_assignment
    price, price_pu, load, load_pu = data[:4]
    evs, incs, inpt, _ = core.get_data_episode(
        n_evs, *data[11:14],
        selection_seed=(seed if infrastructure_assignment == "controlled_overlap" else None))
    demand_incs, demand_inpt = incs, inpt
    incs, inpt = core.apply_infrastructure_scenario(
        incs, inpt, infrastructure_scenario)
    if infrastructure_assignment == "controlled_overlap":
        incs, inpt = core.resolve_controlled_contact_conflicts(incs, inpt)
    evs = core.synchronize_infrastructure_state(evs, incs, inpt)
    schedule = core.get_EVs_exist24(incs, inpt, evs)
    initial_soc = np.array([ev.soc_t_b for ev in evs])
    initial_cap_soc = np.array([ev.soc_t_c for ev in evs])
    capacity = np.array([ev.cap_b for ev in evs])
    demand_cs_hours = np.array([sum(record[1]) if record else 0. for record in demand_incs])
    demand_ter_hours = np.array([sum(record[1]) if record else 0. for record in demand_inpt])
    demand_ter_contact_hour = np.array(
        [float(record[0][0]) if record and record[0] else -1.0
         for record in demand_inpt], dtype=float)
    cs_hours = np.array([sum(record[1]) if record else 0. for record in incs])
    cost, profit = np.zeros(n_evs), np.zeros(n_evs)
    active = np.zeros(n_evs, dtype=bool)
    cap_exposure_hours = 0.
    cap_violation_hours = 0.
    cap_violation_area = 0.
    cap_max_violation = 0.
    cap_low_hours = 0.
    cap_high_hours = 0.
    cap_requested_abs_area = 0.
    cap_executed_abs_area = 0.
    cap_clipped_hours = 0.
    battery_requested_abs_area = 0.
    battery_executed_abs_area = 0.
    battery_clipped_hours = 0.
    battery_action_abs_area = 0.
    battery_action_saturated_hours = 0.
    transformer_violation_steps = 0
    transformer_excess_sum = 0.
    transformer_ev_allowance_sum = 0.
    transformer_max_excess = 0.
    transformer_net_kw = []
    base_load_kw = []
    price_per_kwh = []
    battery_cs_kw, battery_ter_kw = [], []
    capacitor_cs_kw, capacitor_ter_kw = [], []
    from constants import cap_trans
    for hour in range(24):
        existing = schedule[hour]
        p, pp, l, lp = ft.refresh_price_nonEVload(hour, len(price), price, price_pu, load, load_pu)
        base_load_kw.append(float(l[0]))
        transformer_ev_allowance_sum += max(cap_trans - abs(float(l[0])), 0.)
        price_per_kwh.append(float(p[0]))
        if not existing[0]:
            battery_cs_kw.append(0.)
            battery_ter_kw.append(0.)
            capacitor_cs_kw.append(0.)
            capacitor_ter_kw.append(0.)
            net_kw = float(l[0])
            constraint_kw = 0.0 if transformer_definition == "ev_only" else net_kw
            baseline_kw = 0.0 if transformer_definition == "ev_only" else float(l[0])
            total_excess_kw = max(abs(constraint_kw) - cap_trans, 0.)
            baseline_excess_kw = max(abs(baseline_kw) - cap_trans, 0.)
            excess_kw = max(total_excess_kw - baseline_excess_kw, 0.)
            transformer_net_kw.append(net_kw)
            transformer_violation_steps += int(excess_kw > 1e-9)
            transformer_excess_sum += excess_kw
            transformer_max_excess = max(transformer_max_excess, excess_kw)
            continue
        evs, fc, fb = ft.get_evs_group(existing, evs, pp, lp, trained["n_knn"])
        ab, ac, ub, uc = ft.choose_actions_mask5test(
            existing, evs, fc, fb, trained["agent_b"], trained["agent_c"],
            rate_CS_b, rate_CS_c, trained["n_group"])
        requested_ab = np.asarray(ab, dtype=float).copy()
        requested_ac = np.asarray(ac, dtype=float).copy()
        ab, ac, evs = ft.get_real_actions2(existing, evs, ab, ac)
        battery_cs_kw.append(float(sum(a * d for a, d, infra in
                                       zip(ab, existing[1], existing[2]) if infra == 0)))
        battery_ter_kw.append(float(sum(a * d for a, d, infra in
                                        zip(ab, existing[1], existing[2]) if infra == 1)))
        capacitor_cs_kw.append(float(sum(a * d for a, d, infra in
                                         zip(ac, existing[1], existing[2]) if infra == 0)))
        capacitor_ter_kw.append(float(sum(a * d for a, d, infra in
                                          zip(ac, existing[1], existing[2]) if infra == 1)))
        for i, ev_id in enumerate(existing[0]):
            eb = ab[i] * existing[1][i]
            ec = ac[i] * existing[1][i]
            cost[ev_id] += eb * p[0] + abs(eb) * trained["price_degrad"]
            profit[ev_id] -= ec * p[0]
            active[ev_id] = True
        evs = ft.update_dur_action_soc3(len(existing[0]), existing, ab, ac, evs, p[0], ub, uc)
        grid_action = sum((ab_i + ac_i) * duration
                          for ab_i, ac_i, duration, infra in
                          zip(ab, ac, existing[1], existing[2])
                          if transformer_scope == "all" or infra == 0)
        net_kw = float(l[0] + grid_action)
        constraint_kw = float(grid_action if transformer_definition == "ev_only" else net_kw)
        baseline_kw = 0.0 if transformer_definition == "ev_only" else float(l[0])
        total_excess_kw = max(abs(constraint_kw) - cap_trans, 0.)
        baseline_excess_kw = max(abs(baseline_kw) - cap_trans, 0.)
        excess_kw = max(total_excess_kw - baseline_excess_kw, 0.)
        transformer_net_kw.append(net_kw)
        transformer_violation_steps += int(excess_kw > 1e-9)
        transformer_excess_sum += excess_kw
        transformer_max_excess = max(transformer_max_excess, excess_kw)
        for position, (ev_id, duration) in enumerate(zip(existing[0], existing[1])):
            battery_action_abs_area += duration * abs(float(ub[position]))
            battery_action_saturated_hours += duration * (abs(float(ub[position])) >= 0.99)
            battery_requested_abs_area += duration * abs(requested_ab[position])
            battery_executed_abs_area += duration * abs(ab[position])
            battery_clipped_hours += duration * (abs(requested_ab[position] - ab[position]) > 1e-9)
            violation = max(cap_lower - evs[ev_id].soc_t_c, 0.,
                            evs[ev_id].soc_t_c - cap_upper)
            cap_exposure_hours += duration
            cap_violation_hours += duration * (violation > 1e-9)
            cap_violation_area += duration * violation
            cap_max_violation = max(cap_max_violation, violation)
            cap_low_hours += duration * (evs[ev_id].soc_t_c < cap_lower - 1e-9)
            cap_high_hours += duration * (evs[ev_id].soc_t_c > cap_upper + 1e-9)
            cap_requested_abs_area += duration * abs(requested_ac[position])
            cap_executed_abs_area += duration * abs(ac[position])
            cap_clipped_hours += duration * (abs(requested_ac[position] - ac[position]) > 1e-9)
    # Exclude EVs whose recorded departure is beyond the simulated day.
    censored = np.array([any(h >= 24 for record in (demand_incs[i], demand_inpt[i]) if record
                             for h in record[0]) for i in range(n_evs)])
    demand = np.array([bool(demand_incs[i]) or bool(demand_inpt[i]) for i in range(n_evs)])
    included = demand & ~censored
    # Paper metrics costb/psoc/dsoc use only EVs with charging-station exposure;
    # wireless-road-only EVs are physically unable to satisfy the same target.
    is_cs = included & (demand_cs_hours > 0)
    soc = np.array([ev.soc_t_b for ev in evs])
    target = np.array([ev.soc_exp_b for ev in evs])
    result = indicators(cost[included], profit[included], soc[included], target[included], is_cs[included])
    cs_terminal = is_cs
    result.update(
        mean_departure_soc=float(soc[cs_terminal].mean()) if cs_terminal.any() else None,
        mean_departure_target_margin=float(
            (soc[cs_terminal] - target[cs_terminal]).mean()) if cs_terminal.any() else None)
    reachable_soc = np.minimum(
        initial_soc + (rate_CS_b * demand_cs_hours + rate_PT_b * demand_ter_hours)
        / np.maximum(capacity, 1e-12), 1.0)
    reachable_cs = cs_terminal & (reachable_soc >= target - 1e-9)
    unreachable_cs = cs_terminal & ~reachable_cs
    result.update(
        unreachable_cs_rate=float(unreachable_cs.sum() / cs_terminal.sum())
        if cs_terminal.any() else None,
        psoc_reachable=float((soc[reachable_cs] >= target[reachable_cs] - 1e-9).mean())
        if reachable_cs.any() else None,
        dsoc_reachable_cs=float(np.maximum(
            target[reachable_cs] - soc[reachable_cs], 0.0).mean())
        if reachable_cs.any() else None,
        target_tracking_mae_cs=float(np.abs(
            soc[cs_terminal] - target[cs_terminal]).mean())
        if cs_terminal.any() else None,
        excess_soc_all_cs=float(np.maximum(
            soc[cs_terminal] - target[cs_terminal], 0.0).mean())
        if cs_terminal.any() else None)
    terminal_cap_soc = np.array([ev.soc_t_c for ev in evs])
    cap_inventory_change = np.array([
        (terminal_cap_soc[i] - initial_cap_soc[i]) * ev.cap_c
        for i, ev in enumerate(evs)])
    net_profit = profit + cap_inventory_change * float(np.mean(price))
    cap_terminal_target = np.clip(initial_cap_soc, cap_lower, cap_upper)
    per_exposure = (lambda value: value / cap_exposure_hours
                    if cap_exposure_hours > 0 else None)
    result.update(test_seed=seed, n_test=n_evs, n_censored=int(censored.sum()),
                  n_no_activity=int((~active).sum()), degradation_per_kwh=float(trained["price_degrad"]),
                  cap_violation_rate=per_exposure(cap_violation_hours),
                  cap_mean_violation=per_exposure(cap_violation_area),
                  cap_max_violation=cap_max_violation,
                  cap_low_rate=per_exposure(cap_low_hours),
                  cap_high_rate=per_exposure(cap_high_hours),
                  cap_requested_abs_kw=per_exposure(cap_requested_abs_area),
                  cap_executed_abs_kw=per_exposure(cap_executed_abs_area),
                  cap_action_clip_rate=per_exposure(cap_clipped_hours),
                  battery_requested_abs_kw=per_exposure(battery_requested_abs_area),
                  battery_executed_abs_kw=per_exposure(battery_executed_abs_area),
                  battery_action_clip_rate=per_exposure(battery_clipped_hours),
                  battery_action_abs_mean=per_exposure(battery_action_abs_area),
                  battery_action_saturation_rate=per_exposure(battery_action_saturated_hours),
                  transformer_violation_frequency=transformer_violation_steps/24,
                  transformer_overload_to_ev_allowance_percent=(
                      100.0 * transformer_excess_sum / transformer_ev_allowance_sum
                      if transformer_ev_allowance_sum > 0 else None),
                  transformer_mean_excess_kw=transformer_excess_sum/24,
                  transformer_max_excess_kw=transformer_max_excess,
                  transformer_capacity_kw=float(cap_trans),
                  transformer_definition=transformer_definition,
                  infrastructure_scenario=infrastructure_scenario,
                  transformer_scope=transformer_scope,
                  cap_boundary=[cap_lower, cap_upper],
                  cap_exposure_hours=cap_exposure_hours,
                  net_profitu=float(net_profit[included].mean()) if included.any() else None,
                  cap_terminal_soc_change=float(
                      (terminal_cap_soc[included] - initial_cap_soc[included]).mean())
                      if included.any() else None,
                  cap_terminal_target_gap=float(
                      np.abs(terminal_cap_soc[included] - cap_terminal_target[included]).mean())
                      if included.any() else None,
                  metric_scope="cost_all: mean battery cost across matched demand EVs; psoc_all: successful share across all matched demand EVs; dsoc_all_ev: mean positive shortfall across all matched demand EVs; costb: successful EVs in the matched demand cohort; psoc: successful-CS share; dsoc: legacy alias of failed-CS conditional mean shortfall "
                               "with zero for a fully successful scenario; dsoc_failed_cs/dsoc_conditional: "
                               "failed-CS conditional mean shortfall or null when no failure; dsoc_all_cs: "
                               "mean positive shortfall across all CS EVs, with zero assigned to successful CS EVs; profitu: all included EVs; "
                               "capacitor boundary metrics: duration-weighted active EV-hours")
    return result


def summarize(results):
    summary = {}
    for key in ("costb", "cost_all", "profitu", "net_profitu",
                "cap_terminal_soc_change", "cap_terminal_target_gap",
                "psoc", "psoc_all", "dsoc", "dsoc_all_cs", "dsoc_all_ev", "mean_departure_soc",
                "mean_departure_target_margin", "unreachable_cs_rate", "psoc_reachable",
                "dsoc_reachable_cs", "target_tracking_mae_cs", "excess_soc_all_cs",
                "cap_violation_rate",
                "cap_mean_violation", "cap_max_violation", "cap_low_rate",
                "cap_high_rate", "cap_requested_abs_kw", "cap_executed_abs_kw",
                "cap_action_clip_rate", "battery_requested_abs_kw",
                "battery_executed_abs_kw", "battery_action_clip_rate",
                "battery_action_abs_mean", "battery_action_saturation_rate",
                "transformer_violation_frequency", "transformer_overload_to_ev_allowance_percent",
                "transformer_mean_excess_kw", "transformer_max_excess_kw"):
        values = [r[key] for r in results if r[key] is not None]
        summary[key] = dict(
            mean=float(np.mean(values)) if values else None,
            std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
            n_valid=len(values), n_total=len(results))
    return summary


def table_summary(summary):
    units = {"costb": (1, "$"), "cost_all": (1, "$ per demand EV"),
             "profitu": (100, "10^-2 $"),
             "net_profitu": (100, "10^-2 $"),
             "cap_terminal_soc_change": (100, "% SoC"),
             "cap_terminal_target_gap": (100, "% SoC"),
             "psoc": (100, "%"), "psoc_all": (100, "%"), "dsoc": (100, "%"),
             "dsoc_all_cs": (100, "% SoC"),
             "dsoc_all_ev": (100, "% SoC"),
             "mean_departure_soc": (1, "SoC"),
             "mean_departure_target_margin": (1, "SoC"),
             "unreachable_cs_rate": (100, "% CS EVs"),
             "psoc_reachable": (100, "% reachable CS EVs"),
             "dsoc_reachable_cs": (100, "% SoC"),
             "target_tracking_mae_cs": (100, "% SoC"),
             "excess_soc_all_cs": (100, "% SoC"),
             "cap_violation_rate": (100, "% EV-hours"),
             "cap_mean_violation": (100, "% SoC"),
             "cap_max_violation": (100, "% SoC")}
    units.update({"cap_low_rate": (100, "% EV-hours"),
                  "cap_high_rate": (100, "% EV-hours"),
                  "cap_requested_abs_kw": (1, "kW"),
                  "cap_executed_abs_kw": (1, "kW"),
                  "cap_action_clip_rate": (100, "% EV-hours")})
    units.update({"battery_requested_abs_kw": (1, "kW"),
                  "battery_executed_abs_kw": (1, "kW"),
                  "battery_action_clip_rate": (100, "% EV-hours")})
    units.update({"battery_action_abs_mean": (1, "p.u."),
                  "battery_action_saturation_rate": (100, "% EV-hours")})
    units.update({"transformer_violation_frequency": (100, "% hourly intervals"),
                  "transformer_overload_to_ev_allowance_percent": (1, "% of EV-allowable capacity"),
                  "transformer_mean_excess_kw": (1, "kW"),
                  "transformer_max_excess_kw": (1, "kW")})
    return {key: dict(mean=stats["mean"] * units[key][0] if stats["mean"] is not None else None,
                      std=stats["std"] * units[key][0] if stats["std"] is not None else None,
                      n_valid=stats["n_valid"], n_total=stats["n_total"], unit=units[key][1])
            for key, stats in summary.items()}


def evaluate_repeated(trained, output, n_evs, seed, runs, metrics, cap_lower=0.1, cap_upper=0.9,
                      infrastructure_scenario="hybrid", transformer_scope="cs_only",
                      transformer_definition="total_grid", infrastructure_assignment="exclusive"):
    if runs < 2:
        raise ValueError("TEST_RUNS must be at least 2 for sample standard deviation")
    results = [evaluate(
        trained, n_evs, seed + i, cap_lower, cap_upper, infrastructure_scenario,
        transformer_scope, transformer_definition, infrastructure_assignment)
        for i in range(runs)]
    display = table_summary(summarize(results))
    compact = {key: display[key] for key in metrics}
    (output / "indicators.json").write_text(json.dumps(
        {"test_runs": runs, "table_summary": compact}, indent=2, allow_nan=False))
    for key, stats in compact.items():
        mean = f"{stats['mean']:.6f}" if stats["mean"] is not None else "N/A"
        std = f"{stats['std']:.6f}" if stats["std"] is not None else "N/A"
        print(f"{key}: {mean} +/- {std}", flush=True)

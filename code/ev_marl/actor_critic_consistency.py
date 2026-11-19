"""Scoped VX consistency fixes. Archived V2 remains unchanged.

Learning uses a membership-weighted masked critic and the executed masked
actor mixture. Shared parameters receive gradients from all active paths;
private first-layer connections receive only their owning path's gradients.
"""
from contextlib import contextmanager
import copy
import os
import numpy as np
import torch
from scipy.special import expit
from torch.nn import functional as F


def masked_forward(model, state, frozen, action=None):
    layer = model.fc1 if action is None else model.fcs1
    mask = torch.ones_like(layer.weight)
    mask[frozen[0], frozen[1]] = 0
    hidden = F.relu(F.linear(state, layer.weight * mask, layer.bias))
    if action is not None:
        hidden = torch.cat((hidden, action), dim=1)
    out = model.fc3(F.relu(model.fc2(hidden)))
    return torch.tanh(out) if action is None else out


def mixture(model, state, freezes, weights, action=None):
    return sum(weights[:, i:i+1] * masked_forward(model, state, frozen, action)
               for i, frozen in enumerate(freezes))


def learn(self, experiences):
    from ddpg_agents import GAMMA, TAU
    if not hasattr(self, "vx_diagnostics"):
        self.vx_diagnostics = []
    states, next_states, total, next_total, actions, rewards, dones = experiences[:7]
    weights = states[:, 8:8+len(self.actor_freeze)]
    next_weights = next_states[:, 8:8+len(self.actor_freeze)]
    with torch.no_grad():
        next_actions = mixture(self.actor_target, next_states, self.actor_freeze, next_weights)
        next_q = mixture(self.critic_target, next_total, self.critic_freeze, next_weights, next_actions)
        target = rewards + GAMMA * (1 - dones) * next_q
    q = mixture(self.critic_local, total, self.critic_freeze, weights, actions)
    critic_loss = F.mse_loss(q, target)
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_norm = sum(p.grad.square().sum().item() for p in self.critic_local.parameters() if p.grad is not None) ** .5
    self.critic_optimizer.step()
    self.critic_local.requires_grad_(False)
    try:
        predicted = mixture(self.actor_local, states, self.actor_freeze, weights)
        actor_loss = (-mixture(self.critic_local, total, self.critic_freeze,
                               weights, predicted).mean()
                      + getattr(self, "vx_action_regularization", 0.0)
                      * predicted.square().mean())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_norm = sum(p.grad.square().sum().item() for p in self.actor_local.parameters() if p.grad is not None) ** .5
        self.actor_optimizer.step()
    finally:
        self.critic_local.requires_grad_(True)
    self.soft_update(self.critic_local, self.critic_target, TAU)
    self.soft_update(self.actor_local, self.actor_target, TAU)
    td_error = (target-q.detach()).reshape(-1).cpu().numpy()
    episode = int(getattr(self, "vx_current_episode", 0))
    if not hasattr(self, "vx_td_error_moments"):
        self.vx_td_error_moments = {}
    moments = self.vx_td_error_moments.setdefault(episode, [0, 0.0, 0.0])
    moments[0] += int(td_error.size)
    moments[1] += float(td_error.sum())
    moments[2] += float(np.square(td_error).sum())
    self.vx_diagnostics.append((critic_loss.item(), actor_loss.item(),
                                np.abs(td_error).mean(), critic_norm, actor_norm))


def learn_td3(self, experiences):
    """TD3 update for the capacitor while retaining the masked VX networks."""
    from ddpg_agents import GAMMA, TAU
    states, next_states, total, next_total, actions, rewards, dones = experiences[:7]
    weights = states[:, 8:8+len(self.actor_freeze)]
    next_weights = next_states[:, 8:8+len(self.actor_freeze)]
    with torch.no_grad():
        next_actions = mixture(self.actor_target, next_states,
                               self.actor_freeze, next_weights)
        noise = torch.randn_like(next_actions).mul_(0.2).clamp_(-0.5, 0.5)
        next_actions = (next_actions + noise).clamp_(-1.0, 1.0)
        next_q1 = mixture(self.critic_target, next_total,
                          self.critic_freeze, next_weights, next_actions)
        next_q2 = mixture(self.critic2_target, next_total,
                          self.critic_freeze, next_weights, next_actions)
        target = rewards + GAMMA * (1 - dones) * torch.minimum(next_q1, next_q2)

    q1 = mixture(self.critic_local, total, self.critic_freeze, weights, actions)
    q2 = mixture(self.critic2_local, total, self.critic_freeze, weights, actions)
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    self.critic_optimizer.zero_grad()
    self.critic2_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    self.critic2_optimizer.step()

    self.vx_td3_updates += 1
    if self.vx_td3_updates % 2 == 0:
        self.critic_local.requires_grad_(False)
        try:
            predicted = mixture(self.actor_local, states, self.actor_freeze, weights)
            actor_loss = -mixture(self.critic_local, total, self.critic_freeze,
                                  weights, predicted).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            self.critic_local.requires_grad_(True)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.critic2_local, self.critic2_target, TAU)


def nearest(ev_id, candidates, evs, k):
    def preference(i):
        ev = evs[i]
        values = [ev.soc_exp_b, ev.price_exp_pu, *ev.w]
        if hasattr(ev, "vx_type"):
            values.append(ev.cap_c / 6.0)
        return np.array(values)
    origin = preference(ev_id)
    others = [i for i in candidates if i != ev_id]
    return sorted(others, key=lambda i: (float(np.sum((preference(i)-origin)**2)), i))[:k]


def total_states(obs, next_obs, d_obs, n_exist, k, existing, following, evs, size,
                 state_assembly="preference"):
    current = dict(zip(existing[0], obs))
    upcoming = {i: evs[i]._vx_next for i in following[0]}
    def assemble(i, pool):
        if i not in pool:
            return [0.] * size
        if state_assembly == "fixed_global":
            row = []
            for j in sorted(pool)[:k]:
                row.extend(pool[j])
        else:
            row = list(pool[i])
            for j in nearest(i, pool, evs, k-1):
                row.extend(pool[j])
        return row + [0.] * (size-len(row))
    return ([assemble(i, current) for i in existing[0]],
            [assemble(i, upcoming) for i in existing[0]])


def transformer_penalties(existing, evs, base_kw, capacity_kw, weight=2., scope="cs_only",
                          credit="proportional", battery_weight=1., capacitor_weight=3.):
    # One aggregate CS transformer, matching the existing single trans_i=0.
    # Power is averaged over the one-hour scheduling interval.
    include = lambda infra: scope == "all" or infra == 0
    b = np.array([evs[i].actions_b[-1]*duration if include(infra) else 0.
                  for i, duration, infra in zip(*existing)])
    c = np.array([evs[i].actions_c[-1]*duration if include(infra) else 0.
                  for i, duration, infra in zip(*existing)])
    net = base_kw + b.sum() + c.sum()
    excess = max(abs(net)-capacity_kw, 0.)
    if credit in ("marginal", "marginal_soc_gate", "marginal_urgency"):
        # Difference rewards assign each action its counterfactual contribution
        # to aggregate overload; a relieving action receives a negative penalty.
        penalty = lambda action: weight * (
            excess - max(abs(net-action)-capacity_kw, 0.)) / capacity_kw
        pb = np.array([penalty(x) for x in b])
        if credit == "marginal_soc_gate":
            unmet = np.array([evs[i].soc_t_b < evs[i].soc_exp_b
                              for i in existing[0]])
            pb[(pb > 0) & unmet] = 0.0
        elif credit == "marginal_urgency":
            urgency = []
            for i, duration in zip(existing[0], existing[1]):
                ev = evs[i]
                gap_energy = max(ev.soc_exp_b - ev.soc_t_b, 0.) * ev.cap_b
                required_power = gap_energy / max(ev.dur_h, duration, 1e-9)
                urgency.append(np.clip(required_power / ev.p_max_b, 0., 1.))
            urgency = np.asarray(urgency)
            pb = np.where(pb > 0, pb * (1. - urgency), pb) * (battery_weight / weight)
            return pb, np.array([penalty(x) for x in c]) * (capacitor_weight / weight)
        return pb, np.array([penalty(x) for x in c])
    direction = 1 if net >= 0 else -1
    cb, cc = np.maximum(direction*b, 0), np.maximum(direction*c, 0)
    denominator = cb.sum()+cc.sum()
    factor = weight*excess/capacity_kw/denominator if denominator > 0 else 0.
    return cb*factor, cc*factor


def feasible_capacitor_actions(existing, evs, normalized, rated_kw):
    """Map actor decisions to state-dependent feasible capacitor power."""
    actions = []
    for ev_id, duration, decision in zip(existing[0], existing[1], normalized):
        ev = evs[ev_id]
        charge_limit = min(rated_kw, max((0.99-ev.soc_t_c)*ev.cap_c/duration, 0.))
        discharge_limit = min(rated_kw, max((ev.soc_t_c-0.01)*ev.cap_c/duration, 0.))
        actions.append(float(decision) * (charge_limit if decision >= 0 else discharge_limit))
    return actions


def corrective_capacitor_decisions(existing, evs, normalized, lower, upper):
    """Point capacitor actions back toward the operating band when outside it."""
    corrected = []
    for ev_id, decision in zip(existing[0], normalized):
        soc = evs[ev_id].soc_t_c
        if soc < lower:
            decision = 1.0
        elif soc > upper:
            decision = -1.0
        corrected.append(float(decision))
    return corrected


def operating_band_capacitor_actions(existing, evs, normalized, rated_kw, lower, upper):
    """Return to the nearest operating boundary, then resume free arbitrage."""
    actions = []
    for ev_id, duration, decision in zip(existing[0], existing[1], normalized):
        ev = evs[ev_id]
        if ev.soc_t_c < lower:
            action = min((lower - ev.soc_t_c) * ev.cap_c / duration, rated_kw)
        elif ev.soc_t_c > upper:
            action = -min((ev.soc_t_c - upper) * ev.cap_c / duration, rated_kw)
        else:
            charge = min(rated_kw, max((upper-ev.soc_t_c)*ev.cap_c/duration, 0.))
            discharge = min(rated_kw, max((ev.soc_t_c-lower)*ev.cap_c/duration, 0.))
            action = float(decision) * (charge if decision >= 0 else discharge)
        actions.append(float(action))
    return actions


def feasible_battery_actions(existing, evs, normalized, rated_kw):
    """Map normalized decisions directly to the current feasible battery power."""
    actions = []
    for ev_id, duration, decision in zip(existing[0], existing[1], normalized):
        ev = evs[ev_id]
        charge_limit = min(rated_kw, max((0.99-ev.soc_t_b)*ev.cap_b/duration, 0.))
        discharge_limit = min(rated_kw, max((ev.soc_t_b-0.01)*ev.cap_b/duration, 0.))
        actions.append(float(decision) * (charge_limit if decision >= 0 else discharge_limit))
    return actions


def recoverable_battery_actions(existing, evs, normalized, rated_kw, efficiency=0.95):
    """Permit V2G only while the target remains reachable in later contacts."""
    actions = []
    for ev_id, duration, decision in zip(existing[0], existing[1], normalized):
        ev = evs[ev_id]
        remaining = max(float(getattr(ev, "_vx_remaining_contact_h", duration)), duration)
        future_contact = max(remaining - duration, 0.0)
        floor = max(0.01, float(ev.soc_exp_b)
                    - efficiency * rated_kw * future_contact / float(ev.cap_b))
        charge_limit = min(rated_kw, max((0.99-ev.soc_t_b)*ev.cap_b/duration, 0.))
        discharge_limit = min(rated_kw, max((ev.soc_t_b-floor)*ev.cap_b/duration, 0.))
        actions.append(float(decision) * (charge_limit if decision >= 0 else discharge_limit))
        ev._vx_remaining_contact_h = max(remaining - duration, 0.0)
    return actions


def scheduled_recoverable_battery_actions(existing, evs, normalized, rated_kw,
                                           enforce_required_charge=False,
                                           efficiency=0.95):
    """Delay discretionary high-price charging while keeping the target reachable."""
    actions = []
    for ev_id, duration, decision in zip(existing[0], existing[1], normalized):
        ev = evs[ev_id]
        remaining = max(float(getattr(ev, "_vx_remaining_contact_h", duration)), duration)
        future_contact = max(remaining - duration, 0.0)
        target_need = max((float(ev.soc_exp_b) - float(ev.soc_t_b))
                          * float(ev.cap_b), 0.0)
        future_charge = efficiency * rated_kw * future_contact
        required_now = max(target_need - future_charge, 0.0) / max(
            efficiency * duration, 1e-12)
        ev._vx_required_charge_kw = min(required_now, rated_kw)
        ev._vx_planned_charge_kw = min(
            target_need / max(efficiency * remaining, 1e-12), rated_kw)
        ev._vx_charge_urgency = target_need / max(remaining, 1e-12)
        floor = max(0.01, float(ev.soc_exp_b)
                    - future_charge / float(ev.cap_b))
        charge_limit = min(rated_kw, max((0.99-ev.soc_t_b)*ev.cap_b/duration, 0.))
        discharge_limit = min(rated_kw, max((ev.soc_t_b-floor)*ev.cap_b/duration, 0.))
        action = float(decision) * (charge_limit if decision >= 0 else discharge_limit)
        current_price = float(getattr(ev, "_vx_current_price_pu", 0.0))
        if action > 0 and current_price > float(ev.price_exp_pu) and future_charge >= target_need:
            action = 0.0
        if enforce_required_charge and required_now > 0:
            action = max(action, min(required_now, charge_limit))
        actions.append(float(action))
        ev._vx_remaining_contact_h = max(remaining - duration, 0.0)
    return actions


def project_cs_actions_to_transformer(existing, evs, battery, capacitor, capacity_kw,
                                      prioritize_required=False):
    """Project CS power onto the limit, optionally preserving target-critical charge."""
    cs_indices = [i for i, infra in enumerate(existing[2]) if infra == 0]
    positive = sum(max(battery[i], 0.0) + max(capacitor[i], 0.0) for i in cs_indices)
    negative = sum(min(battery[i], 0.0) + min(capacitor[i], 0.0) for i in cs_indices)
    allowed_positive = max(float(capacity_kw) - negative, 0.0)
    if positive <= allowed_positive or positive <= 0:
        return battery, capacitor
    if not prioritize_required:
        scale = allowed_positive / positive
        battery = [value * scale if i in cs_indices and value > 0 else value
                   for i, value in enumerate(battery)]
        capacitor = [value * scale if i in cs_indices and value > 0 else value
                     for i, value in enumerate(capacitor)]
        return battery, capacitor
    required = [0.0] * len(battery)
    for i in cs_indices:
        ev = evs[existing[0][i]]
        required[i] = min(max(float(getattr(ev, "_vx_required_charge_kw", 0.0)), 0.0),
                          max(battery[i], 0.0))
    required_total = sum(required)
    if required_total >= allowed_positive:
        scale = allowed_positive / max(required_total, 1e-12)
        battery = [required[i] * scale if i in cs_indices and battery[i] > 0 else battery[i]
                   for i in range(len(battery))]
        capacitor = [0.0 if i in cs_indices and value > 0 else value
                     for i, value in enumerate(capacitor)]
        return battery, capacitor
    discretionary = positive - required_total
    scale = (allowed_positive - required_total) / max(discretionary, 1e-12)
    battery = [(required[i] + (value-required[i])*scale)
               if i in cs_indices and value > 0 else value
               for i, value in enumerate(battery)]
    capacitor = [value * scale if i in cs_indices and value > 0 else value
                 for i, value in enumerate(capacitor)]
    return battery, capacitor


def allocate_cs_headroom_by_urgency(existing, evs, battery, capacitor, capacity_kw,
                                    preference_aware=False):
    """Reserve CS headroom for target-critical battery charging, then flexibility."""
    cs_indices = [i for i, infra in enumerate(existing[2]) if infra == 0]
    durations = existing[1]
    negative = sum((min(battery[i], 0.0) + min(capacitor[i], 0.0)) * durations[i]
                   for i in cs_indices)
    budget = max(float(capacity_kw) - negative, 0.0)
    out_b, out_c = list(battery), list(capacitor)
    for i in cs_indices:
        if out_b[i] > 0:
            out_b[i] = 0.0
        if out_c[i] > 0:
            out_c[i] = 0.0

    def priority(i):
        ev = evs[existing[0][i]]
        urgency = float(getattr(ev, "_vx_charge_urgency", 0.0))
        return urgency * ((0.25 + 0.75 * float(ev.w[0])) if preference_aware else 1.0)
    ordered = sorted(cs_indices, key=priority, reverse=True)
    mandatory = {}
    for i in ordered:
        ev = evs[existing[0][i]]
        planned = min(float(getattr(ev, "_vx_planned_charge_kw", 0.0)),
                      max((0.99-ev.soc_t_b)*ev.cap_b/durations[i], 0.0))
        energy = min(planned * durations[i], budget)
        mandatory[i] = energy / max(durations[i], 1e-12)
        out_b[i] = mandatory[i]
        budget -= energy

    flexible_b = {}
    for i in cs_indices:
        request = max(battery[i] - mandatory.get(i, 0.0), 0.0) * durations[i]
        if preference_aware:
            ev = evs[existing[0][i]]
            low_price = max(1.0 - float(getattr(ev, "_vx_current_price_pu", 0.0)), 0.0)
            request *= 0.25 + 0.75 * float(ev.w[1]) * low_price
        flexible_b[i] = request
    total_flexible_b = sum(flexible_b.values())
    if total_flexible_b > 0 and budget > 0:
        scale = min(1.0, budget / total_flexible_b)
        for i in cs_indices:
            out_b[i] += flexible_b[i] * scale / max(durations[i], 1e-12)
        budget -= total_flexible_b * scale

    flexible_c = {i: max(capacitor[i], 0.0) * durations[i] for i in cs_indices}
    total_flexible_c = sum(flexible_c.values())
    if total_flexible_c > 0 and budget > 0:
        scale = min(1.0, budget / total_flexible_c)
        for i in cs_indices:
            out_c[i] = flexible_c[i] * scale / max(durations[i], 1e-12)
    return out_b, out_c


def next_state(idx_h_next, n_price, price, price_pu, load, load_pu, evs, done,
               energy_ch, energy_dch, price_seq, existing, following, price_min,
               price_max, price_exp_min, price_exp_max, trade_min, trade_max, degradation, groups,
               soc_done=None):
    from training_utils import refresh_price_nonEVload
    from constants import cap_trans
    global _episode_constraint_current
    if _episode_constraint_current is None:
        _episode_constraint_current = {
            "soc_sum": 0.0, "soc_count": 0,
            "transformer_sum": 0.0, "transformer_violations": 0, "hour_count": 0,
        }
    _, pp, _, lp = refresh_price_nonEVload(idx_h_next, n_price, price, price_pu, load, load_pu)
    for i, duration, infra in zip(*following):
        evs[i]._vx_next = evs[i].get_fea_b_and_c_iddpg(pp, lp, infra, duration, evs[i].dur_h)[0]
    observations, rb, rc = [], [], []
    for i, duration, infra in zip(*existing):
        ev = evs[i]
        base_dim = 19 if os.environ.get("VX_HETEROGENEOUS_STRESS") == "1" else 18
        state_dim = (base_dim - 1 if os.environ.get("VX_NO_MASK_SPECIALIZATION") == "1"
                     else base_dim + groups)
        observations.append(ev._vx_next if i in following[0] else [0.] * state_dim)
        battery_done = i in soc_done if soc_done is not None else i in done
        rb.append(ev.get_reward_b(price_seq[0], battery_done, price_min, price_max,
                                 degradation, trade_max, trade_min, infra)[0])
        rc.append(ev.get_reward_c2(i in done, price_seq[0], price_max, price_min)[0])
    base_power = (0.0 if _transformer_definition == "ev_only"
                  else load[(idx_h_next-1) % n_price])
    pb, pc = transformer_penalties(existing, evs, base_power,
                                   cap_trans, scope=_transformer_scope,
                                   credit=_transformer_credit,
                                   battery_weight=_transformer_battery_weight,
                                   capacitor_weight=_transformer_capacitor_weight)
    if _adaptive_transformer_dual:
        pb, pc = pb * _transformer_dual_lambda, pc * _transformer_dual_lambda
    rb, rc = (np.asarray(rb)-pb).tolist(), (np.asarray(rc)-pc).tolist()
    include = lambda infra: _transformer_scope == "all" or infra == 0
    battery_power = sum(
        evs[i].actions_b[-1] * duration if include(infra) else 0.0
        for i, duration, infra in zip(*existing))
    capacitor_power = sum(
        evs[i].actions_c[-1] * duration if include(infra) else 0.0
        for i, duration, infra in zip(*existing))
    net_power = base_power + battery_power + capacitor_power
    normalized_excess = max(abs(net_power) - cap_trans, 0.0) / cap_trans
    _episode_constraint_current["transformer_sum"] += normalized_excess ** 2
    _episode_constraint_current["transformer_violations"] += int(normalized_excess > 0)
    _episode_constraint_current["hour_count"] += 1
    for i in done:
        target = max(float(evs[i].soc_exp_b), 1e-12)
        shortfall = max(target - float(evs[i].soc_t_b), 0.0) / target
        _episode_constraint_current["soc_sum"] += shortfall ** 2
        _episode_constraint_current["soc_count"] += 1
    by_mask = [[] for _ in range(groups)]
    for i, reward in zip(existing[0], rb):
        by_mask[evs[i].id_mask].append(reward)
    return observations, observations, rb, rc, energy_ch, energy_dch, by_mask


_transformer_scope = "cs_only"
_transformer_definition = "total_grid"
_transformer_credit = "proportional"
_transformer_battery_weight = 1.0
_transformer_capacitor_weight = 3.0
_episode_constraint_history = {"soc": [], "transformer": []}
_episode_constraint_current = None
_adaptive_transformer_dual = False
_transformer_dual_lambda = 1.0
_transformer_dual_lr = 1.0
_transformer_dual_target = 0.05


def reset_episode_constraint_metrics():
    """Reset training-only diagnostics used by the common evaluation return."""
    global _episode_constraint_history, _episode_constraint_current
    _episode_constraint_history = {"soc": [], "transformer": []}
    _episode_constraint_current = None


def get_episode_constraint_metrics(expected_episodes=None):
    """Return per-episode penalties without changing the learner reward."""
    result = {key: np.asarray(values, dtype=float)
              for key, values in _episode_constraint_history.items()}
    if expected_episodes is not None:
        for key, values in result.items():
            if len(values) != expected_episodes:
                raise RuntimeError(
                    f"constraint metric {key} has {len(values)} episodes; "
                    f"expected {expected_episodes}")
    return result


def finish_episode_constraint_metrics():
    """Finalize one episode even when no EV is active in the last hour."""
    global _episode_constraint_current
    current = _episode_constraint_current or {
        "soc_sum": 0.0, "soc_count": 0,
        "transformer_sum": 0.0, "transformer_violations": 0, "hour_count": 0,
    }
    _episode_constraint_history["soc"].append(
        current["soc_sum"] / max(current["soc_count"], 1))
    _episode_constraint_history["transformer"].append(
        current["transformer_sum"] / max(current["hour_count"], 1))
    global _transformer_dual_lambda
    if _adaptive_transformer_dual:
        violation_rate = current["transformer_violations"] / max(current["hour_count"], 1)
        _transformer_dual_lambda = float(np.clip(
            _transformer_dual_lambda
            + _transformer_dual_lr * (violation_rate - _transformer_dual_target),
            0.0, 10.0))
    _episode_constraint_current = None




@contextmanager
def corrected_vx(transformer_scope="cs_only", battery_action_mapping="legacy",
                 battery_action_regularization=0.0,
                 capacitor_operating_band=None, capacitor_td3=False,
                 capacitor_td3_actor_lr=None,
                 transformer_credit="proportional",
                 transformer_battery_weight=1.0,
                 transformer_capacitor_weight=3.0,
                 capacitor_td3_critic_lr=None, capacitor_gradient_steps=1,
                 capacitor_noise_clip=0.02, battery_actor_lr=None,
                 battery_critic_lr=None, battery_gradient_steps=1,
                 state_assembly="preference", transformer_definition="total_grid"):
    if transformer_scope not in ("cs_only", "all"):
        raise ValueError("transformer_scope must be 'cs_only' or 'all'")
    if transformer_definition not in ("total_grid", "ev_only"):
        raise ValueError("transformer_definition must be 'total_grid' or 'ev_only'")
    if battery_action_mapping not in ("legacy", "feasible", "recoverable",
                                      "price_recoverable", "scheduled_recoverable",
                                      "scheduled_transformer", "scheduled_transformer_priority",
                                      "scheduled_lagrangian", "scheduled_headroom",
                                      "preference_headroom"):
        raise ValueError("invalid battery_action_mapping")
    if transformer_credit not in ("proportional", "marginal", "marginal_soc_gate",
                                  "marginal_urgency"):
        raise ValueError("invalid transformer_credit")
    if min(transformer_battery_weight, transformer_capacitor_weight) < 0:
        raise ValueError("transformer weights must be non-negative")
    if not np.isfinite(battery_action_regularization) or battery_action_regularization < 0:
        raise ValueError("battery_action_regularization must be finite and non-negative")
    if (capacitor_operating_band is not None
            and not 0 < capacitor_operating_band[0] < capacitor_operating_band[1] < 1):
        raise ValueError("capacitor_operating_band must lie inside (0, 1)")
    if (capacitor_td3_actor_lr is not None
            and (not np.isfinite(capacitor_td3_actor_lr) or capacitor_td3_actor_lr <= 0)):
        raise ValueError("capacitor_td3_actor_lr must be positive")
    if (capacitor_td3_critic_lr is not None
            and (not np.isfinite(capacitor_td3_critic_lr) or capacitor_td3_critic_lr <= 0)):
        raise ValueError("capacitor_td3_critic_lr must be positive")
    if not isinstance(capacitor_gradient_steps, int) or capacitor_gradient_steps < 1:
        raise ValueError("capacitor_gradient_steps must be a positive integer")
    if battery_actor_lr is not None and battery_actor_lr <= 0:
        raise ValueError("battery_actor_lr must be positive")
    if battery_critic_lr is not None and battery_critic_lr <= 0:
        raise ValueError("battery_critic_lr must be positive")
    if not isinstance(battery_gradient_steps, int) or battery_gradient_steps < 1:
        raise ValueError("battery_gradient_steps must be a positive integer")
    if not np.isfinite(capacitor_noise_clip) or capacitor_noise_clip <= 0:
        raise ValueError("capacitor_noise_clip must be positive")
    if state_assembly not in ("preference", "fixed_global"):
        raise ValueError("state_assembly must be 'preference' or 'fixed_global'")
    global _transformer_scope, _transformer_credit, _transformer_definition
    global _transformer_battery_weight, _transformer_capacitor_weight
    global _adaptive_transformer_dual, _transformer_dual_lambda
    previous_scope = _transformer_scope
    previous_definition = _transformer_definition
    previous_credit = _transformer_credit
    previous_battery_weight = _transformer_battery_weight
    previous_capacitor_weight = _transformer_capacitor_weight
    previous_adaptive_dual = _adaptive_transformer_dual
    previous_dual_lambda = _transformer_dual_lambda
    _transformer_scope = transformer_scope
    _transformer_definition = transformer_definition
    _transformer_credit = transformer_credit
    _transformer_battery_weight = transformer_battery_weight
    _transformer_capacitor_weight = transformer_capacitor_weight
    _adaptive_transformer_dual = battery_action_mapping == "scheduled_lagrangian"
    _transformer_dual_lambda = 1.0
    import training_utils as ft
    import ddpg_agents as agents
    from environment_entities import EV, custom_mapping
    original_reward, original_features = EV.get_reward_b, EV.get_fea_b_and_c_iddpg
    old_next, old_total = ft.get_next_statev11, ft.get_total_s_and_next
    old_choose_train, old_choose_test = ft.choose_actions_mask3, ft.choose_actions_mask5test
    old_real_actions = ft.get_real_actions2
    old_learn = [cls.learn_mask3 for cls in (agents.Agent_b, agents.Agent_c)]
    old_capacitor_step = agents.Agent_c.step_mask3
    old_capacitor_init = agents.Agent_c.__init__
    old_battery_step = agents.Agent_b.step_mask3
    old_battery_init = agents.Agent_b.__init__
    old_regularization = getattr(agents.Agent_b, "vx_action_regularization", None)
    old_cap_noise_clip = getattr(agents.Agent_c, "vx_noise_clip", None)
    agents.Agent_b.vx_action_regularization = battery_action_regularization
    agents.Agent_c.vx_noise_clip = capacitor_noise_clip
    def battery_init(self, *args, **kwargs):
        old_battery_init(self, *args, **kwargs)
        if battery_actor_lr is not None:
            self.actor_optimizer = torch.optim.Adam(
                self.actor_local.parameters(), lr=battery_actor_lr,
                weight_decay=agents.WEIGHT_DECAY)
        if battery_critic_lr is not None:
            self.critic_optimizer = torch.optim.Adam(
                self.critic_local.parameters(), lr=battery_critic_lr,
                weight_decay=agents.WEIGHT_DECAY)
    def battery_step(self, state, state_next, state_total, state_total_next,
                     action, reward, done, id_mask, weight_ls):
        if battery_gradient_steps == 1:
            return old_battery_step(
                self, state, state_next, state_total, state_total_next,
                action, reward, done, id_mask, weight_ls)
        self.memory.add(state, state_next, state_total, state_total_next,
                        action, reward, done, 1, 1, 1, weight_ls, id_mask)
        if len(self.memory) > agents.BATCH_SIZE:
            for _ in range(battery_gradient_steps):
                self.learn_mask3(self.memory.sample())
    def capacitor_init(self, *args, **kwargs):
        old_capacitor_init(self, *args, **kwargs)
        if capacitor_td3_actor_lr is not None:
            self.actor_optimizer = torch.optim.Adam(
                self.actor_local.parameters(), lr=capacitor_td3_actor_lr,
                weight_decay=agents.WEIGHT_DECAY_C)
        if capacitor_td3_critic_lr is not None:
            self.critic_optimizer = torch.optim.Adam(
                self.critic_local.parameters(), lr=capacitor_td3_critic_lr,
                weight_decay=agents.WEIGHT_DECAY_C)
        if capacitor_td3:
            self.critic2_local = copy.deepcopy(self.critic_local)
            self.critic2_target = copy.deepcopy(self.critic_target)
            self.critic2_optimizer = torch.optim.Adam(
                self.critic2_local.parameters(),
                lr=capacitor_td3_critic_lr or agents.LR_CRITIC_C,
                weight_decay=agents.WEIGHT_DECAY_C)
            self.vx_td3_updates = 0
    def capacitor_step(self, state, state_next, state_total, state_total_next,
                       action, reward, done, id_mask, weight_ls):
        if capacitor_gradient_steps == 1:
            return old_capacitor_step(
                self, state, state_next, state_total, state_total_next,
                action, reward, done, id_mask, weight_ls)
        self.memory.add(state, state_next, state_total, state_total_next,
                        action, reward, done, 1, 1, 1, weight_ls, id_mask)
        if len(self.memory) > agents.BATCH_SIZE:
            for _ in range(capacitor_gradient_steps):
                self.learn_mask3(self.memory.sample())
    def reward(self, price_t, done, pmin, pmax, degradation, tmax, tmin, infra):
        if infra == 0:
            return original_reward(self, price_t, done, pmin, pmax, degradation, tmax, tmin, infra)
        action = self.actions_b_pu[-1]
        soc_before = self.soc_t_b - self.actions_b[-1] * self.hr_b[-1] / self.cap_b
        price_pu = (price_t - pmin) / (pmax - pmin)
        r_soc_below = np.interp(action, (-1, 1), (-1, 0)) - 1
        r_price_below = (action * (1 - price_pu) - 1) / 2
        delta = self.price_exp-price_t if action > 0 else price_t-self.price_exp-degradation
        r_price_above = abs(action)*custom_mapping(delta, tmax, tmin)
        from reward_shaping import get_active_soc_tau, record_battery_components
        gate = expit((soc_before - self.soc_exp_b) / get_active_soc_tau())
        r_soc = (1 - gate) * r_soc_below
        r_price = (1 - gate) * r_price_below + gate * r_price_above
        weighted_soc = self.w[0] * r_soc
        weighted_price = self.w[1] * r_price
        record_battery_components(weighted_soc, weighted_price)
        return weighted_soc + weighted_price, action
    def features(self, prices, loads, infra, duration, remaining):
        self._vx_current_price_pu = float(prices[0])
        return original_features(self, prices, loads, infra, duration, remaining)
    def choose_train(*args, **kwargs):
        result = list(old_choose_train(*args, **kwargs))
        if battery_action_mapping == "feasible":
            result[0] = feasible_battery_actions(args[0], args[1], result[2], args[6])
        elif battery_action_mapping == "recoverable":
            result[0] = recoverable_battery_actions(args[0], args[1], result[2], args[6])
        elif battery_action_mapping in ("price_recoverable", "scheduled_recoverable",
                                        "scheduled_transformer", "scheduled_transformer_priority",
                                        "scheduled_lagrangian", "scheduled_headroom",
                                        "preference_headroom"):
            result[0] = scheduled_recoverable_battery_actions(
                args[0], args[1], result[2], args[6],
                enforce_required_charge=battery_action_mapping in
                ("scheduled_recoverable", "scheduled_transformer",
                 "scheduled_transformer_priority", "scheduled_lagrangian",
                 "scheduled_headroom", "preference_headroom"))
        cap_decision = result[3]
        if capacitor_operating_band is not None:
            cap_decision = corrective_capacitor_decisions(
                args[0], args[1], cap_decision, *capacitor_operating_band)
        if capacitor_operating_band is None:
            result[1] = feasible_capacitor_actions(args[0], args[1], cap_decision, args[7])
        else:
            result[1] = operating_band_capacitor_actions(
                args[0], args[1], cap_decision, args[7], *capacitor_operating_band)
        if battery_action_mapping in ("scheduled_transformer", "scheduled_transformer_priority"):
            from constants import cap_trans
            result[0], result[1] = project_cs_actions_to_transformer(
                args[0], args[1], result[0], result[1], cap_trans,
                prioritize_required=battery_action_mapping == "scheduled_transformer_priority")
        elif battery_action_mapping in ("scheduled_headroom", "preference_headroom"):
            from constants import cap_trans
            result[0], result[1] = allocate_cs_headroom_by_urgency(
                args[0], args[1], result[0], result[1], cap_trans,
                preference_aware=battery_action_mapping == "preference_headroom")
        return tuple(result)
    def choose_test(*args, **kwargs):
        result = list(old_choose_test(*args, **kwargs))
        if battery_action_mapping == "feasible":
            result[0] = feasible_battery_actions(args[0], args[1], result[2], args[6])
        elif battery_action_mapping == "recoverable":
            result[0] = recoverable_battery_actions(args[0], args[1], result[2], args[6])
        elif battery_action_mapping in ("price_recoverable", "scheduled_recoverable",
                                        "scheduled_transformer", "scheduled_transformer_priority",
                                        "scheduled_lagrangian", "scheduled_headroom",
                                        "preference_headroom"):
            result[0] = scheduled_recoverable_battery_actions(
                args[0], args[1], result[2], args[6],
                enforce_required_charge=battery_action_mapping in
                ("scheduled_recoverable", "scheduled_transformer",
                 "scheduled_transformer_priority", "scheduled_lagrangian",
                 "scheduled_headroom", "preference_headroom"))
        cap_decision = result[3]
        if capacitor_operating_band is not None:
            cap_decision = corrective_capacitor_decisions(
                args[0], args[1], cap_decision, *capacitor_operating_band)
        if capacitor_operating_band is None:
            result[1] = feasible_capacitor_actions(args[0], args[1], cap_decision, args[7])
        else:
            result[1] = operating_band_capacitor_actions(
                args[0], args[1], cap_decision, args[7], *capacitor_operating_band)
        if battery_action_mapping in ("scheduled_transformer", "scheduled_transformer_priority"):
            from constants import cap_trans
            result[0], result[1] = project_cs_actions_to_transformer(
                args[0], args[1], result[0], result[1], cap_trans,
                prioritize_required=battery_action_mapping == "scheduled_transformer_priority")
        elif battery_action_mapping in ("scheduled_headroom", "preference_headroom"):
            from constants import cap_trans
            result[0], result[1] = allocate_cs_headroom_by_urgency(
                args[0], args[1], result[0], result[1], cap_trans,
                preference_aware=battery_action_mapping == "preference_headroom")
        return tuple(result)
    def real_actions(existing, evs, actions_b, actions_c):
        if battery_action_mapping in ("feasible", "recoverable", "price_recoverable",
                                      "scheduled_recoverable", "scheduled_transformer",
                                      "scheduled_transformer_priority", "scheduled_lagrangian",
                                      "scheduled_headroom", "preference_headroom"):
            battery = list(actions_b)
        else:
            battery, _, evs = old_real_actions(existing, evs, actions_b, actions_c)
        # Capacitor actions were already mapped to the exact feasible interval.
        return battery, list(actions_c), evs
    try:
        reset_episode_constraint_metrics()
        EV.get_reward_b, EV.get_fea_b_and_c_iddpg = reward, features
        def assembled_total_states(*args, **kwargs):
            return total_states(*args, **kwargs, state_assembly=state_assembly)
        ft.get_next_statev11, ft.get_total_s_and_next = next_state, assembled_total_states
        ft.choose_actions_mask3, ft.choose_actions_mask5test = choose_train, choose_test
        ft.get_real_actions2 = real_actions
        agents.Agent_c.__init__ = capacitor_init
        agents.Agent_c.step_mask3 = capacitor_step
        agents.Agent_b.__init__ = battery_init
        agents.Agent_b.step_mask3 = battery_step
        agents.Agent_b.learn_mask3 = learn
        agents.Agent_c.learn_mask3 = learn_td3 if capacitor_td3 else learn
        yield
    finally:
        _transformer_scope = previous_scope
        _transformer_definition = previous_definition
        _transformer_credit = previous_credit
        _transformer_battery_weight = previous_battery_weight
        _transformer_capacitor_weight = previous_capacitor_weight
        _adaptive_transformer_dual = previous_adaptive_dual
        _transformer_dual_lambda = previous_dual_lambda
        EV.get_reward_b, EV.get_fea_b_and_c_iddpg = original_reward, original_features
        ft.get_next_statev11, ft.get_total_s_and_next = old_next, old_total
        ft.choose_actions_mask3, ft.choose_actions_mask5test = old_choose_train, old_choose_test
        ft.get_real_actions2 = old_real_actions
        agents.Agent_c.__init__ = old_capacitor_init
        agents.Agent_c.step_mask3 = old_capacitor_step
        agents.Agent_b.__init__ = old_battery_init
        agents.Agent_b.step_mask3 = old_battery_step
        for cls, method in zip((agents.Agent_b, agents.Agent_c), old_learn):
            cls.learn_mask3 = method
        if old_regularization is None:
            delattr(agents.Agent_b, "vx_action_regularization")
        else:
            agents.Agent_b.vx_action_regularization = old_regularization
        if old_cap_noise_clip is None:
            delattr(agents.Agent_c, "vx_noise_clip")
        else:
            agents.Agent_c.vx_noise_clip = old_cap_noise_clip

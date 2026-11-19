"""Optional V2 reward changes, scoped to a training run."""
from contextlib import contextmanager
import numpy as np
from scipy.special import expit


_battery_component_audit = None
_active_soc_tau = 0.03


def get_active_soc_tau():
    return _active_soc_tau


def record_battery_components(weighted_soc, weighted_price, weighted_terminal=0.0):
    """Record diagnostics without changing the reward returned to the learner."""
    if _battery_component_audit is None:
        return
    _battery_component_audit["n"] += 1
    _battery_component_audit["abs_r_soc"] += abs(weighted_soc)
    _battery_component_audit["abs_r_price"] += abs(weighted_price)
    _battery_component_audit["abs_terminal_target_penalty"] += abs(weighted_terminal)


def boundary_penalty(soc, lower, upper):
    """Dimensionless squared excursions beyond the SoC safety buffer."""
    return ((max(lower - soc, 0.0) / lower) ** 2
            + (max(soc - upper, 0.0) / (1.0 - upper)) ** 2)


@contextmanager
def reward_mode(mode, tau, penalty_weight, lower, upper,
                progress_weight=5.0, cost_weight=1.0,
                action_weight=0.05, terminal_weight=20.0,
                capacitor_scale=1.0):
    global _active_soc_tau
    previous_tau = _active_soc_tau
    _active_soc_tau = tau
    from environment_entities import EV, custom_mapping
    original_b, original_c = EV.get_reward_b, EV.get_reward_c2

    def battery(self, price_t, flag_done, price_min, price_max, price_degrad,
                price_trade_max, price_trade_min, infra_i):
        action = self.actions_b_pu[-1]
        soc_before = self.soc_t_b - self.actions_b[-1] * self.hr_b[-1] / self.cap_b
        price_pu = (price_t - price_min) / (price_max - price_min)
        # Preference semantics: every soft target-service term belongs to
        # r_soc, while every signed economic term belongs to r_price.
        # Physical feasibility and transformer penalties are imposed outside
        # this function and remain independent of driver preference.
        r_soc_below = np.interp(action, (-1, 1), (-1, 0)) - 1
        r_price_below = (action * (1 - price_pu) - 1) / 2
        delta = (self.price_exp - price_t if action > 0
                 else price_t - self.price_exp - price_degrad)
        r_price_above = abs(action) * custom_mapping(
            delta, price_trade_max, price_trade_min)
        gate = expit((soc_before - self.soc_exp_b) / tau)
        r_soc = (1 - gate) * r_soc_below
        r_price = ((1 - gate) * r_price_below + gate * r_price_above)
        terminal_target_penalty = 0.0
        if flag_done and self.soc_t_b < self.soc_exp_b:
            terminal_target_penalty = 10 * (self.soc_exp_b - self.soc_t_b) / self.soc_exp_b
            r_soc -= terminal_target_penalty
        reward = self.w[0] * r_soc + self.w[1] * r_price
        record_battery_components(self.w[0] * r_soc, self.w[1] * r_price,
                                  self.w[0] * terminal_target_penalty)
        return reward, action

    def capacitor(self, flag_done, price_t, price_max, price_min):
        reward, action = original_c(self, flag_done, price_t, price_max, price_min)
        return reward - penalty_weight * boundary_penalty(self.soc_t_c, lower, upper), action

    def capacitor_executed(self, flag_done, price_t, price_max, price_min):
        """Price reward uses executed power; the replay action remains the actor command."""
        action = self.actions_c_pu[-1]
        executed_pu = self.actions_c[-1] / self.p_max_c
        price_signal = custom_mapping(
            self.price_exp - price_t,
            self.price_exp - price_min,
            self.price_exp - price_max,
        )
        reward = executed_pu * price_signal
        return reward - penalty_weight * boundary_penalty(self.soc_t_c, lower, upper), action

    def capacitor_feasible_fraction(self, flag_done, price_t, price_max, price_min):
        """Normalize executed power by the available power in its direction."""
        action = self.actions_c_pu[-1]
        executed = self.actions_c[-1]
        duration = self.hr_c[-1]
        soc_before = self.soc_t_c - executed * duration / self.cap_c
        if executed > 0:
            available = min(self.p_max_c,
                            max((0.99 - soc_before) * self.cap_c / duration, 0.0))
        elif executed < 0:
            available = min(self.p_max_c,
                            max((soc_before - 0.01) * self.cap_c / duration, 0.0))
        else:
            available = 0.0
        executed_fraction = executed / available if available > 1e-12 else 0.0
        price_signal = custom_mapping(
            self.price_exp - price_t,
            self.price_exp - price_min,
            self.price_exp - price_max,
        )
        reward = executed_fraction * price_signal
        return reward - penalty_weight * boundary_penalty(self.soc_t_c, lower, upper), action

    def capacitor_recovery(self, flag_done, price_t, price_max, price_min):
        """Reward realized arbitrage and progress toward the capacitor safety band."""
        action = self.actions_c_pu[-1]
        executed = self.actions_c[-1]
        duration = self.hr_c[-1]
        soc_before = self.soc_t_c - executed * duration / self.cap_c
        if executed > 0:
            available = min(self.p_max_c,
                            max((0.99 - soc_before) * self.cap_c / duration, 0.0))
        elif executed < 0:
            available = min(self.p_max_c,
                            max((soc_before - 0.01) * self.cap_c / duration, 0.0))
        else:
            available = 0.0
        executed_fraction = executed / available if available > 1e-12 else 0.0
        price_signal = custom_mapping(
            self.price_exp - price_t,
            self.price_exp - price_min,
            self.price_exp - price_max,
        )
        safety_progress = (boundary_penalty(soc_before, lower, upper)
                           - boundary_penalty(self.soc_t_c, lower, upper))
        return executed_fraction * price_signal + penalty_weight * safety_progress, action

    def capacitor_profit(self, flag_done, price_t, price_max, price_min):
        """Reward the same realized cash flow reported by ``profitu``."""
        action = self.actions_c_pu[-1]
        energy = self.actions_c[-1] * self.hr_c[-1]
        reward = -energy * price_t / (self.cap_c * price_max)
        return reward, action

    def capacitor_arbitrage(self, flag_done, price_t, price_max, price_min):
        """Reward realized cash flow after marking stored energy at expected price."""
        action = self.actions_c_pu[-1]
        energy = self.actions_c[-1] * self.hr_c[-1]
        reward = -capacitor_scale * energy * (price_t - self.price_exp) / (
            self.cap_c * (price_max - price_min))
        return reward, action

    def capacitor_arbitrage_terminal(self, flag_done, price_t, price_max, price_min):
        reward, action = capacitor_arbitrage(
            self, flag_done, price_t, price_max, price_min)
        if flag_done:
            target = np.clip(self.soc_ini_c, lower, upper)
            reward -= terminal_weight * (self.soc_t_c - target) ** 2
        return reward, action

    def capacitor_arbitrage_urgency(self, flag_done, price_t, price_max, price_min):
        """Gradually increase the inventory-return signal near departure."""
        reward, action = capacitor_arbitrage(
            self, flag_done, price_t, price_max, price_min)
        duration = self.hr_c[-1]
        urgency = duration / max(self.dur_h + duration, duration)
        target = np.clip(self.soc_ini_c, lower, upper)
        reward -= terminal_weight * urgency * (self.soc_t_c - target) ** 2
        return reward, action

    def progress_battery(self, price_t, flag_done, price_min, price_max,
                         price_degrad, price_trade_max, price_trade_min, infra_i):
        if infra_i != 0:
            return original_b(self, price_t, flag_done, price_min, price_max,
                              price_degrad, price_trade_max, price_trade_min, infra_i)
        action = float(self.actions_b_pu[-1])
        duration = float(self.hr_b[-1])
        energy = float(self.actions_b[-1]) * duration
        soc_before = self.soc_t_b - energy / self.cap_b
        gap_before = max(self.soc_exp_b - soc_before, 0.0)
        gap_after = max(self.soc_exp_b - self.soc_t_b, 0.0)
        remaining_before = max(float(self.dur_h) + duration, duration)
        required_power = gap_before * self.cap_b / remaining_before
        urgency = min(required_power / self.p_max_b, 1.0)
        progress = gap_before - gap_after
        price_pu = (price_t - price_min) / (price_max - price_min)
        signed_cost = price_pu * energy / self.cap_b
        reward = (progress_weight * (1.0 + 4.0 * urgency) * progress
                  - cost_weight * signed_cost - action_weight * action ** 2)
        if flag_done and gap_after > 0:
            reward -= terminal_weight * gap_after ** 2
        return reward, action

    def weighted_tradeoff_battery(self, price_t, flag_done, price_min, price_max,
                                  price_degrad, price_trade_max, price_trade_min, infra_i):
        """Explicit SoC-service versus price-trading preference objective."""
        if infra_i != 0:
            return original_b(self, price_t, flag_done, price_min, price_max,
                              price_degrad, price_trade_max, price_trade_min, infra_i)
        action = float(self.actions_b_pu[-1])
        executed = float(self.actions_b[-1])
        duration = float(self.hr_b[-1])
        soc_before = self.soc_t_b - executed * duration / self.cap_b
        gap_before = max(self.soc_exp_b - soc_before, 0.0)
        gap_after = max(self.soc_exp_b - self.soc_t_b, 0.0)
        max_step = max(self.p_max_b * duration / self.cap_b, 1e-12)
        soc_progress = (gap_before - gap_after) / max_step
        threshold_advantage = (self.price_exp - price_t) / (price_max - price_min)
        executed_fraction = executed / max(self.p_max_b, 1e-12)
        price_trade = executed_fraction * threshold_advantage
        w_soc, w_price = map(float, self.w)
        reward = w_soc * soc_progress + w_price * price_trade
        if flag_done and gap_after > 0:
            reward -= 10.0 * w_soc * gap_after / max(self.soc_exp_b, 1e-12)
        return reward, action

    def li_cost_only_battery(self, price_t, flag_done, price_min, price_max,
                             price_degrad, price_trade_max, price_trade_min, infra_i):
        """Li-style economic reward; departure energy is an external CMDP constraint in Li et al."""
        energy = float(self.actions_b[-1]) * float(self.hr_b[-1])
        reward = -energy * float(price_t) / (self.cap_b * max(price_max, 1e-12))
        return reward, self.actions_b_pu[-1]

    def ye_profit_service_battery(self, price_t, flag_done, price_min, price_max,
                                  price_degrad, price_trade_max, price_trade_min, infra_i):
        """Ye-style profit reward plus unmet customer-energy service penalty."""
        energy = float(self.actions_b[-1]) * float(self.hr_b[-1])
        profit = -energy * float(price_t) / (self.cap_b * max(price_max, 1e-12))
        shortfall = max(float(self.soc_exp_b - self.soc_t_b), 0.0)
        service = -10.0 * shortfall / max(float(self.soc_exp_b), 1e-12)
        return profit + (service if flag_done and infra_i == 0 else 0.0), self.actions_b_pu[-1]

    if mode not in ("original", "smooth_battery", "smooth_both",
                    "smooth_both_executed", "smooth_both_feasible_fraction",
                    "smooth_both_cap_recovery", "smooth_both_cap_profit",
                    "smooth_both_cap_arbitrage",
                    "smooth_both_cap_arbitrage_terminal",
                    "smooth_both_cap_arbitrage_urgency",
                    "progress_battery", "weighted_tradeoff_battery_cap_arbitrage",
                    "li_cost_only", "ye_profit_service"):
        raise ValueError("Unknown reward mode")
    if not np.isfinite([tau, penalty_weight, lower, upper]).all():
        raise ValueError("Reward parameters must be finite")
    weights = [progress_weight, cost_weight, action_weight, terminal_weight]
    if tau <= 0 or penalty_weight < 0 or min(weights) < 0 or not 0 < lower < upper < 1:
        raise ValueError("Invalid reward smoothing parameters")
    try:
        if mode in ("smooth_battery", "smooth_both", "smooth_both_executed",
                    "smooth_both_feasible_fraction", "smooth_both_cap_recovery",
                    "smooth_both_cap_profit", "smooth_both_cap_arbitrage",
                    "smooth_both_cap_arbitrage_terminal",
                    "smooth_both_cap_arbitrage_urgency"):
            EV.get_reward_b = battery
        elif mode == "progress_battery":
            EV.get_reward_b = progress_battery
        elif mode == "weighted_tradeoff_battery_cap_arbitrage":
            EV.get_reward_b = weighted_tradeoff_battery
        elif mode == "li_cost_only":
            EV.get_reward_b = li_cost_only_battery
        elif mode == "ye_profit_service":
            EV.get_reward_b = ye_profit_service_battery
        if mode == "smooth_both":
            EV.get_reward_c2 = capacitor
        elif mode == "smooth_both_executed":
            EV.get_reward_c2 = capacitor_executed
        elif mode == "smooth_both_feasible_fraction":
            EV.get_reward_c2 = capacitor_feasible_fraction
        elif mode == "smooth_both_cap_recovery":
            EV.get_reward_c2 = capacitor_recovery
        elif mode == "smooth_both_cap_profit":
            EV.get_reward_c2 = capacitor_profit
        elif mode == "smooth_both_cap_arbitrage":
            EV.get_reward_c2 = capacitor_arbitrage
        elif mode == "smooth_both_cap_arbitrage_terminal":
            EV.get_reward_c2 = capacitor_arbitrage_terminal
        elif mode == "smooth_both_cap_arbitrage_urgency":
            EV.get_reward_c2 = capacitor_arbitrage_urgency
        elif mode == "weighted_tradeoff_battery_cap_arbitrage":
            EV.get_reward_c2 = capacitor_arbitrage
        yield
    finally:
        _active_soc_tau = previous_tau
        EV.get_reward_b, EV.get_reward_c2 = original_b, original_c

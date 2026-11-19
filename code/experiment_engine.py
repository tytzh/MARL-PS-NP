"""DDPG training and result aggregation for Proposed and its ablations."""
import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

METHODS = {
    "proposed": {
        "label": "proposed", "storage_name": "proposed", "trainer": "ddpg",
        "group": 3, "knn": 5, "prune": 10, "state_assembly": "preference",
    },
    "w/o N": {
        "label": "w/o N", "storage_name": "w_o_N", "trainer": "ddpg",
        "group": 1, "knn": 5, "prune": 0, "state_assembly": "preference",
        "no_mask_specialization": True,
    },
    "w/o N+F (C)": {
        "label": "w/o N+F (C)", "storage_name": "w_o_N+F_C", "trainer": "ddpg",
        "group": 1, "knn": 20, "prune": 0, "state_assembly": "fixed_global",
        "no_mask_specialization": True,
    },
    "w/o N+F (D)": {
        "label": "w/o N+F (D)", "storage_name": "w_o_N+F_D", "trainer": "ddpg",
        "group": 1, "knn": 1, "prune": 0, "state_assembly": "preference",
        "no_mask_specialization": True,
    },
}
SEEDS = (1, 2, 3)
RUN_DIR = Path(__file__).resolve().parent.parent / "results" / "10ev"
METRICS = ("profitu", "costb", "psoc", "dsoc", "transformer_overload_to_ev_allowance_percent")

base = SimpleNamespace(
    EPISODES=200, TRAIN_EVS=10, TEST_EVS=10, TEST_RUNS=2, TRAIN_SEEDS=[1, 2, 3],
    TEST_SEED=60, REWARD_MODE="smooth_both_cap_arbitrage", SOC_TAU=0.03,
    CAP_PENALTY_WEIGHT=0.0, BATTERY_ACTION_MAPPING="feasible",
    BATTERY_ACTION_REGULARIZATION=0.20, CAPACITOR_OPERATING_BAND=(0.1, 0.9),
    CAPACITOR_RATE_KW=10.0, CAP_TERMINAL_WEIGHT=20.0,
    CAPACITOR_REWARD_SCALE=1.0, CAPACITOR_TD3=False,
    TRANSFORMER_CREDIT="proportional", TRANSFORMER_BATTERY_WEIGHT=1.0,
    TRANSFORMER_CAPACITOR_WEIGHT=3.0, TRANSFORMER_DEFINITION="total_grid",
    TRANSFORMER_CAPACITY_KW=300.0, INFRASTRUCTURE_SCENARIO="hybrid",
    INFRASTRUCTURE_ASSIGNMENT="exclusive",
)


def method_folder(method_name, run_dir=None):
    base = RUN_DIR if run_dir is None else Path(run_dir)
    return base / METHODS[method_name]["storage_name"]


def _train_worker(method_name, seed):
    method = experiment.METHODS[method_name]
    output = method_folder(method_name, experiment.RUN_DIR) / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    if method.get("no_mask_specialization"):
        os.environ["VX_NO_MASK_SPECIALIZATION"] = "1"
    else:
        os.environ.pop("VX_NO_MASK_SPECIALIZATION", None)

    from training_pipeline import configure
    configure(output)
    from reward_shaping import reward_mode
    from actor_critic_consistency import corrected_vx
    common = {
        "EPISODES": base.EPISODES, "TRAIN_EVS": base.TRAIN_EVS,
        "TRAIN_SEED": seed, "N_GROUP": method["group"],
        "N_KNN": method["knn"], "PRUNE_PERCENT": method["prune"],
        "REWARD_FILTER": False, "CORRECTED_VX": True,
        "INFRASTRUCTURE_SCENARIO": method.get("infrastructure_scenario", "hybrid"),
        "INFRASTRUCTURE_ASSIGNMENT": method.get("infrastructure_assignment", "exclusive"),
        "CAPACITOR_OPERATING_BAND": base.CAPACITOR_OPERATING_BAND,
        "BATTERY_ACTION_MAPPING": method.get("battery_action_mapping", base.BATTERY_ACTION_MAPPING),
    }
    lo, hi = base.CAPACITOR_OPERATING_BAND
    with corrected_vx(
        "cs_only", common["BATTERY_ACTION_MAPPING"],
        method.get("battery_action_regularization", base.BATTERY_ACTION_REGULARIZATION),
        base.CAPACITOR_OPERATING_BAND,
        method.get("capacitor_td3", base.CAPACITOR_TD3),
        method.get("capacitor_td3_actor_lr"),
        method.get("transformer_credit", base.TRANSFORMER_CREDIT),
        method.get("transformer_battery_weight", base.TRANSFORMER_BATTERY_WEIGHT),
        method.get("transformer_capacitor_weight", base.TRANSFORMER_CAPACITOR_WEIGHT),
        method.get("capacitor_td3_critic_lr"),
        method.get("capacitor_gradient_steps", 1),
        method.get("capacitor_noise_clip", 0.02),
        battery_actor_lr=method.get("battery_actor_lr"),
        battery_critic_lr=method.get("battery_critic_lr"),
        battery_gradient_steps=method.get("battery_gradient_steps", 1),
        state_assembly=method.get("state_assembly", "preference"),
        transformer_definition=base.TRANSFORMER_DEFINITION,
    ):
        with reward_mode(
            mode=base.REWARD_MODE, tau=base.SOC_TAU,
            penalty_weight=base.CAP_PENALTY_WEIGHT, lower=lo, upper=hi,
            terminal_weight=base.CAP_TERMINAL_WEIGHT,
            capacitor_scale=base.CAPACITOR_REWARD_SCALE,
        ):
            result = runpy.run_path(
                str(Path(__file__).resolve().parent / "ev_marl" / "training_core.py"),
                init_globals=common, run_name="__main__",
            )
        trained = {
            "agent_b": result["agent_b"], "agent_c": result["agent_c"],
            "centers": np.asarray(result["centers"]), "n_group": method["group"],
            "n_knn": method["knn"], "price_degrad": float(result["price_degrad"]),
        }
        from evaluation import evaluate_repeated
        evaluate_repeated(
            trained, output, base.TEST_EVS, base.TEST_SEED, base.TEST_RUNS, METRICS,
            lo, hi, method.get("infrastructure_scenario", "hybrid"),
            method.get("evaluation_transformer_scope", "cs_only"),
            base.TRANSFORMER_DEFINITION,
            method.get("infrastructure_assignment", "exclusive"),
        )
    config = {
        "method": method_name, "method_config": method, "seed": seed,
        "episodes": base.EPISODES, "train_evs": base.TRAIN_EVS,
        "test_evs": base.TEST_EVS, "train_seeds": list(experiment.SEEDS),
        "test_seed": base.TEST_SEED, "test_runs": base.TEST_RUNS,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


class Experiment:
    METRICS = METRICS
    METHODS = METHODS
    RUN_DIR = RUN_DIR
    SEEDS = SEEDS
    base = base
    __file__ = __file__

    def run_worker(self, method, seed):
        _train_worker(method, seed)

    def read_run(self, folder):
        table = json.loads((Path(folder) / "indicators.json").read_text())["table_summary"]
        return {key: table[key]["mean"] for key in self.METRICS}

    def aggregate(self):
        summary = {}
        for name in self.METHODS:
            runs = [self.read_run(method_folder(name, self.RUN_DIR) / f"seed_{seed}")
                    for seed in self.SEEDS]
            summary[name] = {}
            for key in self.METRICS:
                values = [run[key] for run in runs if run[key] is not None]
                summary[name][key] = {
                    "mean": float(np.mean(values)) if values else None,
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                }
        return summary, None

    def write_reports(self, _summary):
        return None


experiment = Experiment()

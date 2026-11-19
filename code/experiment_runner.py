"""Configure, execute, and aggregate the DDPG structural ablation experiments.

Edit EXPERIMENT_CONFIG and METHOD_CONFIGS below to change a run.
"""
import argparse
import copy
import hashlib
import json
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ev_marl"))
import experiment_engine as runner

METHOD_NAMES = ("proposed", "w/o N", "w/o N+F (C)", "w/o N+F (D)")
DEFAULT_METHODS = {name: copy.deepcopy(runner.METHODS[name]) for name in METHOD_NAMES}

# Shared experiment and environment settings.
EXPERIMENT_CONFIG = {
    "episodes": 400,
    "deterministic_torch": True,
    "python_hash_seed": 0,
    "buffer_size": 500_000,
    "batch_size": 64,
    "gamma": 0.99,
    "target_tau": 1e-3,
    "battery_weight_decay": 5e-4,
    "capacitor_weight_decay": 5e-4,
    "capacitor_noise_clip": 0.02,
    "train_seeds": 1,
    "train_evs": 20,
    "test_evs": 20,
    "test_runs": 2,
    "test_seed": 60,
    "reward_mode": "smooth_both_cap_arbitrage",
    "soc_tau": 0.03,
    "cap_penalty_weight": 0.0,
    "battery_action_mapping": "feasible",
    "battery_action_regularization": 0.20,
    "capacitor_operating_band": (0.1, 0.9),
    "capacitor_rate_kw": 10.0,
    "cap_terminal_weight": 20.0,
    "capacitor_reward_scale": 1.0,
    "capacitor_td3": False,
    "transformer_credit": "proportional",
    "transformer_battery_weight": 1.0,
    "transformer_capacitor_weight": 3.0,
    "transformer_definition": "total_grid",
    "transformer_capacity_kw": 300.0,
    "infrastructure_scenario": "hybrid",
    "infrastructure_assignment": "exclusive",
    "heterogeneous_stress": True,
    "jobs": 3,
}
if isinstance(EXPERIMENT_CONFIG["train_seeds"], int):
    EXPERIMENT_CONFIG["train_seeds"] = (EXPERIMENT_CONFIG["train_seeds"],)

# Per-method algorithm settings. K includes the controlled EV itself.
METHOD_CONFIGS = {
    "proposed": {
        "group": 3, "knn": 5, "prune": 10, "state_assembly": "preference",
        "capacitor_gradient_steps": 2,
        "capacitor_td3_actor_lr": 1e-4, "capacitor_td3_critic_lr": 1e-4,
        "battery_actor_lr": 2e-4, "battery_critic_lr": 2e-4,
        "battery_gradient_steps": 1,
    },
    "w/o N": {
        "group": 1, "knn": 5, "prune": 0, "state_assembly": "preference",
        "no_mask_specialization": True,
        "capacitor_gradient_steps": 2,
        "capacitor_td3_actor_lr": 1e-4, "capacitor_td3_critic_lr": 1e-4,
        "battery_actor_lr": 2e-4, "battery_critic_lr": 2e-4,
        "battery_gradient_steps": 1,
    },
    "w/o N+F (C)": {
        "group": 1, "knn": 20, "prune": 0, "state_assembly": "fixed_global",
        "no_mask_specialization": True,
        "capacitor_gradient_steps": 2,
        "capacitor_td3_actor_lr": 1e-4, "capacitor_td3_critic_lr": 1e-4,
        "battery_actor_lr": 2e-4, "battery_critic_lr": 2e-4,
        "battery_gradient_steps": 1,
    },
    "w/o N+F (D)": {
        "group": 1, "knn": 1, "prune": 0, "state_assembly": "preference",
        "no_mask_specialization": True,
        "capacitor_gradient_steps": 2,
        "capacitor_td3_actor_lr": 1e-4, "capacitor_td3_critic_lr": 1e-4,
        "battery_actor_lr": 2e-4, "battery_critic_lr": 2e-4,
        "battery_gradient_steps": 1,
    },
}


def configured_methods():
    cfg = EXPERIMENT_CONFIG
    methods = copy.deepcopy(DEFAULT_METHODS)
    shared_defaults = {
        "battery_action_regularization": cfg["battery_action_regularization"],
        "capacitor_td3": cfg["capacitor_td3"],
        "transformer_credit": cfg["transformer_credit"],
        "capacitor_noise_clip": cfg["capacitor_noise_clip"],
        "infrastructure_scenario": cfg["infrastructure_scenario"],
        "infrastructure_assignment": cfg["infrastructure_assignment"],
    }
    for method in methods.values():
        method.update(shared_defaults)
    for name, settings in METHOD_CONFIGS.items():
        methods[name].update(settings)
    methods["w/o N+F (C)"]["knn"] = cfg["train_evs"]
    return methods


def output_directory(methods):
    signature_config = {
        "experiment": {k: v for k, v in EXPERIMENT_CONFIG.items() if k != "jobs"},
        "methods": methods,
    }
    encoded = json.dumps(signature_config, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:8]
    cfg = EXPERIMENT_CONFIG
    return ROOT / "Reruns" / (
        f"structural_ablations_{cfg['episodes']}ep_{len(cfg['train_seeds'])}seeds_{signature}")


runner.SEEDS = tuple(EXPERIMENT_CONFIG["train_seeds"])
runner.METHODS = configured_methods()
runner.RUN_DIR = output_directory(runner.METHODS)
runner.experiment.__file__ = str(Path(__file__).resolve())


def configure():
    cfg = EXPERIMENT_CONFIG
    runner.METHODS = configured_methods()
    runner.RUN_DIR = output_directory(runner.METHODS)
    runner.SEEDS = tuple(cfg["train_seeds"])
    runner.experiment.RUN_DIR = runner.RUN_DIR
    runner.experiment.SEEDS = runner.SEEDS
    runner.experiment.METHODS = runner.METHODS
    import torch
    torch.use_deterministic_algorithms(cfg["deterministic_torch"], warn_only=False)
    torch.backends.cudnn.deterministic = cfg["deterministic_torch"]
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    runner.experiment.METRICS = tuple(dict.fromkeys(
        (*runner.experiment.METRICS, "costb", "profitu")))

    base = runner.experiment.base
    base.RUN_DIR, base.METHODS = runner.RUN_DIR, runner.METHODS
    base.EPISODES = cfg["episodes"]
    base.TRAIN_EVS = cfg["train_evs"]
    base.TEST_EVS = cfg["test_evs"]
    base.TEST_RUNS = cfg["test_runs"]
    base.TRAIN_SEEDS = list(cfg["train_seeds"])
    base.TEST_SEED = cfg["test_seed"]
    base.REWARD_MODE = cfg["reward_mode"]
    base.SOC_TAU = cfg["soc_tau"]
    base.CAP_PENALTY_WEIGHT = cfg["cap_penalty_weight"]
    base.BATTERY_ACTION_MAPPING = cfg["battery_action_mapping"]
    base.BATTERY_ACTION_REGULARIZATION = cfg["battery_action_regularization"]
    base.CAPACITOR_OPERATING_BAND = tuple(cfg["capacitor_operating_band"])
    base.CAPACITOR_RATE_KW = cfg["capacitor_rate_kw"]
    base.CAP_TERMINAL_WEIGHT = cfg["cap_terminal_weight"]
    base.CAPACITOR_REWARD_SCALE = cfg["capacitor_reward_scale"]
    base.CAPACITOR_TD3 = cfg["capacitor_td3"]
    base.TRANSFORMER_CREDIT = cfg["transformer_credit"]
    base.TRANSFORMER_BATTERY_WEIGHT = cfg["transformer_battery_weight"]
    base.TRANSFORMER_CAPACITOR_WEIGHT = cfg["transformer_capacitor_weight"]
    base.TRANSFORMER_DEFINITION = cfg["transformer_definition"]
    base.INFRASTRUCTURE_SCENARIO = cfg["infrastructure_scenario"]
    os.environ["VX_HETEROGENEOUS_STRESS"] = "1" if cfg["heterogeneous_stress"] else "0"

    from training_pipeline import configure as configure_paths
    configure_paths(runner.RUN_DIR)
    import constants
    import ddpg_agents as agents
    agents.BUFFER_SIZE = cfg["buffer_size"]
    agents.BATCH_SIZE = cfg["batch_size"]
    agents.GAMMA = cfg["gamma"]
    agents.TAU = cfg["target_tau"]
    agents.WEIGHT_DECAY = cfg["battery_weight_decay"]
    agents.WEIGHT_DECAY_C = cfg["capacitor_weight_decay"]
    constants.cap_trans = cfg["transformer_capacity_kw"]
    constants.rate_CS_c = cfg["capacitor_rate_kw"]
    constants.rate_PT_c = cfg["capacitor_rate_kw"]


runner.configure = configure
runner.experiment.configure = configure


def print_seed_result(method, seed):
    print(f"Completed: {runner.METHODS[method]['label']} | seed {seed}", flush=True)


def run_is_complete(method, seed):
    folder = runner.method_folder(method) / f"seed_{seed}"
    if not (folder / "config.json").is_file() or not (folder / "indicators.json").is_file():
        return False
    try:
        config = json.loads((folder / "config.json").read_text())
        indicators = json.loads((folder / "indicators.json").read_text())
        table = indicators["table_summary"]
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False
    base = runner.experiment.base
    return (
        config.get("method") == method
        and config.get("seed") == seed
        and config.get("method_config") == runner.METHODS[method]
        and config.get("episodes") == base.EPISODES
        and config.get("train_evs") == base.TRAIN_EVS
        and config.get("test_evs") == base.TEST_EVS
        and config.get("train_seeds") == list(runner.SEEDS)
        and config.get("test_runs") == base.TEST_RUNS
        and config.get("test_seed") == base.TEST_SEED
        and indicators.get("test_runs") == base.TEST_RUNS
        and all(key in table and "mean" in table[key] for key in runner.METRICS)
    )


def format_metric(stat):
    if stat["mean"] is None:
        return "N/A"
    if stat["sd"] is None:
        return f"{stat['mean']:.4f} (single seed)"
    return f"{stat['mean']:.4f} ± {stat['sd']:.4f}"


def print_metric_comparison(summary):
    metric_rows = (
        ("profitu (10^-2 $)", "profitu"),
        ("costb ($)", "costb"),
        ("pSoC (%)", "psoc"),
        ("dSoC (%)", "dsoc"),
        ("Total EV overload / total EV-allowable capacity (%)", "transformer_overload_to_ev_allowance_percent"),
    )
    names = list(runner.METHODS)
    headers = ["Metric", *(runner.METHODS[name]["label"] for name in names)]
    rows = []
    for label, key in metric_rows:
        rows.append([label, *(format_metric(summary[name][key]) for name in names)])
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows))
              for i in range(len(headers))]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print("\n=== Metrics: mean ± SD across training seeds ===")
    print(border)
    print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |")
    print(border)
    for row in rows:
        print("| " + " | ".join(value.rjust(widths[i]) for i, value in enumerate(row)) + " |")
    print(border)


def main():
    configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("METHOD", "SEED"))
    parser.add_argument("--jobs", type=int, default=EXPERIMENT_CONFIG["jobs"])
    args = parser.parse_args()
    if args.worker:
        runner.experiment.run_worker(args.worker[0], int(args.worker[1]))
        return

    runner.RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {
        "experiment": EXPERIMENT_CONFIG,
        "methods": runner.METHODS,
    }
    (runner.RUN_DIR / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True))

    pending = [(method, seed) for method in runner.METHODS for seed in runner.SEEDS
               if not run_is_complete(method, seed)]
    skipped = len(runner.METHODS) * len(runner.SEEDS) - len(pending)
    if pending:
        print(f"Training {len(pending)} runs; skipped {skipped} completed runs.", flush=True)
    else:
        print("All requested runs are complete; skipping training and refreshing the report.", flush=True)
    active = []
    while pending or active:
        while pending and len(active) < max(1, args.jobs):
            method, seed = pending.pop(0)
            folder = runner.method_folder(method) / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                        "KMP_USE_SHM": "0", "PYTHONHASHSEED": str(EXPERIMENT_CONFIG["python_hash_seed"]),
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
            log = (folder / "run.log").open("w")
            worker_args = ["--worker", method, str(seed)]
            if Path(__file__).name == "main.py":
                worker_args = ["--evs", str(EXPERIMENT_CONFIG["train_evs"]), *worker_args]
            process = subprocess.Popen(
                [sys.executable, __file__, *worker_args],
                stdout=log, stderr=subprocess.STDOUT, env=env)
            active.append((method, seed, process, log))
        method, seed, process, log = active.pop(0)
        code = process.wait()
        log.close()
        if code:
            log_path = runner.method_folder(method) / f"seed_{seed}" / "run.log"
            raise RuntimeError(f"{method} seed {seed} failed; see {log_path}")
        (runner.method_folder(method) / f"seed_{seed}" / "run.log").unlink(missing_ok=True)
        print_seed_result(method, seed)

    summary, _ = runner.experiment.aggregate()
    if os.environ.get("VERSION3_ENTRYPOINT") != "1":
        print_metric_comparison(summary)
    runner.experiment.write_reports(summary)
    print(f"\nResults: {runner.RUN_DIR}")
    print(f"Report: {runner.RUN_DIR / 'report.md'}")


if __name__ == "__main__":
    main()

"""Run Proposed and its three structural ablations."""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
MAIN = CODE / "ev_marl"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(MAIN))
METHODS = ("proposed", "w/o N", "w/o N+F (C)", "w/o N+F (D)")


def format_stat(stat):
    if stat["mean"] is None:
        return "N/A"
    if stat["sd"] is None:
        return f"{stat['mean']:.4f} (single seed)"
    return f"{stat['mean']:.4f} ± {stat['sd']:.4f}"


def save_report(experiment):
    runner = experiment.runner
    columns = (
        ("profitu (10^-2 $)", "profitu"),
        ("costb ($)", "costb"),
        ("pSoC (%)", "psoc"),
        ("dSoC (%)", "dsoc"),
        ("Total EV overload / total EV-allowable capacity (%)", "transformer_overload_to_ev_allowance_percent"),
    )
    per_seed, results = {}, {}
    for name in METHODS:
        per_seed[name] = {}
        for seed in runner.SEEDS:
            folder = runner.method_folder(name) / f"seed_{seed}"
            table = json.loads((folder / "indicators.json").read_text())["table_summary"]
            per_seed[name][str(seed)] = {
                **{key: table[key]["mean"] for key in
                   ("profitu", "costb", "psoc", "dsoc", "transformer_overload_to_ev_allowance_percent")},
            }
        results[name] = {"label": runner.METHODS[name]["label"]}
        for metric in per_seed[name][str(runner.SEEDS[0])]:
            values = [per_seed[name][str(seed)][metric] for seed in runner.SEEDS]
            values = [value for value in values if value is not None]
            import numpy as np
            results[name][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    cfg = experiment.EXPERIMENT_CONFIG
    payload = {
        "episodes": cfg["episodes"], "train_evs": cfg["train_evs"],
        "test_evs": cfg["test_evs"], "train_seeds": list(cfg["train_seeds"]),
        "test_seeds": list(range(cfg["test_seed"], cfg["test_seed"] + cfg["test_runs"])),
        "methods": runner.METHODS, "results": results, "per_seed": per_seed,
    }
    runner.RUN_DIR.mkdir(parents=True, exist_ok=True)
    (runner.RUN_DIR / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# proposed and ablations", "",
             f"{cfg['train_evs']} train/test EVs; {cfg['episodes']} episodes; "
             f"training seeds {', '.join(map(str, cfg['train_seeds']))}; "
             f"paired test seeds {cfg['test_seed']} and {cfg['test_seed'] + 1}.", "",
             "| Metric | " + " | ".join(results[name]["label"] for name in METHODS) + " |",
             "|---|" + "---:|" * len(METHODS)]
    for label, key in columns:
        cells = []
        for name in METHODS:
            stat = results[name][key]
            cells.append(format_stat(stat))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    (runner.RUN_DIR / "report.md").write_text("\n".join(lines) + "\n")


def display_metrics(experiment):
    runner = experiment.runner
    data = json.loads((runner.RUN_DIR / "comparison.json").read_text())
    columns = (
        ("profitu (10^-2 $)", "profitu"),
        ("costb ($)", "costb"),
        ("pSoC (%)", "psoc"),
        ("dSoC (%)", "dsoc"),
        ("Total EV overload / total EV-allowable capacity (%)", "transformer_overload_to_ev_allowance_percent"),
    )
    headers = ["Method", *(label for label, _ in columns)]
    rows = []
    for name in METHODS:
        metrics = data["results"][name]
        rows.append([data["methods"][name]["label"],
                     *(format_stat(metrics[key]) for _, key in columns)])
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows))
              for i in range(len(headers))]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    title = ("Final metrics (mean ± SD across training seeds):" if len(runner.SEEDS) > 1
             else "Final metrics (single training seed):")
    print("\n" + title, flush=True)
    print(border, flush=True)
    print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |", flush=True)
    print(border, flush=True)
    for row in rows:
        print("| " + " | ".join(value.rjust(widths[i]) for i, value in enumerate(row)) + " |", flush=True)
    print(border, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduce the proposed method and ablations")
    parser.add_argument("--evs", type=int, choices=(10, 40),
                        default=40)
    parser.add_argument("--jobs", type=int, default=2,
                        help="Number of training jobs to run in parallel")
    args, forwarded = parser.parse_known_args(argv)
    train_seeds = (1, 2, 3)
    os.environ["VERSION3_ENTRYPOINT"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import experiment_runner as experiment
    cfg = experiment.EXPERIMENT_CONFIG
    cfg.update({"episodes": 200, "train_seeds": train_seeds, "jobs": args.jobs,
                "train_evs": args.evs, "test_evs": args.evs})
    experiment.output_directory = lambda _methods: (
        HERE / "results" / f"{args.evs}ev_ev_capacity_ratio")
    experiment.__file__ = str(Path(__file__).resolve())
    experiment.runner.experiment.__file__ = str(Path(__file__).resolve())
    sys.argv = [sys.argv[0], *forwarded]
    experiment.main()
    if "--worker" not in forwarded:
        save_report(experiment)
        display_metrics(experiment)


if __name__ == "__main__":
    main()

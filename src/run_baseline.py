import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_sales, prepare_windows
from src.evaluate import bootstrap_by_store, metrics, relative_improvement, stores_with_spikes, summarize_data
from src.train import EXPERIMENTS, fit_model, predict


def run(config):
    sales = load_sales(Path(config["data_path"]))
    print(f"Loaded {len(sales):,} rows", flush=True)
    train, validation, cutoff, origins = prepare_windows(sales, config)
    window_count = config["validation_windows"]
    if not 1 <= window_count <= len(origins):
        raise ValueError("The backtest must leave at least one earlier training window.")

    predictions = {}
    scores = {}
    for name, columns in EXPERIMENTS.items():
        print(f"Fitting {name} for {cutoff.date()}", flush=True)
        predictions[name] = predict(fit_model(train, columns), validation)
        scores[name] = metrics(validation, predictions[name])

    windows = []
    window = pd.Timedelta(days=config["window_days"])
    earlier_origins = origins[-(window_count - 1):] if window_count > 1 else []
    for start in earlier_origins:
        print(f"Backtest starting {start.date()}", flush=True)
        earlier_train = train.loc[train["Date"] < start]
        earlier_validation = train.loc[train["Date"].between(start, start + window, inclusive="left")]
        record = {"start": str(start.date()), "end": str((start + window - pd.Timedelta(days=1)).date())}
        for name in ("no_history", "with_history"):
            model = fit_model(earlier_train, EXPERIMENTS[name])
            record[name] = metrics(earlier_validation, predict(model, earlier_validation))["rmspe"]
        windows.append(record)
    windows.append({
        "start": str(cutoff.date()),
        "end": str(validation["Date"].max().date()),
        "no_history": scores["no_history"]["rmspe"],
        "with_history": scores["with_history"]["rmspe"],
    })
    window_summary = {
        name: {
            "mean": float(np.mean([row[name] for row in windows])),
            "std": float(np.std([row[name] for row in windows], ddof=1)) if len(windows) > 1 else None,
        }
        for name in ("no_history", "with_history")
    }

    spiky_stores = stores_with_spikes(sales.loc[sales["Date"] < cutoff])
    last_year = validation["last_year_available"].to_numpy()
    spiky = validation["Store"].isin(spiky_stores).to_numpy()
    checks = {}
    for label, mask, reference, candidate in [
        ("trend", np.ones(len(validation), dtype=bool), "without_trend", "with_history"),
        ("last_year", last_year, "without_last_year", "with_history"),
        ("spikes", spiky, "with_weekday_mean", "with_history"),
    ]:
        if (validation.loc[mask, "Sales"] > 0).any():
            checks[label] = {
                reference: metrics(validation.loc[mask], predictions[reference][mask]),
                candidate: metrics(validation.loc[mask], predictions[candidate][mask]),
            }

    improvement = relative_improvement(scores["no_history"]["rmspe"], scores["with_history"]["rmspe"])
    report = {
        "data": summarize_data(sales),
        "train": {"first_date": str(origins[0].date()), "rows": int(train["Sales"].gt(0).sum())},
        "validation": windows[-1],
        "experiments": scores,
        "relative_improvement": improvement,
        "min_relative_improvement": config["min_relative_improvement"],
        "meets_threshold": improvement is not None and improvement >= config["min_relative_improvement"],
        "bootstrap_95_ci": bootstrap_by_store(
            validation, predictions["no_history"], predictions["with_history"],
            config["bootstrap_samples"], config["seed"],
        ),
        "backtest": windows,
        "backtest_summary": window_summary,
        "research_checks": checks,
        "spiky_stores": len(spiky_stores),
    }

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    columns = ["Store", "Date", "DayOfWeek", "Open", "Promo", "StateHoliday", "Sales", "last_year_available"]
    validation[columns].assign(**predictions, spiky_store=spiky).to_csv(output / "predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Reports saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the Rossmann baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")))

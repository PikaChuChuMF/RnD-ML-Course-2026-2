import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import add_history, load_sales, prepare_windows
from src.evaluate import bootstrap_by_store, historical_spikes, metrics, relative_improvement, summarize_data
from src.train import EXPERIMENTS, fit_model, predict


def run(config):
    sales = load_sales(Path(config["data_path"]))
    print(f"Loaded {len(sales):,} rows", flush=True)
    train, validation, cutoff, origins = prepare_windows(sales, config)
    window_count = config["validation_windows"]
    if not 1 <= window_count <= len(origins):
        raise ValueError("The backtest must leave at least one earlier training window.")

    models, predictions, scores = {}, {}, {}
    for name, columns in EXPERIMENTS.items():
        print(f"Fitting {name} for {cutoff.date()}", flush=True)
        models[name] = fit_model(train, columns)
        predictions[name] = predict(models[name], validation)
        scores[name] = metrics(validation, predictions[name])

    alpha = config["l2_alpha"]
    regularized = {"alpha": alpha}
    for name in ("no_history", "with_history"):
        print(f"Fitting L2 {name} for {cutoff.date()}", flush=True)
        model = fit_model(train, EXPERIMENTS[name], alpha=alpha)
        predictions[f"l2_{name}"] = predict(model, validation)
        regularized[name] = metrics(validation, predictions[f"l2_{name}"])

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

    past = sales.loc[sales["Date"] < cutoff]
    spikes = historical_spikes(past)
    spiky_stores = spikes["Store"].unique()
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

    without_spikes = np.full(len(validation), np.nan)
    spike_sensitivity = None
    if (validation.loc[spiky, "Sales"] > 0).any():
        affected = validation.loc[spiky]
        clean_past = past.drop(index=spikes.index)
        clean_features = add_history(
            affected, clean_past, cutoff, config["recent_days"], config["year_lag_days"],
        )
        original = predictions["with_history"][spiky]
        filtered = predict(models["with_history"], clean_features)
        without_spikes[spiky] = filtered
        positive = affected["Sales"].gt(0).to_numpy()
        change = np.abs(filtered[positive] - original[positive])
        spike_sensitivity = {
            "removed_rows": len(spikes),
            "original": metrics(affected, original),
            "without_spikes": metrics(affected, filtered),
            "median_absolute_change": float(np.median(change)),
            "p90_absolute_change": float(np.quantile(change, 0.9)),
        }

    test = pd.read_csv(config["test_path"], usecols=["Store", "Date", "Open"], parse_dates=["Date"])
    test_stores = validation["Store"].isin(test["Store"]).to_numpy()
    test_scope = {
        "first_date": str(test["Date"].min().date()),
        "last_date": str(test["Date"].max().date()),
        "days": int(test["Date"].nunique()),
        "stores": int(test["Store"].nunique()),
        "missing_open": test.loc[test["Open"].isna(), "Store"].value_counts().to_dict(),
        "local_subset": {
            name: metrics(validation.loc[test_stores], predictions[name][test_stores])
            for name in ("no_history", "with_history")
        },
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
        "spike_sensitivity": spike_sensitivity,
        "regularization": regularized,
        "official_test": test_scope,
    }

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    columns = ["Store", "Date", "DayOfWeek", "Open", "Promo", "StateHoliday", "Sales", "last_year_available"]
    validation[columns].assign(
        **predictions, spiky_store=spiky, history_without_spikes=without_spikes,
    ).to_csv(output / "predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Reports saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the Rossmann baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")))

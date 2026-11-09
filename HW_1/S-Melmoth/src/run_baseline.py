import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import add_history, load_sales, prepare_windows
from src.evaluate import (
    bootstrap_by_store,
    compare_rmspe_by_store,
    historical_spikes,
    metrics,
    relative_improvement,
    summarize_data,
)
from src.train import EXPERIMENTS, fit_model, predict


def compare_history_over_time(sales, train, origins, cutoff, config, main_result):
    window = pd.Timedelta(days=config["window_days"])
    periods = []
    for offset in (4, 2):
        start = cutoff - offset * window
        if start not in origins:
            continue
        earlier = train.loc[train["Date"] < start]
        target = train.loc[train["Date"].between(start, start + window, inclusive="left")]
        if not earlier["Sales"].gt(0).any() or not target["Sales"].gt(0).any():
            continue

        _, scores, research = evaluate_window(sales, earlier, target, start, config)
        no_history = scores["no_history"]["rmspe"]
        with_history = scores["with_history"]["rmspe"]
        periods.append({
            "start": str(start.date()),
            "end": str((start + window - pd.Timedelta(days=1)).date()),
            "no_history": no_history,
            "with_history": with_history,
            "relative_improvement": relative_improvement(no_history, with_history),
            "comparison_uncertainty": research["comparison_uncertainty"],
        })

    periods.append(main_result)
    summary = {}
    for name in ("no_history", "with_history", "relative_improvement"):
        values = [period[name] for period in periods if period[name] is not None]
        summary[name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    return {"periods": periods, "summary": summary}


def evaluate_window(sales, train, validation, cutoff, config):
    models, predictions, scores = {}, {}, {}
    for name, columns in EXPERIMENTS.items():
        print(f"Fitting {name} for {cutoff.date()}", flush=True)
        models[name] = fit_model(train, columns)
        predictions[name] = predict(models[name], validation)
        scores[name] = metrics(validation, predictions[name])

    past = sales.loc[sales["Date"] < cutoff]
    spikes = historical_spikes(past)
    spiky_stores = spikes["Store"].unique()
    last_year = validation["last_year_available"].to_numpy()
    spiky = validation["Store"].isin(spiky_stores).to_numpy()
    all_rows = np.ones(len(validation), dtype=bool)
    research_pairs = [
        ("trend", all_rows, "without_trend", "with_history"),
        ("last_year", last_year, "without_last_year", "with_history"),
        ("spikes", spiky, "with_weekday_mean", "with_history"),
    ]
    checks = {}
    for label, mask, reference, candidate in research_pairs:
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
        removed_last_year = affected["last_year_available"] & ~clean_features["last_year_available"]
        clean_features.loc[removed_last_year, "last_year"] = clean_features.loc[removed_last_year, "weekday_mean"]
        original = predictions["with_history"][spiky]
        filtered = predict(models["with_history"], clean_features)
        without_spikes[spiky] = filtered
        positive = affected["Sales"].gt(0).to_numpy()
        change = np.abs(filtered[positive] - original[positive])
        spike_sensitivity = {
            "removed_rows": len(spikes),
            "replaced_last_year_scored_rows": int((removed_last_year & affected["Sales"].gt(0)).sum()),
            "original": metrics(affected, original),
            "without_spikes": metrics(affected, filtered),
            "median_absolute_change": float(np.median(change)),
            "p90_absolute_change": float(np.quantile(change, 0.9)),
        }

    comparison_pairs = [
        *research_pairs,
        ("remove_spikes", spiky, "with_history", "history_without_spikes"),
    ]
    comparison_uncertainty = compare_rmspe_by_store(
        validation, {**predictions, "history_without_spikes": without_spikes}, comparison_pairs,
        config["bootstrap_samples"], config["seed"],
    )
    predictions["spiky_store"] = spiky
    predictions["history_without_spikes"] = without_spikes
    research = {
        "research_checks": checks,
        "spiky_stores": len(spiky_stores),
        "spike_sensitivity": spike_sensitivity,
        "comparison_uncertainty": comparison_uncertainty,
    }
    return predictions, scores, research


def run(config):
    sales = load_sales(Path(config["data_path"]))
    print(f"Loaded {len(sales):,} rows", flush=True)
    train, validation, cutoff, origins = prepare_windows(sales, config)
    predictions, scores, research = evaluate_window(sales, train, validation, cutoff, config)
    improvement = relative_improvement(scores["no_history"]["rmspe"], scores["with_history"]["rmspe"])
    improvement_interval = bootstrap_by_store(
        validation, predictions["no_history"], predictions["with_history"],
        config["bootstrap_samples"], config["seed"],
    )

    validation_result = {
        "start": str(cutoff.date()),
        "end": str(validation["Date"].max().date()),
        "no_history": scores["no_history"]["rmspe"],
        "with_history": scores["with_history"]["rmspe"],
    }
    history_over_time = compare_history_over_time(
        sales, train, origins, cutoff, config,
        {
            **validation_result,
            "relative_improvement": improvement,
            "comparison_uncertainty": research["comparison_uncertainty"],
        },
    )
    report = {
        "data": summarize_data(sales),
        "train": {"first_date": str(origins[0].date()), "rows": int(train["Sales"].gt(0).sum())},
        "validation": validation_result,
        "history_over_time": history_over_time,
        "experiments": scores,
        "relative_improvement": improvement,
        "min_relative_improvement": config["min_relative_improvement"],
        "meets_threshold": improvement is not None and improvement >= config["min_relative_improvement"],
        "bootstrap_95_ci": improvement_interval,
        **research,
    }

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    columns = ["Store", "Date", "DayOfWeek", "Open", "Promo", "StateHoliday", "Sales", "last_year_available"]
    validation[columns].assign(**predictions).to_csv(output / "predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Reports saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the Rossmann baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")))

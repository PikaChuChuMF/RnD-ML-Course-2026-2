import numpy as np
import pandas as pd


def metrics(rows, prediction):
    scored = rows["Sales"] > 0
    if not scored.any():
        raise ValueError("RMSPE needs at least one row with positive sales.")
    actual = rows.loc[scored, "Sales"].to_numpy()
    error = prediction[scored] - actual
    return {
        "rows": int(len(actual)),
        "rmspe": float(np.sqrt(np.mean((error / actual) ** 2))),
        "mae": float(np.mean(np.abs(error))),
    }


def relative_improvement(reference, candidate):
    return (reference - candidate) / reference if reference else None


def summarize_data(sales):
    gap_start, gap_end = "2014-07-01", "2014-12-31"
    observed = sales.loc[sales["Date"].between(gap_start, gap_end), "Store"].unique()
    missing_stores = sales["Store"].nunique() - len(observed)
    return {
        "rows": len(sales),
        "stores": int(sales["Store"].nunique()),
        "first_date": str(sales["Date"].min().date()),
        "last_date": str(sales["Date"].max().date()),
        "missing": sales.isna().sum().to_dict(),
        "types": sales.dtypes.astype(str).to_dict(),
        "zero_sales": int(sales["Sales"].eq(0).sum()),
        "closed_days": int(sales["Open"].eq(0).sum()),
        "open_zero_sales": int((sales["Open"].eq(1) & sales["Sales"].eq(0)).sum()),
        "promo_days": int(sales["Promo"].eq(1).sum()),
        "observation_gap": {
            "start": gap_start,
            "end": gap_end,
            "days": len(pd.date_range(gap_start, gap_end)),
            "stores": int(missing_stores),
        },
        "positive_sales_quantiles": sales.loc[sales["Sales"] > 0, "Sales"]
        .quantile([0, 0.01, 0.5, 0.99, 1]).to_dict(),
    }


def historical_spikes(past):
    positive = past.loc[past["Sales"] > 0].copy()
    weekday_median = positive.groupby(["Store", "DayOfWeek"])["Sales"].transform("median")
    positive["spike"] = positive["Sales"] >= 3 * weekday_median
    spike_rate = positive.groupby("Store")["spike"].mean()
    stores = spike_rate.index[(spike_rate > 0) & (spike_rate <= 0.01)]
    return positive.loc[positive["spike"] & positive["Store"].isin(stores)]


def bootstrap_rmspe_by_store(rows, reference, candidate, samples, seed):
    scored = rows["Sales"] > 0
    actual = rows.loc[scored, "Sales"].to_numpy()
    errors = pd.DataFrame({
        "store": rows.loc[scored, "Store"].to_numpy(),
        "reference": ((reference[scored] - actual) / actual) ** 2,
        "candidate": ((candidate[scored] - actual) / actual) ** 2,
        "rows": 1,
    })
    groups = errors.groupby("store")[["reference", "candidate", "rows"]].sum().to_numpy()
    rng = np.random.default_rng(seed)
    rmspe_samples = np.empty((samples, 2))
    for index in range(samples):
        drawn = rng.integers(len(groups), size=len(groups))
        reference_error, candidate_error, count = groups[drawn].sum(axis=0)
        rmspe_samples[index] = np.sqrt([reference_error / count, candidate_error / count])
    return rmspe_samples


def bootstrap_by_store(rows, reference, candidate, samples, seed):
    rmspe_samples = bootstrap_rmspe_by_store(rows, reference, candidate, samples, seed)
    reference_rmspe, candidate_rmspe = rmspe_samples[rmspe_samples[:, 0] > 0].T
    valid = (reference_rmspe - candidate_rmspe) / reference_rmspe
    return np.quantile(valid, [0.025, 0.975]).tolist() if valid.size else None


def compare_rmspe_by_store(rows, predictions, comparisons, samples, seed):
    family_size = len(comparisons)
    tail = 0.05 / (2 * family_size)
    results = {}
    for name, mask, reference, candidate in comparisons:
        subset = rows.loc[mask]
        scored = subset["Sales"].gt(0)
        stores = subset.loc[scored, "Store"].nunique()
        if stores < 2:
            continue
        before = predictions[reference][mask]
        after = predictions[candidate][mask]
        estimates = bootstrap_rmspe_by_store(subset, before, after, samples, seed)
        differences = estimates[:, 1] - estimates[:, 0]
        low, high = np.quantile(differences, [tail, 1 - tail])
        results[name] = {
            "stores": int(stores),
            "rows": int(scored.sum()),
            "delta_rmspe": metrics(subset, after)["rmspe"] - metrics(subset, before)["rmspe"],
            "interval": [float(low), float(high)],
            "excludes_zero": bool(high < 0 or low > 0),
        }
    return {"samples": samples, "family_size": family_size, "comparisons": results}

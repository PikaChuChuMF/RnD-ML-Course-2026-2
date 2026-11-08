from pathlib import Path

import pandas as pd


KEY = ["Store", "Date"]
INPUT_COLUMNS = KEY + ["DayOfWeek", "Sales", "Open", "Promo", "StateHoliday"]
HISTORY_COLUMNS = [
    "store_median",
    "weekday_median",
    "recent_mean",
    "trend",
    "last_year",
    "promo_median",
]


def load_sales(path: Path) -> pd.DataFrame:
    sales = pd.read_csv(path, parse_dates=["Date"], dtype={"StateHoliday": str})
    if sales.empty or sales[INPUT_COLUMNS].isna().any().any() or sales.duplicated(KEY).any():
        raise ValueError("Sales data must have complete, unique store-date rows.")
    if (sales["Sales"] < 0).any():
        raise ValueError("Sales cannot be negative.")
    sales["Month"] = sales["Date"].dt.month
    return sales.sort_values("Date").reset_index(drop=True)


def add_history(
    rows: pd.DataFrame,
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    recent_days: int,
    year_lag_days: int,
) -> pd.DataFrame:
    positive = history.loc[history["Sales"] > 0]
    store_sales = positive.groupby("Store")["Sales"]
    weekday_sales = positive.groupby(["Store", "DayOfWeek"])["Sales"]
    result = rows.copy()
    result["store_median"] = rows["Store"].map(store_sales.median()).fillna(
        positive["Sales"].median()
    )
    weekday_keys = pd.MultiIndex.from_frame(rows[["Store", "DayOfWeek"]])
    result["weekday_median"] = weekday_sales.median().reindex(weekday_keys).to_numpy()
    result["weekday_median"] = result["weekday_median"].fillna(result["store_median"])
    result["weekday_mean"] = weekday_sales.mean().reindex(weekday_keys).to_numpy()
    result["weekday_mean"] = result["weekday_mean"].fillna(result["store_median"])

    recent_start = cutoff - pd.Timedelta(days=recent_days)
    previous_start = recent_start - pd.Timedelta(days=recent_days)
    recent = positive.loc[positive["Date"] >= recent_start].groupby("Store")["Sales"].mean()
    previous = positive.loc[positive["Date"].between(previous_start, recent_start, inclusive="left")]
    previous_mean = previous.groupby("Store")["Sales"].mean()
    result["recent_mean"] = rows["Store"].map(recent).fillna(result["store_median"])
    result["trend"] = rows["Store"].map(recent - previous_mean).fillna(0)

    last_year_dates = rows["Date"] - pd.Timedelta(days=year_lag_days)
    last_year_keys = pd.MultiIndex.from_arrays([rows["Store"], last_year_dates], names=KEY)
    result["last_year"] = history.set_index(KEY)["Sales"].reindex(last_year_keys).to_numpy()
    result["last_year_available"] = result["last_year"].notna()
    result["last_year"] = result["last_year"].fillna(result["weekday_median"])

    promo_sales = positive.loc[positive["Promo"] == 1].groupby("Store")["Sales"].median()
    result["promo_median"] = rows["Store"].map(promo_sales).fillna(result["store_median"])
    result["promo_median"] = result["promo_median"] * rows["Promo"]
    return result


def prepare_windows(sales: pd.DataFrame, config: dict):
    window_days = config["window_days"]
    recent_days = config["recent_days"]
    year_lag_days = config["year_lag_days"]
    if not 0 < window_days <= year_lag_days or recent_days < 1:
        raise ValueError("Use a positive window no longer than the year lag and positive recent days.")

    window = pd.Timedelta(days=window_days)
    cutoff = sales["Date"].max() + pd.Timedelta(days=1) - window
    first_training_cutoff = sales["Date"].min() + pd.Timedelta(days=2 * recent_days)
    origins = pd.date_range(start=cutoff, end=first_training_cutoff, freq=-window)
    origins = origins[origins < cutoff].sort_values()
    if origins.empty:
        raise ValueError("Not enough history for a training window and a validation window.")

    def make_window(origin):
        past = sales.loc[sales["Date"] < origin]
        target = sales.loc[sales["Date"].between(origin, origin + window, inclusive="left")]
        if target.empty or not (past["Sales"] > 0).any():
            raise ValueError(f"Missing history or targets for the window starting {origin.date()}.")
        return add_history(target, past, origin, recent_days, year_lag_days)

    train = pd.concat([make_window(origin) for origin in origins], ignore_index=True)
    validation = make_window(cutoff).reset_index(drop=True)
    return train, validation, cutoff, origins

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.data import HISTORY_COLUMNS


CATEGORICAL = ["Store", "DayOfWeek", "Month"]
EXPERIMENTS = {
    "no_history": [],
    "with_history": HISTORY_COLUMNS,
    "without_trend": [name for name in HISTORY_COLUMNS if name != "trend"],
    "without_last_year": [name for name in HISTORY_COLUMNS if name != "last_year"],
    "with_weekday_mean": ["weekday_mean" if name == "weekday_median" else name for name in HISTORY_COLUMNS],
}


def fit_model(train, history_columns):
    prepare = ColumnTransformer(
        [
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numbers", StandardScaler(), ["Promo", *history_columns]),
        ],
        sparse_threshold=1.0,
    )
    model = TransformedTargetRegressor(
        regressor=make_pipeline(prepare, LinearRegression(tol=1e-6)),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    positive = train.loc[train["Sales"] > 0]
    with threadpool_limits(limits=4, user_api="blas"):
        return model.fit(positive, positive["Sales"])


def predict(model, rows):
    prediction = np.maximum(model.predict(rows), 0)
    return np.where(rows["Open"] == 0, 0, prediction)

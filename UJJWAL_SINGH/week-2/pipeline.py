from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import config


@dataclass(frozen=True)
class TrainingResult:
    model: Pipeline
    cv_rmse: float
    cv_mae: float
    cv_r2: float
    holdout_rmse: float | None = None
    holdout_mae: float | None = None
    holdout_r2: float | None = None


def load_sources(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("No data sources were provided.")

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = [column.strip().lower() for column in combined.columns]
    return combined


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()

    if config.TARGET_COLUMN in engineered.columns:
        engineered["log_price"] = np.log1p(engineered[config.TARGET_COLUMN])
        engineered["price_per_sqft"] = np.where(
            engineered["area"].gt(0),
            engineered[config.TARGET_COLUMN] / engineered["area"],
            np.nan,
        )

    if "built_year" in engineered.columns:
        engineered["age_of_house"] = config.REFERENCE_YEAR - engineered["built_year"]

    return engineered


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' is missing.")

    working = engineer_features(df)
    y = working["log_price"] if "log_price" in working.columns else np.log1p(working[config.TARGET_COLUMN])

    feature_columns = [
        column
        for column in working.columns
        if column
        not in {
            config.TARGET_COLUMN,
            "log_price",
            "price_per_sqft",
        }
    ]

    if "built_year" in feature_columns and "age_of_house" in working.columns:
        feature_columns.remove("built_year")
        feature_columns.append("age_of_house")

    X = working[feature_columns].copy()
    return X, y


def build_pipeline(feature_columns: list[str]) -> Pipeline:
    numeric_columns = [column for column in config.NUMERIC_FEATURES if column in feature_columns]
    if "age_of_house" in feature_columns:
        numeric_columns.append("age_of_house")
    categorical_columns = [column for column in config.CATEGORICAL_FEATURES if column in feature_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(X: pd.DataFrame, y: pd.Series) -> TrainingResult:
    pipeline = build_pipeline(list(X.columns))
    cv = KFold(
            n_splits=5, 
            shuffle=True, 
            random_state=config.RANDOM_STATE
        )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=config.GRID_SEARCH_PARAMS,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    cv_predictions = cross_val_score(
        best_model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    cv_mae_scores = cross_val_score(
        best_model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    cv_r2_scores = cross_val_score(
        best_model,
        X,
        y,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )

    return TrainingResult(
        model=best_model.fit(X, y),
        cv_rmse=float(-cv_predictions.mean()),
        cv_mae=float(-cv_mae_scores.mean()),
        cv_r2=float(cv_r2_scores.mean()),
    )


def prepare_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = engineer_features(df)
    columns_to_drop = [column for column in [config.TARGET_COLUMN, "log_price", "price_per_sqft"] if column in working.columns]
    return working.drop(columns=columns_to_drop)


def create_prediction_output(model: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    features = prepare_prediction_frame(data)
    predicted_log_price = model.predict(features)
    predicted_price = np.expm1(predicted_log_price)

    output = features.copy()
    output["predicted_log_price"] = predicted_log_price
    output["predicted_price"] = predicted_price
    if "area" in output.columns:
        output["predicted_price_per_sqft"] = np.where(output["area"].gt(0), predicted_price / output["area"], np.nan)
    return output


def save_artifacts(model: Pipeline, predictions: pd.DataFrame) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    predictions.to_csv(config.PREDICTIONS_PATH, index=False)


def main() -> None:
    data = load_sources(config.DATA_SOURCES)
    
    # Split data BEFORE training to avoid leakage
    train_data, test_data = train_test_split(
        data, 
        test_size=0.1, 
        random_state=config.RANDOM_STATE
    )
    
    # Train only on train_data
    training_data = engineer_features(train_data)
    X, y = build_feature_matrix(training_data)
    result = train_model(X, y)

    # Predict only on test_data (unseen by model)
    predictions = create_prediction_output(result.model, test_data)

    save_artifacts(result.model, predictions)

    holdout_message = ""
    if result.holdout_rmse is not None:
        holdout_message = (
            f", holdout RMSE={result.holdout_rmse:.4f}, "
            f"MAE={result.holdout_mae:.4f}, R2={result.holdout_r2:.4f}"
        )

    print(
        "Training complete: "
        f"CV RMSE={result.cv_rmse:.4f} \n CV MAE={result.cv_mae:.4f} \n CV R2={result.cv_r2:.4f}"
        f"{holdout_message}"
    )
    print(f"Saved model to: {config.MODEL_PATH}")
    print(f"Saved predictions to: {config.PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()

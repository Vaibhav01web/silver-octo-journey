import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

submission_ids = test["Index"]


for col in ["RoadType", "Weather"]:
    train[col] = train[col].fillna("Unknown")
    test[col] = test[col].fillna("Unknown")

temp_median = train["Temperature"].median()

train["Temperature"] = train["Temperature"].fillna(temp_median)
test["Temperature"] = test["Temperature"].fillna(temp_median)


def feature_engineering(df):

    time_split = df["timestamp"].astype(str).str.split(":", expand=True)

    df["hour"] = time_split[0].astype(int)
    df["minute"] = time_split[1].astype(int)

    # Cyclical Time Features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Time Buckets
    df["is_morning"] = (
        (df["hour"] >= 6) &
        (df["hour"] <= 11)
    ).astype(int)

    df["is_afternoon"] = (
        (df["hour"] >= 12) &
        (df["hour"] <= 16)
    ).astype(int)

    df["is_evening"] = (
        (df["hour"] >= 17) &
        (df["hour"] <= 21)
    ).astype(int)

    df["is_night"] = (
        (df["hour"] >= 22) |
        (df["hour"] <= 5)
    ).astype(int)

    df["is_peak_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 10))
        |
        ((df["hour"] >= 17) & (df["hour"] <= 20))
    ).astype(int)

    # Interaction Features
    df["lane_hour"] = (
        df["NumberofLanes"] * df["hour"]
    )

    df["lane_square"] = (
        df["NumberofLanes"] ** 2
    )

    df["hour_square"] = (
        df["hour"] ** 2
    )

    # Combined Categorical Features
    df["day_hour"] = (
        df["day"].astype(str)
        + "_"
        + df["hour"].astype(str)
    )

    df["road_lane"] = (
        df["RoadType"].astype(str)
        + "_"
        + df["NumberofLanes"].astype(str)
    )

    return df


train = feature_engineering(train)
test = feature_engineering(test)

train["temp_lane"] = (
    train["Temperature"] *
    train["NumberofLanes"]
)

test["temp_lane"] = (
    test["Temperature"] *
    test["NumberofLanes"]
)

train["temp_hour"] = (
    train["Temperature"] *
    train["hour"]
)

test["temp_hour"] = (
    test["Temperature"] *
    test["hour"]
)


train.drop(
    ["Index", "timestamp"],
    axis=1,
    inplace=True
)

test.drop(
    ["Index", "timestamp"],
    axis=1,
    inplace=True
)


X = train.drop("demand", axis=1)
y = train["demand"]

X_test = test.copy()

cat_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

for col in cat_features:
    X[col] = X[col].fillna("Unknown").astype(str)
    X_test[col] = X_test[col].fillna("Unknown").astype(str)

X = X.fillna(0)
X_test = X_test.fillna(0)

print("\nCategorical Features:")
print(cat_features)
kf = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

oof = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))

scores = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):

    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}")
    print(f"{'='*60}")

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]

    model = CatBoostRegressor(
        iterations=6000,
        learning_rate=0.015,
        depth=9,
        l2_leaf_reg=8,
        bagging_temperature=1,
        loss_function="RMSE",
        eval_metric="R2",
        random_seed=42,
        verbose=500
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=500
    )

    valid_preds = model.predict(X_valid)

    fold_r2 = r2_score(
        y_valid,
        valid_preds
    )

    print(f"Fold {fold+1} R2 = {fold_r2:.6f}")

    scores.append(fold_r2)

    oof[valid_idx] = valid_preds

    test_predictions += (
        model.predict(X_test) / 10
    )

overall_r2 = r2_score(
    y,
    oof
)

print("\n" + "="*60)
print("CV R2 :", overall_r2)
print("Mean Fold R2 :", np.mean(scores))
print("="*60)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features")
print(importance.head(20))


submission = pd.DataFrame({
    "Index": submission_ids,
    "demand": test_predictions
})

submission.to_csv(
    "submission_v2.csv",
    index=False
)

print("\nsubmission_v2.csv generated successfully!")
print(submission.head())
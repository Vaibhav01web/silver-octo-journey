import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# ── CONFIG ────────────────────────────────────────────────────
USE_LOG_TARGET = True   # Try True first; flip to False if R² drops
N_SPLITS       = 10
SEED           = 42
# ─────────────────────────────────────────────────────────────

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
submission_ids = test["Index"]

# ── FILL MISSING ──────────────────────────────────────────────
for col in ["RoadType", "Weather"]:
    train[col] = train[col].fillna("Unknown")
    test[col]  = test[col].fillna("Unknown")

temp_median = train["Temperature"].median()
train["Temperature"] = train["Temperature"].fillna(temp_median)
test["Temperature"]  = test["Temperature"].fillna(temp_median)

# ── WEATHER SEVERITY MAP ──────────────────────────────────────
weather_severity = {
    "Clear": 0, "Sunny": 0,
    "Cloudy": 1, "Overcast": 1,
    "Fog": 2, "Mist": 2,
    "Rain": 3, "Drizzle": 3,
    "HeavyRain": 4, "Snow": 4, "Storm": 5,
    "Unknown": 1
}

road_order = {"Local": 0, "Arterial": 1, "Highway": 2, "Expressway": 3, "Unknown": 1}

# ── FEATURE ENGINEERING ───────────────────────────────────────
def feature_engineering(df):
    ts = df["timestamp"].astype(str).str.split(":", expand=True)
    df["hour"]   = ts[0].astype(int)
    df["minute"] = ts[1].astype(int)

    # Cyclical
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]   / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]   / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    # Time of day buckets
    df["is_morning"]   = ((df["hour"] >= 6)  & (df["hour"] <= 11)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] <= 16)).astype(int)
    df["is_evening"]   = ((df["hour"] >= 17) & (df["hour"] <= 21)).astype(int)
    df["is_night"]     = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_peak_hour"] = (
        ((df["hour"] >= 7)  & (df["hour"] <= 10)) |
        ((df["hour"] >= 17) & (df["hour"] <= 20))
    ).astype(int)
    df["is_late_night"] = ((df["hour"] >= 0) & (df["hour"] <= 4)).astype(int)

    # Polynomial / interaction
    df["lane_hour"]   = df["NumberofLanes"] * df["hour"]
    df["lane_square"] = df["NumberofLanes"] ** 2
    df["hour_square"] = df["hour"] ** 2
    df["hour_cube"]   = df["hour"] ** 3

    # Temperature features
    df["temp_lane"]    = df["Temperature"] * df["NumberofLanes"]
    df["temp_hour"]    = df["Temperature"] * df["hour"]
    df["temp_peak"]    = df["Temperature"] * df["is_peak_hour"]
    df["temp_squared"] = df["Temperature"] ** 2

    # Weather severity (numerical)
    df["weather_severity"] = df["Weather"].map(weather_severity).fillna(1)
    df["road_order"]       = df["RoadType"].map(road_order).fillna(1)

    # Cross features
    df["road_lane"]     = df["RoadType"].astype(str) + "_" + df["NumberofLanes"].astype(str)
    df["day_hour"]      = df["day"].astype(str)      + "_" + df["hour"].astype(str)
    df["weather_hour"]  = df["Weather"].astype(str)  + "_" + df["hour"].astype(str)
    df["road_hour"]     = df["RoadType"].astype(str) + "_" + df["hour"].astype(str)
    df["road_peak"]     = df["RoadType"].astype(str) + "_" + df["is_peak_hour"].astype(str)
    df["weather_road"]  = df["Weather"].astype(str)  + "_" + df["RoadType"].astype(str)

    # Severity × lanes
    df["severity_lane"] = df["weather_severity"] * df["NumberofLanes"]
    df["severity_peak"] = df["weather_severity"] * df["is_peak_hour"]

    # Minute bucket
    df["minute_bucket"] = (df["minute"] // 15).astype(int)   # 0,1,2,3
    df["time_of_day"]   = df["hour"] * 4 + df["minute_bucket"]  # 0‑95 granular slot

    return df

train = feature_engineering(train)
test  = feature_engineering(test)

# ── DROP COLUMNS ──────────────────────────────────────────────
train.drop(["Index", "timestamp"], axis=1, inplace=True)
test.drop(["Index", "timestamp"],  axis=1, inplace=True)

X      = train.drop("demand", axis=1)
y      = train["demand"]
X_test = test.copy()

cat_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
for col in cat_features:
    X[col]      = X[col].fillna("Unknown").astype(str)
    X_test[col] = X_test[col].fillna("Unknown").astype(str)

X      = X.fillna(0)
X_test = X_test.fillna(0)

print("Categorical Features:", cat_features)
print("Total features:", X.shape[1])

# ── OPTIONAL LOG TRANSFORM ────────────────────────────────────
if USE_LOG_TARGET:
    y_fit = np.log1p(y)
else:
    y_fit = y.copy()

# ── CROSS-VALIDATION ──────────────────────────────────────────
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof              = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))
scores           = []

CATBOOST_PARAMS = dict(
    iterations          = 8000,
    learning_rate       = 0.01,
    depth               = 10,
    l2_leaf_reg         = 5,
    bagging_temperature = 0.8,
    subsample           = 0.85,          # row sampling per tree
    colsample_bylevel   = 0.75,          # feature sampling per level
    min_data_in_leaf    = 20,
    loss_function       = "RMSE",
    eval_metric         = "R2",
    random_seed         = SEED,
    verbose             = 500,
    thread_count        = -1,            # use all CPU cores
)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    print(f"\n{'='*60}\nFOLD {fold+1}\n{'='*60}")

    X_tr, y_tr = X.iloc[train_idx], y_fit.iloc[train_idx]
    X_vl, y_vl = X.iloc[valid_idx], y_fit.iloc[valid_idx]

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(
        X_tr, y_tr,
        cat_features=cat_features,
        eval_set=(X_vl, y_vl),
        use_best_model=True,
        early_stopping_rounds=300,
    )

    vp = model.predict(X_vl)
    oof[valid_idx] = vp

    fold_r2 = r2_score(y_vl, vp)
    print(f"Fold {fold+1} R2 (transformed space) = {fold_r2:.6f}")
    scores.append(fold_r2)

    test_predictions += model.predict(X_test) / N_SPLITS

# ── INVERSE TRANSFORM ─────────────────────────────────────────
if USE_LOG_TARGET:
    oof_orig  = np.expm1(oof)
    pred_orig = np.expm1(test_predictions)
else:
    oof_orig  = oof
    pred_orig = test_predictions

# Clip negatives just in case
oof_orig  = np.clip(oof_orig,  0, None)
pred_orig = np.clip(pred_orig, 0, None)

overall_r2 = r2_score(y, oof_orig)
print(f"\n{'='*60}")
print(f"OOF R2 (original scale) : {overall_r2:.6f}")
print(f"Mean Fold R2            : {np.mean(scores):.6f}")
print(f"Std  Fold R2            : {np.std(scores):.6f}")
print("="*60)

# ── FEATURE IMPORTANCE ────────────────────────────────────────
importance = (
    pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    .sort_values("Importance", ascending=False)
)
print("\nTop 20 Features")
print(importance.head(20).to_string(index=False))

# ── SUBMISSION ────────────────────────────────────────────────
submission = pd.DataFrame({"Index": submission_ids, "demand": pred_orig})
submission.to_csv("submission_v3.csv", index=False)
print("\nsubmission_v3.csv generated successfully!")
print(submission.head())
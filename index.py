import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

USE_XGB = False
DATE_FORMAT = "%d/%m/%y"

def load_data(path):
    df = pd.read_csv(path)
    if "Date" in df.columns:
        parsed = pd.to_datetime(df["Date"], errors="coerce", format=DATE_FORMAT)
        # Fallback if the format doesn't match most rows.
        if parsed.isna().mean() > 0.5:
            parsed = pd.to_datetime(df["Date"], errors="coerce")
        df["Date"] = parsed
    else:
        df["Date"] = pd.NaT
    if "FTR" in df.columns:
        df["FTR"] = df["FTR"].astype(str).str.strip()
        df.loc[df["FTR"] == "", "FTR"] = np.nan
        df = df.dropna(subset=["FTR"])
    for c in ["HomeTeam", "AwayTeam", "Referee", "HTR"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def _get_numeric(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype=float)

def build_features(df):
    df = df.copy()
    if "Date" in df.columns:
        df["MatchMonth"] = df["Date"].dt.month
        df["MatchDay"] = df["Date"].dt.day
        df["MatchWeekday"] = df["Date"].dt.weekday
    else:
        df["MatchMonth"] = np.nan
        df["MatchDay"] = np.nan
        df["MatchWeekday"] = np.nan
    fthg = _get_numeric(df, "FTHG")
    ftag = _get_numeric(df, "FTAG")
    hthg = _get_numeric(df, "HTHG")
    htag = _get_numeric(df, "HTAG")
    hs = _get_numeric(df, "HS")
    hst = _get_numeric(df, "HST")
    a_s = _get_numeric(df, "AS")
    ast = _get_numeric(df, "AST")
    hy = _get_numeric(df, "HY")
    ay = _get_numeric(df, "AY")
    hr = _get_numeric(df, "HR")
    ar = _get_numeric(df, "AR")
    df["FT_GoalDiff"] = fthg - ftag
    df["HT_GoalDiff"] = hthg - htag
    df["Home_ShotEfficiency"] = hst / (hs + 1)
    df["Away_ShotEfficiency"] = ast / (a_s + 1)
    df["YellowDiff"] = hy - ay
    df["RedDiff"] = hr - ar
    y = df["FTR"] if "FTR" in df.columns else None
    X = df.drop(columns=["Date", "FTR"], errors="ignore")
    return X, y

def train_model(path):
    df = load_data(path)
    X, y = build_features(df)
    if y is None or y.empty:
        raise ValueError("FTR column missing or empty after cleaning; cannot train.")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    cat_cols = ["HomeTeam", "AwayTeam", "Referee", "HTR"]
    cat_cols = [c for c in cat_cols if c in X.columns]
    for c in cat_cols:
        X[c] = X[c].astype(str)
    num_cols = [c for c in X.columns if c not in cat_cols]

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_cols),
            ("num", num_transformer, num_cols)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=450,
        max_depth=22,
        random_state=42
    )

    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    class_counts = pd.Series(y_encoded).value_counts()
    stratify = y_encoded if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=stratify
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    labels = list(range(len(label_encoder.classes_)))
    print(classification_report(
        y_test,
        preds,
        labels=labels,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    joblib.dump({"pipeline": pipeline, "label_encoder": label_encoder}, "premier_model.joblib")
    print("Model saved as premier_model.joblib")

def predict_single(home, away):
    artifact = joblib.load("premier_model.joblib")
    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]

    data = {
        "HomeTeam": [home], "AwayTeam": [away], "Referee": ["Unknown"], "HTR": ["D"],
        "FTHG": [0], "FTAG": [0], "HTHG": [0], "HTAG": [0],
        "HS": [10], "AS": [10], "HST": [5], "AST": [5],
        "HF": [10], "AF": [10], "HC": [4], "AC": [4],
        "HY": [1], "AY": [1], "HR": [0], "AR": [0],
        "Date": [pd.Timestamp("2024-01-01")]
    }

    df = pd.DataFrame(data)
    X, _ = build_features(df)

    probs = pipeline.predict_proba(X)[0]
    pred = pipeline.predict(X)[0]
    label = label_encoder.inverse_transform([pred])[0]

    classes = list(label_encoder.classes_)
    prob_map = dict(zip(classes, probs))

    home_prob = prob_map.get("H", 0) * 100
    draw_prob = prob_map.get("D", 0) * 100
    away_prob = prob_map.get("A", 0) * 100

    print("\n==============================")
    print(f"   {home} vs {away}")
    print("==============================\n")
    print(f"{home} (H): {home_prob:.2f}%")
    print(f"{away} (A): {away_prob:.2f}%\n")
    print(f"Draw: {draw_prob:.2f}%\n")

    return label, probs

if __name__ == "__main__":
    train_model("/Users/shinobi/bibek_code/match _prieditor/prem_matches.csv")

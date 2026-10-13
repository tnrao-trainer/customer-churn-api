"""
Train a customer churn model by loading data from data/customer_churn.csv

Expected CSV columns:
CustomerID, Age, Tenure, MonthlyCharges, TotalCharges, Contract,
InternetService, SupportCalls, PaymentMethod, Churn
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# Paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"


def load_data() -> pd.DataFrame:
    """Load customer churn data from CSV file."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found at: {DATA_PATH}\n"
            "Please place your dataset at data/customer_churn.csv"
        )

    # >>> This is the line that loads data from the CSV file <<<
    df = pd.read_csv("data/customer_churn.csv")

    print(f"Loaded data from: {DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def train():
    print("Loading data from CSV...")
    df = load_data()

    # Normalize column names to lowercase with underscores for consistency
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map possible column name variations
    rename_map = {
        "customerid": "customer_id",
        "monthlycharges": "monthly_charges",
        "totalcharges": "total_charges",
        "internetservice": "internet_service",
        "supportcalls": "support_calls",
        "paymentmethod": "payment_method",
    }
    df = df.rename(columns=rename_map)

    print("\nSample rows:")
    print(df.head(4).to_string(index=False))

    # Features used for training (CustomerID is excluded)
    feature_cols = [
        "age",
        "tenure",
        "monthly_charges",
        "total_charges",
        "contract",
        "internet_service",
        "support_calls",
        "payment_method",
    ]

    # Ensure required columns exist
    missing = [c for c in feature_cols + ["churn"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    X = df[feature_cols]
    # Convert Churn Yes/No to 1/0
    y = df["churn"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0})
    if y.isna().any():
        # fallback if already numeric
        y = df["churn"].astype(int)

    print(f"\nChurn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = ["age", "tenure", "monthly_charges", "total_charges", "support_calls"]
    categorical_features = ["contract", "internet_service", "payment_method"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")
    print("Done.")


if __name__ == "__main__":
    train()

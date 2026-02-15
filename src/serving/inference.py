"""
INFERENCE PIPELINE - Production ML Model Serving
================================================

This module provides the core inference functionality for the Telco Churn model.

Responsibilities:
1. Load exported production model artifacts
2. Ensure feature consistency with training
3. Apply preprocessing pipeline (if saved)
4. Generate business-friendly predictions

Architecture:
Training → MLflow tracking
Export → artifacts/ bundle
Serving → Loads directly from artifacts (no MLflow dependency)

This ensures:
- Stable Docker deployments
- No run_id dependency
- No MLflow requirement in production
- Protection against train/serve skew
"""

import os
import json
import joblib
import pandas as pd


# ==========================================================
# === MODEL & ARTIFACT LOADING
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../artifacts"))

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model", "model.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessing.pkl")
FEATURE_COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.json")


# --- Load Model ---
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


# --- Load Feature Schema ---
try:
    with open(FEATURE_COLUMNS_PATH, "r") as f:
        FEATURE_COLS = json.load(f)
    print(f"Loaded {len(FEATURE_COLS)} feature columns")
except Exception as e:
    raise RuntimeError(f"Failed to load feature columns: {e}")


# --- Load Preprocessing Pipeline (Optional but Recommended) ---
try:
    preprocessing = joblib.load(PREPROCESSOR_PATH)
    print("Preprocessing pipeline loaded")
except Exception:
    preprocessing = None
    print("No preprocessing.pkl found — using manual transform fallback")


# ==========================================================
# === MANUAL FALLBACK TRANSFORM (Only used if no pipeline)
# ==========================================================

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Manual fallback transformation logic.
    Only used if no preprocessing pipeline was exported.
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    # --- Numeric Coercion ---
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # --- Binary Encoding ---
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

    # --- One-Hot Encoding ---
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # --- Boolean to Int ---
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # --- Feature Alignment ---
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df


# ==========================================================
# === PREDICTION FUNCTION
# ==========================================================

def predict(input_dict: dict) -> str:
    """
    Main inference entry point.

    Steps:
    1. Convert raw input to DataFrame
    2. Apply preprocessing pipeline (preferred)
    3. Align features
    4. Generate prediction
    5. Return business-friendly output
    """

    # Step 1: Create single-row DataFrame
    df = pd.DataFrame([input_dict])

    # Step 2: Apply preprocessing
    if preprocessing:
        df_enc = preprocessing.transform(df)
        df_enc = pd.DataFrame(df_enc, columns=FEATURE_COLS)
    else:
        df_enc = _serve_transform(df)

    # Step 3: Generate prediction
    try:
        preds = model.predict(df_enc)

        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        result = preds[0] if isinstance(preds, (list, tuple)) else preds

    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    # Step 4: Business output
    return "Likely to churn" if result == 1 else "Not likely to churn"

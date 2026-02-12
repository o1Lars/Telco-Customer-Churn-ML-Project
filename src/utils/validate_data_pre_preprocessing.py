from typing import Tuple, List
import pandas as pd

def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Raw data validation for Telco Customer Churn dataset (BEFORE preprocessing).

    Uses pandas only.
    Returns:
        (success: bool, failed_checks: List[str])
    """

    print("Starting raw data validation (pandas-based)...")

    failed_checks = []

    # ==================================================
    # 1. SCHEMA VALIDATION
    # ==================================================
    print("Validating schema and required columns...")

    required_columns = [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    for col in required_columns:
        if col not in df.columns:
            failed_checks.append(f"missing_column_{col}")

    if "customerID" in df.columns:
        if df["customerID"].isnull().any():
            failed_checks.append("customerID_contains_null")

    # ==================================================
    # 2. BUSINESS LOGIC VALIDATION
    # ==================================================
    print("Validating business logic constraints...")

    def check_allowed_values(column, allowed_values):
        if column in df.columns:
            invalid = ~df[column].isin(allowed_values)
            if invalid.any():
                failed_checks.append(f"{column}_contains_invalid_values")

    check_allowed_values("gender", ["Male", "Female"])
    check_allowed_values("Partner", ["Yes", "No"])
    check_allowed_values("Dependents", ["Yes", "No"])
    check_allowed_values("PhoneService", ["Yes", "No"])
    check_allowed_values("Contract", ["Month-to-month", "One year", "Two year"])
    check_allowed_values("InternetService", ["DSL", "Fiber optic", "No"])

    # ==================================================
    # 3. NUMERIC COLUMN PRESENCE (but do NOT fail for blanks)
    # ==================================================
    print("Checking numeric column presence...")

    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_columns:
        if col not in df.columns:
            continue
        # Only check if column is entirely missing or all non-numeric
        if df[col].isnull().all():
            failed_checks.append(f"{col}_all_missing")

    # ==================================================
    # 4. DATA CONSISTENCY CHECK (optional lenient)
    # ==================================================
    print("Validating basic numeric consistency (lenient)...")

    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
        total = pd.to_numeric(df["TotalCharges"], errors="coerce")
        monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

        # Ignore NaNs for raw data
        comparison = total >= monthly
        if comparison.dropna().empty:
            # If all are NaN, skip consistency check
            pass
        else:
            valid_ratio = comparison.dropna().mean()
            if valid_ratio < 0.95:
                failed_checks.append("TotalCharges_not_consistent_with_MonthlyCharges")

    # ==================================================
    # 5. SUMMARY
    # ==================================================
    success = len(failed_checks) == 0

    if success:
        print("Raw data validation PASSED")
    else:
        print("Raw data validation completed with warnings (expected for raw data)")
        print("Failed checks:", failed_checks)

    return success, failed_checks

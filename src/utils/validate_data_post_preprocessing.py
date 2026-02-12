import pandas as pd

def validate_data(df: pd.DataFrame, target_col: str = "Churn") -> tuple:
    """
    Validates dataframe AFTER preprocessing has been applied.

    Returns:
        Tuple of (success: bool, failed_checks: list).
        - success = True if no failed checks
        - failed_checks = list of failed validation messages
    """

    failed_checks = []

    # --------------------------------------------------
    # 1. Basic sanity checks
    # --------------------------------------------------
    if df.empty:
        failed_checks.append("dataset_is_empty")

    if target_col not in df.columns:
        failed_checks.append("missing_target_column")

    # --------------------------------------------------
    # 2. Target validation
    # --------------------------------------------------
    if target_col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            failed_checks.append("target_not_numeric")

        unique_vals = set(df[target_col].unique())
        if not unique_vals.issubset({0, 1}):
            failed_checks.append("target_not_binary_0_1")

    # --------------------------------------------------
    # 3. ID column should be removed
    # --------------------------------------------------
    forbidden_id_cols = ["customerID", "CustomerID", "customer_id"]
    for col in forbidden_id_cols:
        if col in df.columns:
            failed_checks.append("id_column_not_removed")

    # --------------------------------------------------
    # 4. TotalCharges must be numeric and non-null
    # --------------------------------------------------
    if "TotalCharges" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["TotalCharges"]):
            failed_checks.append("TotalCharges_not_numeric")

        if df["TotalCharges"].isnull().any():
            failed_checks.append("TotalCharges_contains_null")

    # --------------------------------------------------
    # 5. SeniorCitizen should be int 0/1
    # --------------------------------------------------
    if "SeniorCitizen" in df.columns:
        if not pd.api.types.is_integer_dtype(df["SeniorCitizen"]):
            failed_checks.append("SeniorCitizen_not_integer")

        unique_vals = set(df["SeniorCitizen"].unique())
        if not unique_vals.issubset({0, 1}):
            failed_checks.append("SeniorCitizen_not_binary")

    # --------------------------------------------------
    # 6. No numeric columns should contain nulls
    # --------------------------------------------------
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if df[numeric_cols].isnull().any().any():
        failed_checks.append("numeric_columns_contain_null")

    # --------------------------------------------------
    # 7. No completely empty columns
    # --------------------------------------------------
    for col in df.columns:
        if df[col].isnull().all():
            failed_checks.append(f"{col}_completely_null")

    # --------------------------------------------------
    # 8. Return success + failed checks
    # --------------------------------------------------
    success = len(failed_checks) == 0
    return success, failed_checks

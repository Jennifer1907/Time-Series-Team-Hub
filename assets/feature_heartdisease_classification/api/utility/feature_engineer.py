import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

def preprocess_and_engineer_features(df: pd.DataFrame):
    """
    Expected columns:
    ['age','height','weight','gender','ap_hi','ap_lo','cholesterol','gluc',
    'smoke','alco','active','cardio']
    """
    df = df.copy()

    target_col = 'cardio'
    assert target_col in df.columns, f"Missing target column '{target_col}'"
    y = df[target_col].astype(object)
    X = df.drop(columns=[target_col, "id"]).copy()


    num_cols = ['age','height','weight','ap_hi','ap_lo']
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce')

    num_imputer = SimpleImputer(strategy='median')
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

    swap = X['ap_hi'] < X['ap_lo']
    if swap.any():
        X.loc[swap, ['ap_hi', 'ap_lo']] = X.loc[swap, ['ap_lo', 'ap_hi']].values

    for c in ['height','weight','ap_hi','ap_lo','age']:
        lo, hi = X[c].quantile([0.01, 0.99]).values
        X[c] = X[c].clip(lo, hi)

    X['age_years'] = np.floor(X['age'] / 365.25).astype(int)

    h_m = X['height'] / 100.0
    X['bmi'] = X['weight'] / (h_m**2 + 1e-9)

    X['pp']  = X['ap_hi'] - X['ap_lo']                    # pulse pressure
    X['map'] = X['ap_lo'] + X['pp'] / 3.0                 # mean arterial pressure
    X['sbp_dbp_ratio'] = X['ap_hi'] / (X['ap_lo'] + 1e-6)

    X['is_hypertensive'] = ((X['ap_hi'] >= 140) | (X['ap_lo'] >= 90)).astype(int)
    X['prehypertensive'] = (((X['ap_hi'] >= 120) & (X['ap_hi'] < 140)) |
                            ((X['ap_lo'] >= 80)  & (X['ap_lo'] < 90))).astype(int)
    X['wide_pp'] = (X['pp'] >= 60).astype(int)

    # cholesterol: 1 normal, 2 above, 3 well above (keep as ordinal)
    # glucose:     1 normal, 2 above, 3 well above (keep as ordinal)
    for c in ['cholesterol','gluc','gender','smoke','alco','active']:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')

    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[['cholesterol','gluc','gender','smoke','alco','active']] = cat_imputer.fit_transform(
        X[['cholesterol','gluc','gender','smoke','alco','active']]
    )

    X['chol_high'] = (X['cholesterol'] >= 2).astype(int)
    X['gluc_high'] = (X['gluc'] >= 2).astype(int)

    # Behavior summary
    X['risk_behaviors'] = X['smoke'] + X['alco'] + (1 - X['active'])

    X['age_bin'] = pd.cut(
        X['age_years'], bins=[0, 39, 49, 59, 69, 120], labels=[0,1,2,3,4]
    ).astype(int)

    X['bmi_class'] = pd.cut(
        X['bmi'], bins=[-np.inf, 18.5, 25, 30, np.inf], labels=[0,1,2,3]
    ).astype(int)

    X['bp_category'] = pd.cut(
        X['ap_hi'], bins=[-np.inf, 120, 130, 140, np.inf], labels=[0,1,2,3]
    ).astype(int)

    X['bmi_x_age'] = X['bmi'] * X['age_years']
    X['pp_x_age']  = X['pp']  * X['age_years']
    X['htn_x_chol'] = X['is_hypertensive'] * X['chol_high']

    to_scale = [
        'age','age_years','height','weight','ap_hi','ap_lo',
        'bmi','pp','map','sbp_dbp_ratio','bmi_x_age','pp_x_age'
    ]
    # Ensure all exist
    to_scale = [c for c in to_scale if c in X.columns]

    scaler = RobustScaler()
    X_scaled = X.copy()
    X_scaled[to_scale] = scaler.fit_transform(X_scaled[to_scale])


    print(f"Final feature shape: {X_scaled.shape}")
    return X_scaled, y, scaler
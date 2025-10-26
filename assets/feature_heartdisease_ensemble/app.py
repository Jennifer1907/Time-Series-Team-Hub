import os
import io
import json
import warnings
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    roc_auc_score, log_loss, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from mealpy.utils.space import FloatVar
from mealpy.bio_based import TSA

warnings.filterwarnings("ignore")
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)

st.set_page_config(page_title="Heart Ensemble (Stacking + TSA)", layout="wide")

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("⚙️ Cấu hình")
data_source = st.sidebar.radio(
    "Nguồn dữ liệu",
    ["Upload CSV", "URL CSV", "Local file path"],
    index=0
)

uploaded = None
csv_url = ""
csv_path = ""
if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Tải lên cleveland.csv", type=["csv"])
elif data_source == "URL CSV":
    csv_url = st.sidebar.text_input("Dán URL đến cleveland.csv")
else:
    csv_path = st.sidebar.text_input("Đường dẫn file cục bộ (ví dụ: ./cleveland.csv)", value="cleveland.csv")

K_features = st.sidebar.number_input("Top K features (DT selector)", min_value=3, max_value=40, value=10, step=1)
n_splits = st.sidebar.slider("K-fold (OOF) cho base models", 3, 10, 5, 1)
do_calib = st.sidebar.checkbox("Calibrate xác suất base models", value=True)
calib_method = st.sidebar.selectbox("Calibration method", ["sigmoid", "isotonic"], index=0)
calib_cv = st.sidebar.slider("CV trong calibration", 2, 5, 3, 1)

tsa_epoch = st.sidebar.slider("TSA epoch", 50, 500, 250, 25)
tsa_pop = st.sidebar.slider("TSA population size", 10, 100, 40, 5)

variant = st.sidebar.selectbox(
    "Dataset variant",
    ["Original", "FE", "Original + DT", "FE + DT"]
)

show_mi = st.sidebar.checkbox("Hiển thị MI chart (Train)", value=False)
show_coef = st.sidebar.checkbox("Hiển thị |coef| của Stacking-LR", value=True)
show_shap = st.sidebar.checkbox("Hiển thị SHAP (beeswarm & waterfall)", value=True)
show_tsa = st.sidebar.checkbox("Hiển thị trọng số TSA & local contributions", value=True)
sample_idx = st.sidebar.number_input("Sample index (XAI)", min_value=0, value=0, step=1)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Chạy huấn luyện & đánh giá")

# =========================
# Original code pieces refactored as functions
# =========================

TARGET = "target"
COLUMNS = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal','target']
numeric_cols = ['age','trestbps','chol','thalach','oldpeak']
categorical_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

class AddNewFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        self.columns_ = X.columns
        self.new_features_ = []
        if {'chol','age'} <= set(X.columns): self.new_features_.append('chol_per_age')
        if {'trestbps','age'} <= set(X.columns): self.new_features_.append('bps_per_age')
        if {'thalach','age'} <= set(X.columns): self.new_features_.append('hr_ratio')
        if 'age' in X.columns: self.new_features_.append('age_bin')
        return self
    def transform(self, X, y=None):
        df = X.copy()
        if {'chol','age'} <= set(df.columns): df['chol_per_age'] = df['chol']/df['age']
        if {'trestbps','age'} <= set(df.columns): df['bps_per_age'] = df['trestbps']/df['age']
        if {'thalach','age'} <= set(df.columns): df['hr_ratio'] = df['thalach']/df['age']
        if 'age' in df.columns:
            df['age_bin'] = pd.cut(df['age'], bins=5, labels=False).astype('category')
        return df
    def get_feature_names_out(self, input_features=None):
        return list(self.columns_) + getattr(self, "new_features_", [])

def base_models_list():
    return [
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_split=2, min_samples_leaf=1,
            max_features='sqrt', bootstrap=True, random_state=42, class_weight=None
        )),
        ("cat", CatBoostClassifier(
            iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, rsm=0.8,
            bootstrap_type="Bayesian", bagging_temperature=0.5, loss_function="Logloss",
            eval_metric="AUC", auto_class_weights="Balanced",
            random_state=42, verbose=0, allow_writing_files=False, thread_count=-1
        )),
        ("xgb", XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.1,
            subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
            eval_metric="logloss", random_state=42
        )),
        ("lgbm", LGBMClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42
        )),
    ]

@st.cache_data(show_spinner=False)
def load_data(uploaded_file, url: str, path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
    elif url:
        df = pd.read_csv(url, header=None)
    else:
        df = pd.read_csv(path, header=None)
    df.columns = COLUMNS
    for c in ['age','trestbps','chol','thalach','oldpeak','ca','thal']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['target'] = (df['target'] > 0).astype(int)
    return df

def build_preprocessors():
    cat_proc = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    num_proc = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler())
    ])
    preprocess = ColumnTransformer([
        ('num', num_proc, numeric_cols),
        ('cat', cat_proc, categorical_cols)
    ])
    return preprocess

def preprocess_split_variants(raw: pd.DataFrame, K_features: int):
    # Split
    X_all = raw.drop(columns=[TARGET])
    y_all = raw[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # ----- Original processed -----
    preprocess = build_preprocessors()
    raw_pipe = Pipeline([('preprocess', preprocess)])
    Xtr = raw_pipe.fit_transform(X_train, y_train)
    Xva = raw_pipe.transform(X_val)
    Xte = raw_pipe.transform(X_test)

    # names
    pre_feat_names = []
    for name, transformer, cols in preprocess.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            pre_feat_names.extend(transformer.get_feature_names_out(cols))
        else:
            pre_feat_names.extend(cols)

    Xtr_df = pd.DataFrame(Xtr, columns=pre_feat_names, index=X_train.index)
    Xva_df = pd.DataFrame(Xva, columns=pre_feat_names, index=X_val.index)
    Xte_df = pd.DataFrame(Xte, columns=pre_feat_names, index=X_test.index)

    # DT selector on original
    dt_sel = Pipeline([
        ('preprocess', preprocess),
        ('dt', DecisionTreeClassifier(random_state=42))
    ])
    dt_sel.fit(X_train, y_train)
    fi = pd.Series(dt_sel.named_steps['dt'].feature_importances_, index=pre_feat_names)
    top_dt = fi.sort_values(ascending=False).head(K_features).index.tolist()

    X_dt_tr = Xtr_df[top_dt]
    X_dt_va = Xva_df[top_dt]
    X_dt_te = Xte_df[top_dt]

    # ----- FE processed -----
    gen_num = ['chol_per_age', 'bps_per_age', 'hr_ratio']
    gen_cat = ['age_bin']
    all_nums = [c for c in numeric_cols] + gen_num
    all_cats = [c for c in categorical_cols] + gen_cat

    num_proc = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat_proc = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    pre = ColumnTransformer([
        ('num', num_proc, all_nums),
        ('cat', cat_proc, all_cats)
    ], verbose_feature_names_out=False).set_output(transform='pandas')

    fe_pre = Pipeline([
        ('add_new', AddNewFeaturesTransformer()),
        ('pre', pre),
    ]).set_output(transform='pandas')

    Xt_tr = fe_pre.fit_transform(X_train, y_train)
    Xt_va = fe_pre.transform(X_val)
    Xt_te = fe_pre.transform(X_test)

    # drop constant cols
    nz_cols = Xt_tr.columns[Xt_tr.nunique(dropna=False) > 1]
    Xt_tr, Xt_va, Xt_te = Xt_tr[nz_cols], Xt_va[nz_cols], Xt_te[nz_cols]

    # MI for info (optional plot)
    ohe = fe_pre.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
    cat_names = list(ohe.get_feature_names_out(all_cats))
    is_discrete = np.array([c in cat_names for c in Xt_tr.columns], dtype=bool)
    mi = mutual_info_classif(Xt_tr.values, y_train.values, discrete_features=is_discrete, random_state=42)
    mi_series = pd.Series(mi, index=Xt_tr.columns).sort_values(ascending=False)

    # FE + DT selector
    dt_fe = Pipeline([('preprocess', fe_pre), ('dt', DecisionTreeClassifier(random_state=42))])
    dt_fe.fit(X_train, y_train)
    pipe_feat_names = dt_fe.named_steps['preprocess'].get_feature_names_out()
    fi_fe = pd.Series(dt_fe.named_steps['dt'].feature_importances_, index=pipe_feat_names).sort_values(ascending=False)
    top_dt_fe = fi_fe.head(K_features).index.tolist()

    X_fe_dt_tr = Xt_tr[top_dt_fe]
    X_fe_dt_va = Xt_va[top_dt_fe]
    X_fe_dt_te = Xt_te[top_dt_fe]

    return {
        "splits": {
            "Original": (Xtr_df, Xva_df, Xte_df, y_train, y_val, y_test),
            "FE": (Xt_tr, Xt_va, Xt_te, y_train, y_val, y_test),
            "Original + DT": (X_dt_tr, X_dt_va, X_dt_te, y_train, y_val, y_test),
            "FE + DT": (X_fe_dt_tr, X_fe_dt_va, X_fe_dt_te, y_train, y_val, y_test),
        },
        "mi_series": mi_series
    }

def train_bases_oof_and_refit(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, X_test: pd.DataFrame,
    n_splits=5, calibrate=True, calib_method="sigmoid", calib_cv=3, seed=42
):
    bases = base_models_list()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_models = len(bases)

    P_train_oof = np.zeros((len(X_train), n_models), dtype=float)
    P_val = np.zeros((len(X_val), n_models), dtype=float)
    P_test = np.zeros((len(X_test), n_models), dtype=float)

    fitted = []

    for j, (_, base) in enumerate(bases):
        oof_col = np.zeros(len(X_train), dtype=float)
        for tr_idx, va_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]

            if calibrate:
                model = CalibratedClassifierCV(clone(base), method=calib_method, cv=calib_cv)
            else:
                model = clone(base)
            model.fit(X_tr, y_tr)
            oof_col[va_idx] = model.predict_proba(X_va)[:, 1]

        P_train_oof[:, j] = oof_col

        if calibrate:
            model_full = CalibratedClassifierCV(clone(base), method=calib_method, cv=calib_cv)
        else:
            model_full = clone(base)
        model_full.fit(X_train, y_train)
        P_val[:, j] = model_full.predict_proba(X_val)[:, 1]
        P_test[:, j] = model_full.predict_proba(X_test)[:, 1]
        fitted.append(model_full)

    base_names = [name for name, _ in bases]
    return P_train_oof, P_val, P_test, fitted, base_names

def metrics_report(y_true, p):
    # chống TH dự đoán hằng số gây lỗi log_loss
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return {
        "AUC": float(roc_auc_score(y_true, p)),
        "LogLoss": float(log_loss(y_true, p)),
        "AP": float(average_precision_score(y_true, p)),
    }

def equal_weight_probs(P):
    return P.mean(axis=1)

def stack_lr_fit(P_train_oof, y_train):
    meta = LogisticRegression(C=1.0, penalty="l2", max_iter=5000, solver="lbfgs")
    meta.fit(P_train_oof, y_train)
    return meta

def stack_tsa_predict_auc(P_val, y_val, P_test, epoch=250, pop_size=40, seed=42):
    def obj_func(w):
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s <= 0: return 1e9
        w = w / s
        p = P_val @ w
        if (p.max() - p.min()) < 1e-12:  # dự đoán gần hằng số
            return 1.0
        return 1.0 - roc_auc_score(y_val, p)

    n_models = P_val.shape[1]
    bounds = [FloatVar(lb=0.0, ub=1.0) for _ in range(n_models)]
    problem = {"obj_func": obj_func, "bounds": bounds, "minmax": "min"}
    algo = TSA.OriginalTSA(epoch=epoch, pop_size=pop_size, seed=seed)
    best = algo.solve(problem)

    w = np.maximum(np.asarray(best.solution, float), 0.0)
    w = w / (w.sum() + 1e-12)

    p_val = P_val @ w
    p_test = P_test @ w
    return p_val, p_test, w

def evaluate_version(
    X_train, y_train, X_val, y_val, X_test, y_test,
    n_splits, do_calib, calib_method, calib_cv,
    tsa_epoch, tsa_pop
):
    P_train_oof, P_val, P_test, _, base_names = train_bases_oof_and_refit(
        X_train, y_train, X_val, X_test,
        n_splits=n_splits, calibrate=do_calib, calib_method=calib_method, calib_cv=calib_cv
    )
    # Equal weights
    p_eq_test = equal_weight_probs(P_test)
    # Stack LR
    meta = stack_lr_fit(P_train_oof, y_train)
    p_stack_test = meta.predict_proba(P_test)[:, 1]
    # TSA
    _, p_tsa_test, w_tsa = stack_tsa_predict_auc(
        P_val, y_val, P_test, epoch=tsa_epoch, pop_size=tsa_pop
    )
    results = [
        {"name": "EqualWeights", **metrics_report(y_test, p_eq_test)},
        {"name": "Stack_LR",     **metrics_report(y_test, p_stack_test)},
        {"name": "Stack_TSA",    **metrics_report(y_test, p_tsa_test)},
    ]
    return results, {"w_tsa": w_tsa, "P_train_oof": P_train_oof, "P_test": P_test, "base_names": base_names, "meta": meta}

# =========================
# Main UI
# =========================

st.title("❤️ Heart Disease Prediction Model using Ensemble Deep Learning with Optimized Weight")
st.caption("Stacking (LR) & TSA-weighted ensemble với các biến thể dữ liệu: Original / FE / DT / FE+DT")

with st.expander("ℹ️ Hướng dẫn / Notes", expanded=False):
    st.markdown(
        "- Tải **cleveland.csv** (14 cột theo thứ tự chuẩn, cột cuối là `target`).\n"
        "- Chọn biến thể dữ liệu ở sidebar → nhấn **Chạy**.\n"
        "- Bật **Calibrate** nếu muốn hiệu chỉnh xác suất các base models trước khi stack.\n"
        "- XAI: |coef| cho Stacking-LR, SHAP (beeswarm + waterfall), trọng số toàn cục TSA và đóng góp cục bộ theo mẫu.\n"
        "- Bạn có thể tải về các **splits** để tái sử dụng.\n"
    )

# Load Data
with st.spinner("Đang nạp dữ liệu…"):
    try:
        raw = load_data(uploaded, csv_url, csv_path)
    except Exception as e:
        st.error(f"Không đọc được CSV. Kiểm tra lại. Chi tiết: {e}")
        st.stop()

st.success(f"Đã nạp dữ liệu: shape = {raw.shape}")
st.write(raw.head())

# Build variants
with st.spinner("Đang tiền xử lý & tạo biến thể…"):
    built = preprocess_split_variants(raw, K_features=K_features)

# Optionally show MI
if show_mi:
    mi_series = built["mi_series"]
    N = min(20, len(mi_series))
    topN = mi_series.head(N).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35*N)))
    ax.barh(topN.index, topN.values)
    ax.set_title("Top MI scores (Train)")
    ax.set_xlabel("MI score"); ax.set_ylabel("Feature")
    st.pyplot(fig)

# Select variant
X_tr, X_va, X_te, y_tr, y_va, y_te = built["splits"][variant]
st.write(f"### Biến thể: **{variant}**")
st.write("Train:", X_tr.shape, "Val:", X_va.shape, "Test:", X_te.shape)
st.write("Target balance (Train):")
st.write(y_tr.value_counts())

# Train & Evaluate
if run_btn:
    with st.spinner("Đang huấn luyện ensemble & đánh giá…"):
        results, extras = evaluate_version(
            X_tr, y_tr, X_va, y_va, X_te, y_te,
            n_splits=n_splits, do_calib=do_calib, calib_method=calib_method,
            calib_cv=calib_cv, tsa_epoch=tsa_epoch, tsa_pop=tsa_pop
        )

    st.subheader("📈 Kết quả (TEST)")
    st.dataframe(pd.DataFrame(results).set_index("name"))

    base_names = extras["base_names"]
    P_train_oof = extras["P_train_oof"]
    P_test = extras["P_test"]
    meta = extras["meta"]
    w_tsa = extras["w_tsa"]

    # ========== XAI for Stacking-LR ==========
    if show_coef or show_shap:
        st.subheader("🧠 XAI – Stacking (Logistic Regression)")
        if show_coef:
            coef = meta.coef_.ravel()
            imp = np.abs(coef)
            imp_df = (pd.DataFrame({"base_model": base_names, "abs_coef": imp, "coef": coef})
                      .sort_values("abs_coef", ascending=False)
                      .reset_index(drop=True))
            st.write("Global importance (|coef|):")
            st.dataframe(imp_df)

            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(imp_df["base_model"], imp_df["abs_coef"])
            ax.set_title("Stacking-LR | |coef| (global importance)")
            st.pyplot(fig)

        if show_shap:
            try:
                st.write("SHAP beeswarm & waterfall trên không gian OOF/TEST (meta-LR)")
                masker = shap.maskers.Independent(P_train_oof)
                explainer = shap.LinearExplainer(
                    model=meta,
                    masker=masker,
                    feature_perturbation="interventional",
                    link=shap.links.logit
                )
                sv = explainer(P_test)
                sv.feature_names = list(base_names)

                # Beeswarm
                fig = plt.figure(figsize=(6,4))
                shap.plots.beeswarm(sv, max_display=min(12, len(base_names)), show=False)
                st.pyplot(fig)

                # Waterfall for one sample
                if sample_idx >= 0 and sample_idx < P_test.shape[0]:
                    fig = plt.figure(figsize=(7,4))
                    shap.plots.waterfall(sv[sample_idx], max_display=12, show=False)
                    st.pyplot(fig)
                else:
                    st.info("Sample index nằm ngoài phạm vi test set.")
            except Exception as e:
                st.warning(f"Không vẽ được SHAP: {e}")

    # ========== XAI for TSA ==========
    if show_tsa:
        st.subheader("🧠 XAI – TSA Weights")
        w = np.asarray(w_tsa, dtype=float).ravel()
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            w = w / (w.sum() + 1e-12)
        df_w = (pd.DataFrame({"base_model": base_names, "weight": w})
                .sort_values("weight", ascending=False).reset_index(drop=True))
        st.write("Trọng số toàn cục (TSA):")
        st.dataframe(df_w)

        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(df_w["base_model"], df_w["weight"])
        ax.set_title("TSA global weight distribution")
        st.pyplot(fig)

        # Local contributions cho 1 sample
        if sample_idx >= 0 and sample_idx < P_test.shape[0]:
            p_row = P_test[int(sample_idx), :]
            contrib_raw = w * p_row
            p_ens = float(contrib_raw.sum())

            p_base_mean = P_test.mean(axis=0)
            baseline = float((w * p_base_mean).sum())
            contrib = contrib_raw - (w * p_base_mean)

            df_local = (pd.DataFrame({"base_model": base_names, "contribution": contrib})
                        .sort_values("contribution", ascending=True).reset_index(drop=True))
            st.write(f"Local contributions (sample {sample_idx}) – p_ens={p_ens:.3f}, baseline={baseline:.3f}")
            st.dataframe(df_local)

            fig, ax = plt.subplots(figsize=(7, max(3, 0.4*len(base_names))))
            ax.axvline(x=0.0, ls="--", lw=1)
            ax.barh(df_local["base_model"], df_local["contribution"])
            ax.set_title("Local TSA contributions")
            st.pyplot(fig)
        else:
            st.info("Sample index nằm ngoài phạm vi test set.")

    # ========== Download splits ==========
    st.subheader("📥 Tải về Splits")
    # Tạo lại CSVs theo variant đang dùng
    out = Path("splits"); out.mkdir(exist_ok=True, parents=True)
    pd.concat([X_tr, y_tr.rename(TARGET)], axis=1).to_csv(out / f'{variant.lower().replace(" ","_")}_train.csv', index=False)
    pd.concat([X_va, y_va.rename(TARGET)], axis=1).to_csv(out / f'{variant.lower().replace(" ","_")}_val.csv', index=False)
    pd.concat([X_te, y_te.rename(TARGET)], axis=1).to_csv(out / f'{variant.lower().replace(" ","_")}_test.csv', index=False)

    zip_buf = io.BytesIO()
    import zipfile
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in out.glob(f"{variant.lower().replace(' ','_')}_*.csv"):
            zf.writestr(fn.name, fn.read_bytes())
    st.download_button("Tải bộ splits (zip)", data=zip_buf.getvalue(),
                       file_name=f"{variant.lower().replace(' ','_')}_splits.zip", mime="application/zip")

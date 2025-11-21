# ml/train_risk_models_prob.py
# 확률 라벨링(로지스틱 링크) + 학습(LogReg vs RF) + 캘리브레이션 + 저장

import warnings
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------- 0) helpers ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------- 1) Synthetic data with probabilistic labels ----------
def generate_synthetic_data(
    n_samples: int = 100_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    상관 구조를 가진 PHQ/GAD/K10을 만들고,
    라벨은 logit=... -> p=sigmoid(logit) -> y~Bernoulli(p) 방식으로 확률적으로 생성.
    """

    # 상관 구조
    mean = np.zeros(3)
    cov = np.array([
        [1.0, 0.6, 0.5],
        [0.6, 1.0, 0.5],
        [0.5, 0.5, 1.0]
    ])
    Z = np.random.multivariate_normal(mean, cov, size=n_samples)

    # 0~1 스케일로 정규화
    U = (Z - Z.min(axis=0)) / (Z.max(axis=0) - Z.min(axis=0) + 1e-9)

    phq_total = np.round(U[:, 0] * 27).astype(int)      # 0~27
    gad_total = np.round(U[:, 1] * 21).astype(int)      # 0~21
    k10_total = np.round(10 + U[:, 2] * 40).astype(int) # 10~50

    # 약간의 잡음
    phq_total = np.clip(phq_total + np.random.randint(-1, 2, size=n_samples), 0, 27)
    gad_total = np.clip(gad_total + np.random.randint(-1, 2, size=n_samples), 0, 21)
    k10_total = np.clip(k10_total + np.random.randint(-2, 3, size=n_samples), 10, 50)

    # item9/ASQ 생성 (PHQ와 느슨히 연동)
    phq_item9 = np.clip((phq_total // 7) + np.random.randint(-1, 2, size=n_samples), 0, 3)
    base_prob = 0.03 + 0.03*(phq_total/27) + 0.20*(phq_item9/3)
    base_prob = np.clip(base_prob + np.random.normal(0, 0.02, size=n_samples), 0, 0.85)
    asq_any_yes = (np.random.rand(n_samples) < base_prob).astype(int)

    # ---- 확률 라벨링 (로지스틱 링크) ----
    suic_logit = (
        -3.2
        + 0.12 * phq_total
        + 0.85 * phq_item9
        + 1.10 * asq_any_yes
        + np.random.normal(0, 0.25, size=n_samples)
    )
    suicidal = (np.random.rand(n_samples) < sigmoid(suic_logit)).astype(int)

    depr_logit = (
        -2.6
        + 0.11 * phq_total
        + 0.10 * gad_total
        + np.random.normal(0, 0.25, size=n_samples)
    )
    depression = (np.random.rand(n_samples) < sigmoid(depr_logit)).astype(int)

    stress_logit = (
        -3.0
        + 0.09 * k10_total
        + 0.05 * gad_total
        + np.random.normal(0, 0.25, size=n_samples)
    )
    stress = (np.random.rand(n_samples) < sigmoid(stress_logit)).astype(int)

    # 피처: [PHQ, GAD, K10, item9, ASQ]
    X = np.column_stack([phq_total, gad_total, k10_total, phq_item9, asq_any_yes])

    print(
        "[prevalence] suicidal={:.2f}%  depression={:.2f}%  stress={:.2f}%".format(
            100 * suicidal.mean(),
            100 * depression.mean(),
            100 * stress.mean(),
        )
    )
    return X, suicidal, depression, stress


# ---------- 2) Preprocessor ----------
def make_preprocessor() -> ColumnTransformer:
    # 연속형(0~3): 표준화, 이진형(4): 그대로
    return ColumnTransformer([
        ("num_std", StandardScaler(), [0, 1, 2, 3]),
        ("bin", "passthrough", [4]),
    ])


# ---------- 3) Train single label ----------
def train_one_label(
    label_name: str,
    X: np.ndarray,
    y: np.ndarray,
    outdir: Path,
    test_size: float = 0.2
) -> Path:
    pre = make_preprocessor()

    # base estimators (fit은 calibration이 내부에서 수행)
    logreg = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])
    rf = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ))
    ])

    # 라벨별 stratify split
    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(
        idx, test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    # calibration CV (재현성 강화)
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    cal_logreg = CalibratedClassifierCV(estimator=logreg, method="sigmoid", cv=cv3)
    cal_rf     = CalibratedClassifierCV(estimator=rf,     method="sigmoid", cv=cv3)

    cal_logreg.fit(Xtr, ytr)
    cal_rf.fit(Xtr, ytr)

    def eval_model(est):
        p = est.predict_proba(Xte)[:, 1]
        pr = average_precision_score(yte, p)
        try:
            roc = roc_auc_score(yte, p)
        except ValueError:
            roc = float("nan")
        return pr, roc

    lg_pr, lg_roc = eval_model(cal_logreg)
    rf_pr, rf_roc = eval_model(cal_rf)

    print(
        f"[{label_name}] Cal-LogReg PR-AUC={lg_pr:.4f} ROC-AUC={lg_roc:.4f}  |  "
        f"Cal-RF PR-AUC={rf_pr:.4f} ROC-AUC={rf_roc:.4f}"
    )

    # PR-AUC 우선, 근소하면 LogReg 선호
    if (rf_pr - lg_pr) > 0.005 or (np.isfinite(rf_roc) and np.isfinite(lg_roc) and rf_roc > lg_roc + 0.005):
        best = cal_rf
        tag = "cal_rf"
    else:
        best = cal_logreg
        tag = "cal_logreg"

    print(f"[{label_name}] >>> selected={tag}")

    outdir.mkdir(exist_ok=True, parents=True)
    final_path = outdir / f"{label_name}_model.joblib"
    joblib.dump(best, final_path)
    return final_path


# ---------- 4) Orchestrator ----------
def train_and_save_all(n_samples: int = 100_000, outdir: Path = None) -> Dict[str, Path]:
    if outdir is None:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        outdir = PROJECT_ROOT / "models"

    print(f"[train] generate n={n_samples:,}")
    X, y_suic, y_dep, y_str = generate_synthetic_data(n_samples)

    paths = {
        "suicidal":   train_one_label("suicidal",   X, y_suic, outdir),
        "depression": train_one_label("depression", X, y_dep,  outdir),
        "stress":     train_one_label("stress",     X, y_str,  outdir),
    }

    # feature order 저장(프론트/서버 alignment용)
    feat_order = ["phq_total", "gad_total", "k10_total", "phq_item9", "asq_any_yes"]
    joblib.dump(feat_order, outdir / "feature_order.joblib")

    print("\n[train] saved:")
    for k, v in paths.items():
        print(f" - {k}: {v}")

    return paths


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_and_save_all(n_samples=100_000)

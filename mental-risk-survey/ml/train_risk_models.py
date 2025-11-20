# ml/train_risk_models_prob.py
# 확률 라벨링(로지스틱 링크) + 간단 학습(LogReg vs RF, 필요시 캘리브레이션)
import warnings
from pathlib import Path
from typing import Tuple
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
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
def generate_synthetic_data(n_samples: int = 100_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    상관 구조를 가진 PHQ/GAD/K10을 만들고,
    라벨은 logit=... -> p=sigmoid(logit) -> y~Bernoulli(p) 방식으로 확률적으로 생성.
    """
    # 상관 구조
    mean = np.zeros(3)
    cov = np.array([[1.0, 0.6, 0.5],
                    [0.6, 1.0, 0.5],
                    [0.5, 0.5, 1.0]])
    Z = np.random.multivariate_normal(mean, cov, size=n_samples)
    U = (Z - Z.min(axis=0)) / (Z.max(axis=0) - Z.min(axis=0) + 1e-9)

    phq_total = np.round(U[:,0] * 27).astype(int)             # 0~27
    gad_total = np.round(U[:,1] * 21).astype(int)             # 0~21
    k10_total = np.round(10 + U[:,2] * 40).astype(int)        # 10~50

    # 약간의 잡음
    phq_total = np.clip(phq_total + np.random.randint(-1,2,size=n_samples), 0,27)
    gad_total = np.clip(gad_total + np.random.randint(-1,2,size=n_samples), 0,21)
    k10_total = np.clip(k10_total + np.random.randint(-2,3,size=n_samples), 10,50)

    # item9/ASQ 생성 (PHQ와 느슨히 연동)
    phq_item9 = np.clip((phq_total // 7) + np.random.randint(-1,2,size=n_samples), 0,3)
    base_prob = 0.03 + 0.03*(phq_total/27) + 0.20*(phq_item9/3)
    base_prob = np.clip(base_prob + np.random.normal(0,0.02,size=n_samples), 0, 0.85)
    asq_any_yes = (np.random.rand(n_samples) < base_prob).astype(int)

    # ---- 확률 라벨링(로지스틱 링크) ----
    # 적당한 유병률이 나오도록 절편/계수를 설정 (원하면 쉽게 조정 가능)
    # 1) 자살 신호: PHQ, item9, ASQ 영향이 큼
    suic_logit = (
        -3.2
        + 0.12*phq_total
        + 0.85*phq_item9
        + 1.10*asq_any_yes
        + np.random.normal(0, 0.25, size=n_samples)
    )
    suic_p = sigmoid(suic_logit)
    suicidal = (np.random.rand(n_samples) < suic_p).astype(int)

    # 2) 우울 위험: PHQ, GAD 중심(요청대로 컷오프 대신 연속 영향)
    depr_logit = (
        -2.6
        + 0.11*phq_total
        + 0.10*gad_total
        + np.random.normal(0, 0.25, size=n_samples)
    )
    # (원하면 GAD 민감도를 올리고 싶을 때: + 0.12*gad_total 처럼 계수↑)
    depr_p = sigmoid(depr_logit)
    depression = (np.random.rand(n_samples) < depr_p).astype(int)

    # 3) 스트레스 위험: K10, GAD 영향
    stress_logit = (
        -3.0
        + 0.09*k10_total
        + 0.05*gad_total
        + np.random.normal(0, 0.25, size=n_samples)
    )
    stress_p = sigmoid(stress_logit)
    stress = (np.random.rand(n_samples) < stress_p).astype(int)

    # 피처: [PHQ, GAD, K10, item9, ASQ]
    X = np.column_stack([phq_total, gad_total, k10_total, phq_item9, asq_any_yes])

    # 유병률 프린트(디버그)
    print("[prevalence] suicidal={:.2f}%  depression={:.2f}%  stress={:.2f}%".format(
        100*suicidal.mean(), 100*depression.mean(), 100*stress.mean()
    ))
    return X, suicidal, depression, stress

# ---------- 2) Preprocessor ----------
def make_preprocessor() -> ColumnTransformer:
    # 연속(0~3): 표준화, 이진(4): 그대로
    return ColumnTransformer([
        ("num_std", StandardScaler(), [0,1,2,3]),
        ("bin", "passthrough", [4]),
    ])

# ---------- 3) Train single label (LogReg vs RF) with calibration ----------
def train_one_label(label_name: str, Xtr, ytr, Xte, yte, outdir: Path) -> Path:
    pre = make_preprocessor()

    # 베이스 모델
    logreg = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=1000, solver="liblinear",
            class_weight=None, random_state=RANDOM_STATE
        ))
    ])
    rf = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])

    logreg.fit(Xtr, ytr)
    rf.fit(Xtr, ytr)

    # 확률 보정(Platt scaling)
    cal_logreg = CalibratedClassifierCV(estimator=logreg, method="sigmoid", cv=3)
    cal_logreg.fit(Xtr, ytr)

    cal_rf = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3)
    cal_rf.fit(Xtr, ytr)

    def eval_model(est):
        p = est.predict_proba(Xte)[:,1]
        return (average_precision_score(yte, p), roc_auc_score(yte, p))

    lg_pr, lg_roc = eval_model(cal_logreg)
    rf_pr, rf_roc = eval_model(cal_rf)
    print(f"[{label_name}] Cal-LogReg PR-AUC={lg_pr:.4f} ROC-AUC={lg_roc:.4f}  |  Cal-RF PR-AUC={rf_pr:.4f} ROC-AUC={rf_roc:.4f}")

    # 성능 근소시(예: PR-AUC 차이 ≤ 0.005)에는 LogReg 선호 → 확률 안정적
    if (rf_pr - lg_pr) > 0.005 or ((rf_pr - lg_pr) >= -1e-9 and rf_roc > lg_roc + 0.005):
        best = cal_rf
        tag  = "cal_rf"
    else:
        best = cal_logreg
        tag  = "cal_logreg"

    print(f"[{label_name}] >>> selected={tag}")

    outdir.mkdir(exist_ok=True, parents=True)
    final_path = outdir / f"{label_name}_model.joblib"
    joblib.dump(best, final_path)
    return final_path

# ---------- 4) Orchestrator ----------
def train_and_save_all(n_samples: int = 100_000, outdir: Path = None):
    if outdir is None:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        outdir = PROJECT_ROOT / "models"

    print(f"[train] generate n={n_samples:,}")
    X, y_suic, y_dep, y_str = generate_synthetic_data(n_samples)

    # 하나의 split을 모든 라벨에 공통 적용
    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y_suic)
    Xtr, Xte = X[tr_idx], X[te_idx]
    ysu_tr, ysu_te = y_suic[tr_idx], y_suic[te_idx]
    yde_tr, yde_te = y_dep[tr_idx],  y_dep[te_idx]
    yst_tr, yst_te = y_str[tr_idx],  y_str[te_idx]

    paths = {}
    paths["suicidal"]   = train_one_label("suicidal",   Xtr, ysu_tr, Xte, ysu_te, outdir)
    paths["depression"] = train_one_label("depression", Xtr, yde_tr, Xte, yde_te, outdir)
    paths["stress"]     = train_one_label("stress",     Xtr, yst_tr, Xte, yst_te, outdir)

    print("\n[train] saved:")
    for k, v in paths.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_and_save_all(n_samples=100_000)  # 필요시 200_000으로 올려도 OK

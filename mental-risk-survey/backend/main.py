from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
import traceback
import sys

app = FastAPI(
    title="Mental Risk Survey API (ML ver.)",
    description="청년 정신건강 설문 ML 데모 API",
    version="0.2.0",
)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- 경로/모델 경로 ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
SUICIDAL_MODEL_PATH = MODELS_DIR / "suicidal_model.joblib"
DEPRESSION_MODEL_PATH = MODELS_DIR / "depression_model.joblib"
STRESS_MODEL_PATH = MODELS_DIR / "stress_model.joblib"

# uvicorn --reload가 동작할 때도 패키지 임포트 경로가 꼬이지 않도록
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------- 입력/출력 스키마 ----------------
class RiskInput(BaseModel):
    phq_total: int
    gad_total: int
    k10_total: int
    phq_item9: int
    asq_any_yes: bool


class RiskOutput(BaseModel):
    suicidal_signal_pct: float
    depression_risk_pct: float
    stress_risk_pct: float


# ---------------- 모델 로딩 ----------------
suicidal_model = None
depression_model = None
stress_model = None


def load_models():
    global suicidal_model, depression_model, stress_model
    suicidal_model = joblib.load(SUICIDAL_MODEL_PATH)
    depression_model = joblib.load(DEPRESSION_MODEL_PATH)
    stress_model = joblib.load(STRESS_MODEL_PATH)
    print("[models] loaded all.")


@app.on_event("startup")
def on_startup():
    # 모델 로드
    print(f"[startup] PROJECT_ROOT={PROJECT_ROOT}")
    print(f"[startup] MODELS_DIR={MODELS_DIR}")
    try:
        load_models()
    except Exception as e:
        print(f"[models] 로드 실패: {e}")
        traceback.print_exc()


# ---------------- 유틸 ----------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def predict_proba_01(model, features):
    """predict_proba([[...]] )의 양성 클래스 확률(0~1) 반환"""
    proba = model.predict_proba(features)[0, 1]
    return float(clamp01(float(proba)))


def soften(p: float, eps: float = 0.005) -> float:
    """0~1 확률을 [eps, 1-eps]로 살짝 완충 (표시용)"""
    return max(eps, min(1.0 - eps, p))


# ---------------- API ----------------
@app.post("/predict_risk", response_model=RiskOutput)
def predict_risk(payload: RiskInput):
    if suicidal_model is None or depression_model is None or stress_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # 학습 시와 동일한 피처 순서
    X = np.array(
        [
            [
                payload.phq_total,
                payload.gad_total,
                payload.k10_total,
                payload.phq_item9,
                1 if payload.asq_any_yes else 0,
            ]
        ]
    )

    suicidal_p = predict_proba_01(suicidal_model, X)
    depression_p = predict_proba_01(depression_model, X)
    stress_p = predict_proba_01(stress_model, X)

    return RiskOutput(
        suicidal_signal_pct=suicidal_p * 100.0,
        depression_risk_pct=depression_p * 100.0,
        stress_risk_pct=stress_p * 100.0,
    )


@app.get("/")
def root():
    return {"message": "Mental Risk Survey ML API is running"}


@app.get("/model_info")
def model_info():
    return {
        "project_root": str(PROJECT_ROOT),
        "models_dir": str(MODELS_DIR),
        "files": {
            "suicidal": str(SUICIDAL_MODEL_PATH),
            "depression": str(DEPRESSION_MODEL_PATH),
            "stress": str(STRESS_MODEL_PATH),
        },
        "exists": {
            "suicidal": SUICIDAL_MODEL_PATH.exists(),
            "depression": DEPRESSION_MODEL_PATH.exists(),
            "stress": STRESS_MODEL_PATH.exists(),
        },
    }

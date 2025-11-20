// js/survey_app.js
import {
  API_BASE,
  PHQ9A_ITEMS,
  GAD7_ITEMS,
  K10_ITEMS,
  ASQ_ITEMS,
  PHQ_GAD_CHOICES,
  K10_CHOICES,
} from "./survey_config.js";

import {
  sum,
  bandPHQ9,
  bandGAD7,
  bandK10,
  phqBandLabel,
  gadBandLabel,
  k10BandLabel,
  computeOverallTier,
} from "./survey_scoring.js";

/** ======== 상태 ======== */
// 지역 리스트 & 선택값
const REGIONS = [
  "서울특별시",
  "부산광역시",
  "대구광역시",
  "인천광역시",
  "광주광역시",
  "대전광역시",
  "울산광역시",
  "세종광역시",
  "경기도",
  "강원도",
  "충청북도",
  "충청남도",
  "전라북도",
  "전라남도",
  "경상북도",
  "경상남도",
  "제주특별자치도",
];
let region = null;

let step = 0; // 0~5 (지역이 0단계로 추가)
const totalSteps = 6; // 기존 5 -> 6

const phq9a = Array(9).fill(null);
const gad7 = Array(7).fill(null);
const k10 = Array(10).fill(null);
const asq = Array(4).fill(false);

let riskScores = null; // { suicidalPct, depressionPct, stressPct }
let riskLoading = false;
let riskError = null;

/** ======== DOM ======== */
const stepContainer = document.getElementById("step-container");
const progressBar = document.getElementById("progress-bar");
const stepLabel = document.getElementById("step-label");
const prevButton = document.getElementById("prev-button");
const nextButton = document.getElementById("next-button");

const crisisModal = document.getElementById("crisis-modal");
const crisisLaterBtn = document.getElementById("crisis-later-btn");
const crisisConnectBtn = document.getElementById("crisis-connect-btn");

/** ======== 렌더 ======== */
function render() {
  const pct = ((step + 1) / totalSteps) * 100;
  progressBar.style.width = pct + "%";
  stepLabel.textContent = `단계 ${step + 1}/${totalSteps}`;

  prevButton.disabled = step === 0;
  nextButton.textContent = step === totalSteps - 1 ? "제출" : "다음";

  stepContainer.innerHTML = "";

  if (step === 0) {
    renderRegionStep(); // 지역 선택
  } else if (step === 1) {
    renderIntro(); // 소개 단계
  } else if (step === 2) {
    renderScale(
      "PHQ-9A (지난 2주)",
      "각 문항에 대해 지난 2주 동안의 빈도를 선택하세요.",
      PHQ9A_ITEMS,
      phq9a,
      PHQ_GAD_CHOICES
    );
  } else if (step === 3) {
    renderScale(
      "GAD-7 (지난 2주)",
      "지난 2주의 평균적인 느낌을 선택하세요.",
      GAD7_ITEMS,
      gad7,
      PHQ_GAD_CHOICES
    );
  } else if (step === 4) {
    renderScale(
      "K10 (지난 4주)",
      "지난 4주 동안 얼마나 자주 그랬는지 선택하세요.",
      K10_ITEMS,
      k10,
      K10_CHOICES
    );
  } else if (step === 5) {
    renderASQAndSummary();
  }
}

/** 0단계: 지역 선택 */
function renderRegionStep() {
  const title = document.createElement("h2");
  title.className = "text-xl font-semibold";
  title.textContent = "지역 선택";

  const subtitle = document.createElement("p");
  subtitle.className = "text-sm text-slate-600 mb-4";
  subtitle.textContent = "거주 지역을 하나 선택해주세요.";

  const label = document.createElement("label");
  label.className = "block text-sm font-medium text-slate-700";
  label.textContent = "지역";

  const select = document.createElement("select");
  select.className =
    "mt-1 block w-full border border-slate-300 rounded-md px-3 py-2";

  const ph = document.createElement("option");
  ph.value = "";
  ph.textContent = "지역을 선택하세요";
  ph.disabled = true;
  ph.selected = region == null;
  select.appendChild(ph);

  REGIONS.forEach((r) => {
    const opt = document.createElement("option");
    opt.value = r;
    opt.textContent = r;
    if (region === r) opt.selected = true;
    select.appendChild(opt);
  });

  select.addEventListener("change", () => {
    region = select.value || null;
    render();
  });

  stepContainer.appendChild(title);
  stepContainer.appendChild(subtitle);
  const wrap = document.createElement("div");
  wrap.className = "space-y-2";
  wrap.appendChild(label);
  wrap.appendChild(select);
  stepContainer.appendChild(wrap);

  const note = document.createElement("p");
  note.className = "text-xs text-slate-500";
  note.textContent = "※ 선택한 지역은 요약과 지도 분석에 사용됩니다.";
  stepContainer.appendChild(note);
}

/** 소개 */
function renderIntro() {
  const title = document.createElement("h2");
  title.className = "text-xl font-semibold";
  title.textContent = "시작하기 전에";

  const desc = document.createElement("p");
  desc.className = "text-sm text-slate-600";
  desc.textContent =
    "본 설문은 선별(screening) 목적이며 의학적 진단이 아닙니다. 응답은 비밀로 취급되며, 고위험 응답 시 즉시 도움 연결 절차가 진행됩니다.";

  const alertBox = document.createElement("div");
  alertBox.className =
    "border border-amber-300 bg-amber-50 text-amber-900 px-4 py-3 rounded-md text-sm mt-4";
  alertBox.innerHTML =
    "<b>안전 안내</b><br/>자해·자살 관련 응답이 감지되면 별도 도움 안내 창이 자동으로 표시됩니다.";

  stepContainer.appendChild(title);
  stepContainer.appendChild(desc);
  stepContainer.appendChild(alertBox);
}

/** 공통 척도 렌더 */
function renderScale(titleText, subtitleText, items, answersArray, choices) {
  const title = document.createElement("h2");
  title.className = "text-xl font-semibold";
  title.textContent = titleText;

  const subtitle = document.createElement("p");
  subtitle.className = "text-sm text-slate-600 mb-4";
  subtitle.textContent = subtitleText;

  stepContainer.appendChild(title);
  stepContainer.appendChild(subtitle);

  items.forEach((q, index) => {
    const wrapper = document.createElement("div");
    wrapper.className = "space-y-2 mb-3";

    const label = document.createElement("div");
    label.className = "font-semibold text-sm";
    label.textContent = `${index + 1}. ${q}`;

    const choiceRow = document.createElement("div");
    choiceRow.className = "grid grid-cols-2 sm:grid-cols-4 gap-2";

    choices.forEach((c) => {
      const btn = document.createElement("button");
      btn.type = "button";
      const active = answersArray[index] === c.value;
      btn.className =
        "border rounded-md text-sm px-3 py-2 text-left " +
        (active
          ? "bg-blue-600 text-white border-blue-600"
          : "bg-white text-slate-700 border-slate-300 hover:bg-slate-50");
      btn.textContent = c.label;

      btn.addEventListener("click", () => {
        answersArray[index] = c.value;
        render();
      });

      choiceRow.appendChild(btn);
    });

    wrapper.appendChild(label);
    wrapper.appendChild(choiceRow);
    stepContainer.appendChild(wrapper);
  });

  const phqTotal = sum(phq9a);
  const gadTotal = sum(gad7);
  const k10Total = sum(k10);

  const totalBox = document.createElement("div");
  totalBox.className = "mt-4 text-sm text-slate-600 border-t pt-3";

  if (titleText.startsWith("PHQ-9A")) {
    const band = bandPHQ9(phqTotal);
    totalBox.textContent = `현재 점수: ${phqTotal} / 27 — ${phqBandLabel(
      band
    )}`;
  } else if (titleText.startsWith("GAD-7")) {
    const band = bandGAD7(gadTotal);
    totalBox.textContent = `현재 점수: ${gadTotal} / 21 — ${gadBandLabel(
      band
    )}`;
  } else if (titleText.startsWith("K10")) {
    const band = bandK10(k10Total);
    totalBox.textContent = `현재 점수: ${k10Total} / 50 — ${k10BandLabel(
      band
    )}`;
  }

  stepContainer.appendChild(totalBox);
}

/** ASQ + 요약 + ML 스코어 */
function renderASQAndSummary() {
  const title = document.createElement("h2");
  title.className = "text-xl font-semibold";
  title.textContent = "ASQ (빠른 안전 확인)";

  const subtitle = document.createElement("p");
  subtitle.className = "text-sm text-slate-600 mb-4";
  subtitle.textContent =
    "예/아니오로 답해주세요. 하나라도 ‘예’이면 즉시 도움 안내가 표시됩니다.";

  stepContainer.appendChild(title);
  stepContainer.appendChild(subtitle);

  ASQ_ITEMS.forEach((q, idx) => {
    const wrapper = document.createElement("div");
    wrapper.className = "space-y-2 mb-3";

    const label = document.createElement("div");
    label.className = "font-semibold text-sm";
    label.textContent = `${idx + 1}. ${q}`;

    const btnRow = document.createElement("div");
    btnRow.className = "flex gap-3";

    const yesBtn = document.createElement("button");
    yesBtn.type = "button";
    yesBtn.className =
      "px-3 py-1.5 rounded-md text-sm " +
      (asq[idx]
        ? "bg-blue-600 text-white"
        : "border border-slate-300 bg-white text-slate-700");
    yesBtn.textContent = "예";
    yesBtn.addEventListener("click", () => {
      asq[idx] = true;
      openCrisisModal();
      render();
    });

    const noBtn = document.createElement("button");
    noBtn.type = "button";
    noBtn.className =
      "px-3 py-1.5 rounded-md text-sm " +
      (!asq[idx]
        ? "bg-blue-600 text-white"
        : "border border-slate-300 bg-white text-slate-700");
    noBtn.textContent = "아니오";
    noBtn.addEventListener("click", () => {
      asq[idx] = false;
      render();
    });

    btnRow.appendChild(yesBtn);
    btnRow.appendChild(noBtn);

    wrapper.appendChild(label);
    wrapper.appendChild(btnRow);
    stepContainer.appendChild(wrapper);
  });

  const phqTotal = sum(phq9a);
  const gadTotal = sum(gad7);
  const k10Total = sum(k10);

  const phqBand = bandPHQ9(phqTotal);
  const gadBand = bandGAD7(gadTotal);
  const k10Band = bandK10(k10Total);

  const item9Flag = (phq9a[8] ?? 0) >= 1;
  const asqAnyYes = asq.some(Boolean);

  const overallTier = computeOverallTier(
    phqTotal,
    phqBand,
    gadBand,
    k10Band,
    item9Flag,
    asqAnyYes
  );

  const summaryBox = document.createElement("div");
  summaryBox.className =
    "mt-4 p-3 border rounded-md bg-slate-50 text-sm space-y-1";

  summaryBox.innerHTML = `
    <div class="font-semibold">요약</div>
    <div>지역: ${region ?? "-"}</div>
    <div>PHQ-9A: ${phqTotal}/27 — ${phqBandLabel(phqBand)}</div>
    <div>GAD-7: ${gadTotal}/21 — ${gadBandLabel(gadBand)}</div>
    <div>K10: ${k10Total}/50 — ${k10BandLabel(k10Band)}</div>
  `;

  const overallLine = document.createElement("div");
  const color =
    overallTier.tier === "high"
      ? "text-red-600"
      : overallTier.tier === "mid"
      ? "text-amber-600"
      : "text-emerald-600";
  overallLine.innerHTML = `종합 선별: <b class="${color}">${overallTier.tier.toUpperCase()}</b>`;
  summaryBox.appendChild(overallLine);

  if (overallTier.rationale.length > 0) {
    const reason = document.createElement("div");
    reason.className = "text-slate-500";
    reason.textContent = `근거: ${overallTier.rationale.join(", ")}`;
    summaryBox.appendChild(reason);
  }

  const note = document.createElement("div");
  note.className = "text-xs text-slate-400 mt-1";
  note.textContent = "※ 본 결과는 선별 목적이며 진단이 아닙니다.";
  summaryBox.appendChild(note);

  stepContainer.appendChild(summaryBox);

  const mlBox = document.createElement("div");
  mlBox.className = "mt-4 p-3 border rounded-md bg-white text-sm space-y-2";

  const mlTitle = document.createElement("div");
  mlTitle.className = "font-semibold";
  mlTitle.textContent = "머신러닝 기반 위험 스코어";
  mlBox.appendChild(mlTitle);

  if (riskLoading) {
    const loading = document.createElement("div");
    loading.className = "text-sm text-slate-500";
    loading.textContent = "위험 스코어 계산 중입니다...";
    mlBox.appendChild(loading);
  }

  if (riskError) {
    const errDiv = document.createElement("div");
    errDiv.className = "text-sm text-red-600";
    errDiv.textContent = riskError;
    mlBox.appendChild(errDiv);
  }

  if (riskScores) {
    const grid = document.createElement("div");
    grid.className = "grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm";

    const makeCard = (title, value) => {
      const card = document.createElement("div");
      card.className = "border rounded-lg p-3";
      const t = document.createElement("div");
      t.className = "font-semibold";
      t.textContent = title;
      const v = document.createElement("div");
      v.className = "text-lg";
      v.textContent = `${value.toFixed(1)}%`;
      card.appendChild(t);
      card.appendChild(v);
      return card;
    };

    grid.appendChild(
      makeCard("자살 관련 위험 신호", riskScores.suicidalPct || 0)
    );
    grid.appendChild(makeCard("우울 위험", riskScores.depressionPct || 0));
    grid.appendChild(makeCard("스트레스 위험", riskScores.stressPct || 0));

    mlBox.appendChild(grid);
  }

  const mlNote = document.createElement("div");
  mlNote.className = "text-xs text-slate-400";
  mlNote.textContent =
    "※ 위 스코어는 머신러닝 모델(또는 위험 스코어링 로직)의 예측값이며, 실제 임상 진단이 아닙니다.";
  mlBox.appendChild(mlNote);

  stepContainer.appendChild(mlBox);

  // ▼▼▼ 지도 버튼 ▼▼▼
  if (riskScores) {
    const mapBtn = document.createElement("button");
    mapBtn.className = "mt-3 px-3 py-2 rounded-md bg-emerald-600 text-white";
    mapBtn.textContent = "지도 보기";
    mapBtn.onclick = async () => {
      window.open("./map/지역_위험_지도.html", "_blank");
    };
    mlBox.appendChild(mapBtn);
  }
}

/** ======== 완료 체크 ======== */
function isStepComplete() {
  if (step === 0) return !!region; // 지역 필수
  if (step === 1) return true; // 소개
  if (step === 2) return phq9a.every((v) => v != null);
  if (step === 3) return gad7.every((v) => v != null);
  if (step === 4) return k10.every((v) => v != null);
  if (step === 5) return true;
  return false;
}

/** ======== API 호출 ======== */
async function submitAndFetchRisk() {
  const phqTotal = sum(phq9a);
  const gadTotal = sum(gad7);
  const k10Total = sum(k10);
  const phqItem9 = phq9a[8] ?? 0;
  const asqAnyYes = asq.some(Boolean);

  riskLoading = true;
  riskError = null;
  riskScores = null;
  render();

  try {
    const resp = await fetch(`${API_BASE}/predict_risk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        phq_total: phqTotal,
        gad_total: gadTotal,
        k10_total: k10Total,
        phq_item9: phqItem9,
        asq_any_yes: asqAnyYes,
      }),
    });

    const raw = await resp.text();

    if (!resp.ok) {
      let detail = raw;
      try {
        const j = JSON.parse(raw);
        if (j?.detail)
          detail =
            typeof j.detail === "string"
              ? j.detail
              : JSON.stringify(j.detail);
      } catch (_) {
        /* ignore */
      }

      throw new Error(`API ${resp.status}\n${String(detail).slice(0, 800)}`);
    }

    let data;
    try {
      data = JSON.parse(raw);
    } catch (e) {
      throw new Error(`Invalid JSON from backend:\n${raw.slice(0, 800)}`);
    }

    const s = Number(data?.suicidal_signal_pct);
    const d = Number(data?.depression_risk_pct);
    const t = Number(data?.stress_risk_pct);
    if ([s, d, t].some((v) => Number.isNaN(v))) {
      throw new Error(
        `Missing/NaN fields in response:\n${raw.slice(0, 800)}`
      );
    }

    riskScores = { suicidalPct: s, depressionPct: d, stressPct: t };

    alert(
      "제출 완료: 선별 결과 및 ML 스코어가 계산되었습니다. (본 결과는 진단이 아닙니다)"
    );
  } catch (e) {
    const msg = e && e.message ? e.message : String(e);
    riskError = `위험 스코어 계산 중 오류가 발생했습니다.\n${msg}`;
    alert(
      `위험 스코어 계산 중 오류가 발생했습니다.\n--- 오류 상세 ---\n${msg}`
    );
  } finally {
    riskLoading = false;
    render();
  }
}

/** ======== Crisis modal ======== */
function openCrisisModal() {
  crisisModal.classList.remove("hidden");
  crisisModal.classList.add("flex");
}
function closeCrisisModal() {
  crisisModal.classList.add("hidden");
  crisisModal.classList.remove("flex");
}
crisisLaterBtn.addEventListener("click", closeCrisisModal);
crisisConnectBtn.addEventListener("click", closeCrisisModal);

/** ======== 네비 버튼 ======== */
prevButton.addEventListener("click", () => {
  if (step > 0) {
    step -= 1;
    render();
  }
});

nextButton.addEventListener("click", () => {
  if (step < totalSteps - 1) {
    if (!isStepComplete()) {
      alert("현재 단계의 모든 문항에 응답해주세요.");
      return;
    }
    step += 1;
    render();
  } else {
    submitAndFetchRisk();
  }
});

/** ======== 초기 렌더 ======== */
render();

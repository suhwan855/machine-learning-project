// js/survey-scoring.js

// 공통 합계
export const sum = (arr) =>
  arr.reduce((a, b) => a + (b == null ? 0 : b), 0);

// 밴드 계산
export const bandPHQ9 = (total) => {
  if (total <= 4) return "none";
  if (total <= 9) return "mild";
  if (total <= 14) return "moderate";
  if (total <= 19) return "moderately_severe";
  return "severe";
};

export const bandGAD7 = (total) => {
  if (total <= 4) return "none";
  if (total <= 9) return "mild";
  if (total <= 14) return "moderate";
  return "severe";
};

export const bandK10 = (total) => {
  if (total <= 15) return "low";
  if (total <= 21) return "medium";
  if (total <= 29) return "high";
  return "very_high";
};

// 라벨
export const phqBandLabel = (b) => {
  switch (b) {
    case "none":
      return "정상/최소";
    case "mild":
      return "경도";
    case "moderate":
      return "중등도";
    case "moderately_severe":
      return "중등도-중증";
    case "severe":
      return "중증";
    default:
      return b;
  }
};

export const gadBandLabel = (b) => {
  switch (b) {
    case "none":
      return "정상/최소";
    case "mild":
      return "경도";
    case "moderate":
      return "중등도";
    case "severe":
      return "중증";
    default:
      return b;
  }
};

export const k10BandLabel = (b) => {
  switch (b) {
    case "low":
      return "낮음";
    case "medium":
      return "중간";
    case "high":
      return "높음";
    case "very_high":
      return "매우 높음";
    default:
      return b;
  }
};

// 종합 tier 계산
export function computeOverallTier(
  phqTotal,
  phqBand,
  gadBand,
  k10Band,
  item9Flag,
  asqAnyYes
) {
  const reasons = [];
  let tier = "low";

  if (asqAnyYes || item9Flag || phqTotal >= 20) {
    tier = "high";
    if (asqAnyYes) reasons.push("ASQ에서 위험 응답(예)");
    if (item9Flag) reasons.push("PHQ-9A 9번 응답 > 0");
    if (phqTotal >= 20) reasons.push("PHQ-9A 총점 20 이상");
    return { tier, rationale: reasons };
  }

  const highish =
    phqBand === "moderately_severe" ||
    phqBand === "severe" ||
    gadBand === "severe" ||
    k10Band === "very_high";

  const midish =
    phqBand === "moderate" ||
    gadBand === "moderate" ||
    k10Band === "high";

  if (highish) {
    tier = "high";
    if (phqBand === "moderately_severe" || phqBand === "severe")
      reasons.push(`PHQ-9A ${phqBand.replace("_", " ")}`);
    if (gadBand === "severe") reasons.push("GAD-7 severe");
    if (k10Band === "very_high") reasons.push("K10 very high");
  } else if (midish) {
    tier = "mid";
    if (phqBand === "moderate") reasons.push("PHQ-9A moderate");
    if (gadBand === "moderate") reasons.push("GAD-7 moderate");
    if (k10Band === "high") reasons.push("K10 high");
  }

  return { tier, rationale: reasons };
}

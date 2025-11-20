import folium
import pandas as pd
import json
import requests
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# ============================
# 1) 데이터 로드
# ============================

BASE_DIR = Path(__file__).resolve().parent  # danger_map.py가 있는 map 폴더
file_path = BASE_DIR / "mental_socioeconomic_dataset.xlsx"
try:
    mental = pd.read_excel(file_path, sheet_name="mental_health_status")
    socio = pd.read_excel(file_path, sheet_name="socioeconomic_status")
except FileNotFoundError:
    print("⚠ 파일이 없어서 더미데이터 사용합니다.")
    data = {
        'year': [2018, 2018, 2019, 2019, 2020, 2020],
        'region': ['서울', '부산', '서울', '부산', '서울', '부산'],
        'youth_suicide_rate': [15.1, 12.5, 16.0, 13.0, 14.5, 12.0],
        'depression_rate': [25.0, 20.0, 26.0, 21.0, 24.0, 19.0],
        'stress_rate': [35.0, 30.0, 36.0, 31.0, 34.0, 29.0],
        'youth_unemployment_rate': [9.0, 7.5, 9.5, 8.0, 8.5, 7.0]
    }
    mental = pd.DataFrame(data)
    socio = pd.DataFrame(data)

df = pd.merge(mental, socio, on=["year", "region"])

# ============================
# 2) 지표 목록
# ============================
metrics = {
    "자살률": "youth_suicide_rate",
    "우울감 경험률": "depression_rate",
    "스트레스 인지율": "stress_rate",
    "청년 실업률": "youth_unemployment_rate",
}

metric_keys = list(metrics.keys())

# ============================
# 3) 과거 데이터 JSON 생성
# ============================
base_year_list = sorted(df["year"].unique())
data_dict = {}

for year in base_year_list:
    year_df = df[df["year"] == year]
    metric_dict = {}
    for label, col in metrics.items():
        vals = year_df[["region", col]].dropna()
        metric_dict[label] = dict(zip(vals["region"], vals[col]))
    data_dict[str(year)] = metric_dict

# ============================
# 4) 예측값 생성 (다음 해)
# ============================
pred_year = max(base_year_list) + 1
regions = df["region"].unique()

# 예측 결과를 data_dict에 바로 추가
data_dict[str(pred_year)] = {label: {} for label in metrics.keys()}

for region in regions:
    region_data = df[df["region"] == region]
    for label, col in metrics.items():
        X = region_data[["year"]].values
        y = region_data[col].values

        if len(y) >= 2:  # 최소 2년 이상 있어야 예측 가능
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X, y)
            pred = float(model.predict([[pred_year]])[0])
        else:
            pred = None

        data_dict[str(pred_year)][label][region] = pred

# 최종 연도 목록 (과거 + 예측)
year_list_all = sorted(int(y) for y in data_dict.keys())

# JSON 직렬화
data_json = json.dumps(data_dict, ensure_ascii=False)

initial_year = year_list_all[0]
initial_metric = metric_keys[0]

# ============================
# 5) GeoJSON
# ============================
geo_url = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_provinces_geo.json"
geo_data = requests.get(geo_url).json()

# ============================
# 6) 지도 생성
# ============================
m = folium.Map(location=[36.5, 127.8], zoom_start=7)

geo_layer = folium.GeoJson(
    geo_data,
    name="region_layer",
    style_function=lambda f: {
        "fillColor": "white",
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.8,
    },
)
geo_layer.add_to(m)

# ============================
# 7) JS 코드 (5단계 위험도 + 예측연도 + 차트)
# ============================

# 연도 콤보박스 옵션 HTML (예측연도는 표시만 다르게)
year_options_html = ""
for y in year_list_all:
    if y == pred_year:
        year_options_html += f'<option value="{y}">{y} (예측)</option>'
    else:
        year_options_html += f'<option value="{y}">{y}</option>'

metric_options_html = ''.join([f'<option value="{k}">{k}</option>' for k in metric_keys])

custom_js = f"""
<script>
var regionData = {data_json};
var predYear = "{pred_year}";

/////////////////////////////////////////////////////////////////////
// 5단계 색상 기준 (고정 위험도 구간)
/////////////////////////////////////////////////////////////////////
function getColor(value) {{
    if (value === null || value === undefined || isNaN(value)) {{
        return "#CCCCCC"; // 데이터 없음
    }}
    if (value <= 10) return "#6CC24A";      // 매우 낮음
    if (value <= 15) return "#A8E05F";      // 낮음
    if (value <= 20) return "#FFD700";      // 중간
    if (value <= 25) return "#FF8C42";      // 높음
    return "#FF5733";                       // 매우 높음
}}

/////////////////////////////////////////////////////////////////////
// 지도 업데이트
/////////////////////////////////////////////////////////////////////
function updateMap() {{
    var mapObj = null;
    for (var k in window) {{
        if (window[k] instanceof L.Map) {{
            mapObj = window[k];
            break;
        }}
    }}
    if (!mapObj) return;

    var year = document.getElementById("yearSelect").value;
    var metric = document.getElementById("metricSelect").value;

    if (!regionData[year] || !regionData[year][metric]) {{
        console.log("선택된 연도/지표 데이터 없음");
        return;
    }}

    var selectedData = regionData[year][metric];

    // 컬러바 제목 업데이트
    var legendTitleElement = document.getElementById("colorBarTitle");
    if (legendTitleElement) {{
        var titleText = metric + " (" + year;
        if (year === predYear) {{
            titleText += "년, 예측값";
        }} else {{
            titleText += "년";
        }}
        titleText += ")";
        legendTitleElement.innerText = titleText;
    }}

    mapObj.eachLayer(function(layer) {{
        if (layer.feature && layer.feature.properties && layer.setStyle) {{
            let name = layer.feature.properties.name;
            let val = selectedData[name];

            layer.off('mouseover').off('mouseout').off('click');

            if (val !== undefined && val !== null && !isNaN(val)) {{
                let color = getColor(val);

                layer.setStyle({{
                    fillColor: color,
                    fillOpacity: 0.8,
                    color: "black",
                    weight: 1
                }});

                layer.on("mouseover", function(e) {{
                    e.target.setStyle({{ weight: 3, color: "#222", fillOpacity: 0.95 }});
                }});

                layer.on("mouseout", function(e) {{
                    e.target.setStyle({{
                        weight: 1,
                        color: "black",
                        fillOpacity: 0.8,
                        fillColor: color
                    }});
                }});

                layer.on("click", function(e) {{
                    // 팝업 + 그래프 둘 다
                    var currentYearLabel = year;
                    if (year === predYear) {{
                        currentYearLabel = year + " (예측)";
                    }}
                    var valueText = val.toFixed(2);
                    var popupHtml = "<b>" + name + "</b><br>" +
                                    "지표: " + metric + "<br>" +
                                    "값: " + valueText + "<br>" +
                                    "연도: " + currentYearLabel;

                    e.target.bindPopup(popupHtml).openPopup();
                    showTrendChart(name, metric);
                }});
            }} else {{
                // 데이터 없음
                layer.setStyle({{
                    fillColor: "#CCCCCC",
                    fillOpacity: 0.6,
                    color: "black",
                    weight: 1
                }});

                layer.on("mouseover", function(e) {{
                    e.target.setStyle({{ weight: 3, color: "#222", fillOpacity: 0.8 }});
                }});

                layer.on("mouseout", function(e) {{
                    e.target.setStyle({{
                        weight: 1,
                        color: "black",
                        fillOpacity: 0.6,
                        fillColor: "#CCCCCC"
                    }});
                }});

                layer.on("click", function(e) {{
                    var popupHtml = "<b>" + name + "</b><br>데이터 없음";
                    e.target.bindPopup(popupHtml).openPopup();
                    showTrendChart(name, metric); // 그래프는 0으로 들어감
                }});
            }}
        }}
    }});
}}

/////////////////////////////////////////////////////////////////////
// 오른쪽 차트 패널 + Chart.js 그래프 출력
/////////////////////////////////////////////////////////////////////
function showTrendChart(regionName, metric) {{
    var panel = document.getElementById("infoPanel");
    panel.style.display = "block";

    // 연도 정렬 (숫자 기준)
    var years = Object.keys(regionData).sort(function(a, b) {{
        return parseInt(a) - parseInt(b);
    }});

    // 그래프용 라벨(예측연도는 표시만 다르게)
    var labels = years.map(function(y) {{
        if (y === predYear) {{
            return y + " (예측)";
        }}
        return y;
    }});

    // 값 (없으면 0으로 채움)
    var values = years.map(function(y) {{
        var block = regionData[y];
        if (!block || !block[metric]) return 0;
        var v = block[metric][regionName];
        if (v === undefined || v === null || isNaN(v)) return 0;
        return v;
    }});

    // 현재 콤보박스에서 선택된 연도 값 표시
    var currentYear = document.getElementById("yearSelect").value;
    var currentLabel = currentYear;
    if (currentYear === predYear) {{
        currentLabel = currentYear + " (예측)";
    }}

    var currentVal = 0;
    if (regionData[currentYear] &&
        regionData[currentYear][metric] &&
        regionData[currentYear][metric][regionName] !== undefined &&
        regionData[currentYear][metric][regionName] !== null &&
        !isNaN(regionData[currentYear][metric][regionName])) {{
        currentVal = regionData[currentYear][metric][regionName];
    }}

    var infoText = regionName + " / " + metric + " / " + currentLabel +
                   " : " + currentVal.toFixed(2);
    var infoElem = document.getElementById("regionInfoText");
    if (infoElem) {{
        infoElem.innerText = infoText;
    }}

    if (window.trendChartObj) {{
        window.trendChartObj.destroy();
    }}

    var ctx = document.getElementById("trendChart").getContext("2d");

    window.trendChartObj = new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: labels,
            datasets: [{{
                label: regionName + " - " + metric,
                data: values,
                borderColor: "#FF5733",
                backgroundColor: "rgba(255, 87, 51, 0.2)",
                tension: 0.25,
                pointRadius: 4
            }}]
        }},
        options: {{
            scales: {{
                y: {{ beginAtZero: false }}
            }}
        }}
    }});
}}

/////////////////////////////////////////////////////////////////////
// 오른쪽 상단 위험도 5단계 컬러바
/////////////////////////////////////////////////////////////////////
function addColorBar() {{
    var bar = document.createElement("div");
    bar.style.position = "absolute";
    bar.style.top = "70px";
    bar.style.right = "10px";
    bar.style.zIndex = "9999";
    bar.style.padding = "12px";
    bar.style.background = "rgba(255,255,255,0.95)";
    bar.style.borderRadius = "8px";
    bar.style.width = "220px";
    bar.style.boxShadow = "0 2px 6px rgba(0,0,0,0.25)";
    bar.style.fontSize = "13px";

    bar.innerHTML =
        '<b id="colorBarTitle">{initial_metric} ({initial_year}년)</b>' +
        '<hr style="margin: 8px 0 10px 0; border: 0; border-top: 1px solid #ddd;">' +
        '<div style="margin-bottom:5px; display:flex; align-items:center;">' +
            '<i style="background:#6CC24A;width:14px;height:14px;display:inline-block;margin-right:8px;border:1px solid #333;"></i>' +
            '<span>0 ~ 10 (매우 낮음)</span>' +
        '</div>' +
        '<div style="margin-bottom:5px; display:flex; align-items:center;">' +
            '<i style="background:#A8E05F;width:14px;height:14px;display:inline-block;margin-right:8px;border:1px solid #333;"></i>' +
            '<span>10 ~ 15 (낮음)</span>' +
        '</div>' +
        '<div style="margin-bottom:5px; display:flex; align-items:center;">' +
            '<i style="background:#FFD700;width:14px;height:14px;display:inline-block;margin-right:8px;border:1px solid #333;"></i>' +
            '<span>15 ~ 20 (중간)</span>' +
        '</div>' +
        '<div style="margin-bottom:5px; display:flex; align-items:center;">' +
            '<i style="background:#FF8C42;width:14px;height:14px;display:inline-block;margin-right:8px;border:1px solid #333;"></i>' +
            '<span>20 ~ 25 (높음)</span>' +
        '</div>' +
        '<div style="margin-bottom:5px; display:flex; align-items:center;">' +
            '<i style="background:#FF5733;width:14px;height:14px;display:inline-block;margin-right:8px;border:1px solid #333;"></i>' +
            '<span>25 이상 (매우 높음)</span>' +
        '</div>' +
        '<div style="margin-top:10px; padding-top:6px; border-top:1px dashed #ddd;">' +
            '<i style="background:#CCCCCC;width:14px;height:14px;display:inline-block;margin-right:8px;border:1px solid #333;"></i>' +
            '<span>데이터 없음</span>' +
        '</div>';

    document.body.appendChild(bar);
}}

/////////////////////////////////////////////////////////////////////
// 중앙 제목
/////////////////////////////////////////////////////////////////////
function addTitle() {{
    var title = document.createElement("div");
    title.style.position = "absolute";
    title.style.top = "10px";
    title.style.left = "50%";
    title.style.transform = "translateX(-50%)";
    title.style.background = "rgba(255,255,255,0.95)";
    title.style.padding = "10px 18px";
    title.style.borderRadius = "8px";
    title.style.fontSize = "20px";
    title.style.fontWeight = "bold";
    title.style.zIndex = "9999";
    title.style.boxShadow = "0 2px 6px rgba(0,0,0,0.35)";
    title.innerHTML = "대한민국 청년 정신건강 위험도 지도";
    document.body.appendChild(title);
}}

/////////////////////////////////////////////////////////////////////
// 초기 실행
/////////////////////////////////////////////////////////////////////
document.addEventListener("DOMContentLoaded", function() {{
    addTitle();
    addColorBar();

    // 지도가 완전히 로드된 후 색칠 (약간의 딜레이)
    setTimeout(updateMap, 800);
}});
</script>

<!-- 오른쪽 정보 패널 -->
<div id="infoPanel" style="
    position:absolute;
    top:70px;
    right:240px;
    width:320px;
    height:360px;
    background:white;
    padding:12px 15px;
    border-radius:8px;
    box-shadow:0 2px 6px rgba(0,0,0,0.25);
    display:none;
    z-index:9999;
    overflow-y:auto;
">
    <div id="regionInfoText" style="margin-bottom:8px; font-size:13px; font-weight:bold;"></div>
    <canvas id="trendChart" width="290" height="260"></canvas>
</div>

<!-- Chart.js -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<!-- 왼쪽 상단 UI (연도, 지표 선택) -->
<div style="
    position:absolute;
    top:70px; left:10px;
    background:white;
    padding:10px 14px;
    border-radius:8px;
    box-shadow:0 2px 6px rgba(0,0,0,0.25);
    z-index:9999;
    font-size:14px;
">
    <label><b>연도:</b></label>
    <select id="yearSelect">
        {year_options_html}
    </select>

    <label style="margin-left:8px;"><b>지표:</b></label>
    <select id="metricSelect">
        {metric_options_html}
    </select>

    <button onclick="updateMap()" style="margin-left:10px;">업데이트</button>
</div>
"""

m.get_root().html.add_child(folium.Element(custom_js))

# ============================
# 저장
# ============================
m.save("지역_위험_지도.html")
print("생성 완료: 지역_위험_지도.html")
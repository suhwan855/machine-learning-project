from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


# 엑셀 파일 로드
file_path = "../mental_health_data.xlsx"

mental = pd.read_excel(file_path, sheet_name="mental_health_stats")
socio = pd.read_excel(file_path, sheet_name="socioeconomic_stats")

# year + region 기준으로 병합
df = pd.merge(mental, socio, on=["year", "region"])

# 예측 목표(Y)
target = "youth_suicide_rate"

# 포함할 특징(X)
features = [
    "youth_unemployment_rate",
    "average_income",
    "one_person_household_rate",
    "education_index",
    "housing_cost_index",
    "crime_rate",
    "welfare_access",
    "stress_rate",
    "depression_rate"
]

X = df[features]
y = df[target]

# train/test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4)
model.fit(X_train, y_train)

pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("XGBoost RMSE:", rmse)

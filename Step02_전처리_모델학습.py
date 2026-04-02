# =============================================================
#  Step 02 — 전처리 & 모델 학습
#  강의: 통계에 의한 AI 활용 예측 및 분석 솔루션 개발
#  목표: 피처 엔지니어링 → 모델 학습 → 성능 비교 → 저장
# =============================================================
# 실행: python Step02_전처리_모델학습.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, warnings, joblib
warnings.filterwarnings('ignore')

# ── 한글 폰트 ─────────────────────────────────────────
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    try:
        plt.rcParams['font.family'] = 'NanumGothic'
    except Exception:
        plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'

# =============================================================
#  1. 데이터 로드
# =============================================================
print("=" * 55)
print("  STEP 02 — 전처리 & 모델 학습")
print("=" * 55)

# Step01에서 저장한 클린 데이터 우선 사용, 없으면 원본 사용
clean_path = os.path.join(BASE_DIR, 'data', 'sales_clean.csv')
raw_path   = os.path.join(BASE_DIR, 'data', 'sample_sales.csv')
path = clean_path if os.path.exists(clean_path) else raw_path
df   = pd.read_csv(path)
df['date'] = pd.to_datetime(df['date'])
print(f"\n  데이터 로드: {path} ({len(df)}행)")

# =============================================================
#  2. 피처 엔지니어링 (Feature Engineering)
# =============================================================
print("\n[1] 피처 엔지니어링")
print("-" * 40)

# 결측치 재처리 (원본 사용 시)
df['temp']       = df['temp'].fillna(df['temp'].mean())
df['prev_sales'] = df['prev_sales'].fillna(df['prev_sales'].mean())

# 날씨 인코딩 (one-hot encoding)
df['weather_sunny'] = (df['weather'] == '맑음').astype(int)
df['weather_rain']  = (df['weather'] == '비').astype(int)
df['weather_snow']  = (df['weather'] == '눈').astype(int)

# 월 (계절성 반영)
df['month'] = df['date'].dt.month

# 분기
df['quarter'] = df['date'].dt.quarter

# 피처 목록 정의
FEATURES = [
    'temp',          # 기온
    'is_weekend',    # 주말 여부
    'holiday',       # 공휴일 여부
    'event',         # 행사 여부
    'prev_sales',    # 전주 동일요일 매출
    'month',         # 월 (계절성)
    'weather_sunny', # 맑음 여부
    'weather_rain',  # 비 여부
    'weather_snow',  # 눈 여부
]
TARGET = 'sales'

X = df[FEATURES]
y = df[TARGET]

print(f"  입력 피처 수: {len(FEATURES)}개")
print(f"  피처 목록: {FEATURES}")
print(f"  타겟: {TARGET} (범위: {y.min():.1f} ~ {y.max():.1f}만원)")

# =============================================================
#  3. 훈련/테스트 분리
# =============================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[2] 데이터 분리 (8:2)")
print(f"  훈련: {len(X_train)}행  |  테스트: {len(X_test)}행")

# =============================================================
#  4. 다중 모델 학습 및 비교
# =============================================================
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

print("\n[3] 5가지 알고리즘 학습 및 비교")
print("-" * 55)

models = {
    '선형 회귀 (Linear Regression)':        LinearRegression(),
    'Ridge 회귀 (Ridge Regression)':         Ridge(alpha=1.0),
    '의사결정 나무 (Decision Tree)':         DecisionTreeRegressor(max_depth=6, random_state=42),
    '랜덤 포레스트 (Random Forest)':         RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':                      GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
print(f"{'모델':<38} {'RMSE':>8} {'MAE':>8} {'R²':>7}")
print("-" * 65)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred  = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, pred))
    mae   = mean_absolute_error(y_test, pred)
    r2    = r2_score(y_test, pred)
    results[name] = {'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2, 'pred': pred}
    mark = " ★" if rmse == min(r['rmse'] for r in results.values()) else ""
    print(f"  {name:<36} {rmse:>7.2f} {mae:>7.2f} {r2:>6.4f}{mark}")

print("-" * 65)

# 최고 성능 모델 선택
best_name  = min(results, key=lambda k: results[k]['rmse'])
best_model = results[best_name]['model']
best_rmse  = results[best_name]['rmse']

print(f"\n  ★ 최고 성능 모델: {best_name}")
print(f"     RMSE: {best_rmse:.2f}만원  |  R²: {results[best_name]['r2']:.4f}")

# =============================================================
#  5. 랜덤 포레스트 하이퍼파라미터 튜닝
# =============================================================
print("\n[4] 랜덤 포레스트 하이퍼파라미터 탐색")
print("-" * 40)

from sklearn.model_selection import cross_val_score

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [None, 5, 10],
}

best_cv_rmse = float('inf')
best_params  = {}

for n_est in param_grid['n_estimators']:
    for max_d in param_grid['max_depth']:
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=42)
        scores = cross_val_score(rf, X_train, y_train,
                                  scoring='neg_root_mean_squared_error', cv=5)
        cv_rmse = -scores.mean()
        if cv_rmse < best_cv_rmse:
            best_cv_rmse = cv_rmse
            best_params  = {'n_estimators': n_est, 'max_depth': max_d}
        print(f"  n_estimators={n_est:>3}, max_depth={str(max_d):>5} → CV-RMSE: {cv_rmse:.2f}")

print(f"\n  ✅ 최적 파라미터: {best_params} (CV-RMSE: {best_cv_rmse:.2f}만원)")

# 최종 모델 재학습
final_model = RandomForestRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)
final_pred  = final_model.predict(X_test)
final_rmse  = np.sqrt(mean_squared_error(y_test, final_pred))
final_r2    = r2_score(y_test, final_pred)

print(f"\n  최종 모델 테스트 성능")
print(f"    RMSE : {final_rmse:.2f}만원")
print(f"    MAE  : {mean_absolute_error(y_test, final_pred):.2f}만원")
print(f"    R²   : {final_r2:.4f}")
print(f"    목표(RMSE < 15만원): {'✅ 달성!' if final_rmse < 15 else '❌ 미달성 — 피처 추가 필요'}")

# =============================================================
#  6. 모델 비교 차트
# =============================================================
print("\n[5] 모델 비교 차트 — 저장: model_comparison.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('모델 성능 비교', fontsize=14, fontweight='bold')

names_short = ['선형회귀', 'Ridge', '결정나무', '랜덤포레스트', 'GBM']
rmse_vals   = [results[n]['rmse'] for n in models.keys()]
r2_vals     = [results[n]['r2']   for n in models.keys()]
bar_colors  = ['#BDC3C7']*3 + ['#28B463'] + ['#2E86C1']

axes[0].barh(names_short, rmse_vals, color=bar_colors, edgecolor='white')
axes[0].set_title('RMSE 비교 (낮을수록 좋음)', fontweight='bold')
axes[0].set_xlabel('RMSE (만원)')
for i, v in enumerate(rmse_vals):
    axes[0].text(v+0.2, i, f'{v:.2f}', va='center', fontsize=10)

axes[1].barh(names_short, r2_vals, color=bar_colors, edgecolor='white')
axes[1].set_title('R² 비교 (높을수록 좋음)', fontweight='bold')
axes[1].set_xlabel('R² (결정계수)')
axes[1].set_xlim(0, 1.0)
for i, v in enumerate(r2_vals):
    axes[1].text(v+0.005, i, f'{v:.3f}', va='center', fontsize=10)

plt.tight_layout()
save_path = os.path.join(BASE_DIR, 'model_comparison.png')
plt.savefig(save_path, dpi=120, bbox_inches='tight')
plt.show()
print(f"    저장 완료: {save_path}")

# =============================================================
#  7. 모델 저장
# =============================================================
print("\n[6] 모델 저장")
model_path    = os.path.join(BASE_DIR, 'sales_model.pkl')
features_path = os.path.join(BASE_DIR, 'features.pkl')
joblib.dump(final_model, model_path)
joblib.dump(FEATURES, features_path)
print(f"    모델 저장: {model_path}")
print(f"    피처 목록 저장: {features_path}")

print("\n" + "=" * 55)
print("  ✅ Step 02 완료!")
print(f"  최종 모델: RandomForest (RMSE: {final_rmse:.2f}만원)")
print("  다음: python Step03_평가_시각화.py")
print("=" * 55)

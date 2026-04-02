# =============================================================
#  Step 03 — 모델 평가 & 시각화 & 인사이트 도출
#  강의: 통계에 의한 AI 활용 예측 및 분석 솔루션 개발
#  목표: 예측 vs 실제 비교 / Feature Importance / 오차 분석
# =============================================================
# 실행: python Step03_평가_시각화.py
# =============================================================

import pandas as pd
import numpy as np
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
#  1. 모델 & 데이터 로드
# =============================================================
print("=" * 55)
print("  STEP 03 — 모델 평가 & 시각화")
print("=" * 55)

# 저장된 모델 로드
model_path    = os.path.join(BASE_DIR, 'sales_model.pkl')
features_path = os.path.join(BASE_DIR, 'features.pkl')

if not os.path.exists(model_path):
    print("  ⚠️  sales_model.pkl 없음 → Step02 먼저 실행하세요")
    print("     python Step02_전처리_모델학습.py")
    import sys; sys.exit(1)

model    = joblib.load(model_path)
FEATURES = joblib.load(features_path)
print(f"\n  모델 로드: {model_path}")

# 데이터 로드 & 전처리
path = os.path.join(BASE_DIR, 'data', 'sales_clean.csv')
if not os.path.exists(path):
    path = os.path.join(BASE_DIR, 'data', 'sample_sales.csv')

df = pd.read_csv(path)
df['date'] = pd.to_datetime(df['date'])
df['temp']          = df['temp'].fillna(df['temp'].mean())
df['prev_sales']    = df['prev_sales'].fillna(df['prev_sales'].mean())
df['weather_sunny'] = (df['weather'] == '맑음').astype(int)
df['weather_rain']  = (df['weather'] == '비').astype(int)
df['weather_snow']  = (df['weather'] == '눈').astype(int)
df['month']         = df['date'].dt.month

from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

X = df[FEATURES]
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)

# =============================================================
#  2. 성능 지표 출력
# =============================================================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n[1] 최종 모델 성능 지표")
print("-" * 40)
print(f"  RMSE  : {rmse:.2f}만원  (예측 오차 ±{rmse:.1f}만원)")
print(f"  MAE   : {mae:.2f}만원")
print(f"  R²    : {r2:.4f}  ({r2*100:.1f}% 설명력)")
print(f"  MAPE  : {mape:.2f}%  (평균 오차율)")
print(f"\n  목표 달성 여부 (RMSE < 15만원): {'✅ 달성!' if rmse < 15 else '❌ 미달성'}")

# =============================================================
#  3. 예측 vs 실제 4대 차트
# =============================================================
print("\n[2] 평가 차트 4종 — 저장: evaluation_charts.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(f'모델 평가 결과  (RMSE: {rmse:.2f}만원  |  R²: {r2:.3f})',
             fontsize=14, fontweight='bold')

errors = y_pred - y_test.values

# (1) 예측 vs 실제 산점도
perfect = np.linspace(y_test.min(), y_test.max(), 100)
axes[0,0].scatter(y_test, y_pred, alpha=0.45, s=25, c=np.abs(errors), cmap='RdYlGn_r')
axes[0,0].plot(perfect, perfect, 'r--', linewidth=1.5, label='완벽한 예측 선')
axes[0,0].set_title('예측 vs 실제 산점도', fontweight='bold')
axes[0,0].set_xlabel('실제 매출 (만원)')
axes[0,0].set_ylabel('예측 매출 (만원)')
axes[0,0].legend()
axes[0,0].text(0.05, 0.92, f'R² = {r2:.3f}', transform=axes[0,0].transAxes,
               fontsize=11, color='navy', fontweight='bold')

# (2) 오차(잔차) 분포
axes[0,1].hist(errors, bins=25, color='#2E86C1', edgecolor='white', alpha=0.8)
axes[0,1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0,1].axvline(rmse,  color='orange', linestyle=':', linewidth=1.5, label=f'+RMSE({rmse:.1f})')
axes[0,1].axvline(-rmse, color='orange', linestyle=':', linewidth=1.5, label=f'-RMSE({rmse:.1f})')
axes[0,1].set_title('예측 오차(잔차) 분포', fontweight='bold')
axes[0,1].set_xlabel('오차 (예측 - 실제, 만원)')
axes[0,1].set_ylabel('빈도')
axes[0,1].legend(fontsize=9)

# (3) Feature Importance
feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
feat_colors = ['#117A65' if v == feat_imp.max() else '#2E86C1' for v in feat_imp.values]
axes[1,0].barh(feat_imp.index, feat_imp.values, color=feat_colors, edgecolor='white')
axes[1,0].set_title('피처 중요도 (Feature Importance)', fontweight='bold')
axes[1,0].set_xlabel('중요도')
for i, v in enumerate(feat_imp.values):
    axes[1,0].text(v+0.001, i, f'{v:.3f}', va='center', fontsize=9)

# (4) 시간순 예측 vs 실제
test_df = X_test.copy()
test_df['actual'] = y_test.values
test_df['pred']   = y_pred
test_df = test_df.sort_index().head(60)  # 처음 60일
axes[1,1].plot(range(len(test_df)), test_df['actual'], 'b-o',
               markersize=4, linewidth=1.5, alpha=0.8, label='실제 매출')
axes[1,1].plot(range(len(test_df)), test_df['pred'],   'r--s',
               markersize=4, linewidth=1.5, alpha=0.8, label='예측 매출')
axes[1,1].fill_between(range(len(test_df)),
                        test_df['actual'] - rmse,
                        test_df['actual'] + rmse,
                        alpha=0.12, color='blue', label=f'±RMSE({rmse:.1f})')
axes[1,1].set_title('테스트 데이터 예측 추이 (60건)', fontweight='bold')
axes[1,1].set_xlabel('샘플 번호')
axes[1,1].set_ylabel('매출 (만원)')
axes[1,1].legend(fontsize=9)

plt.tight_layout()
save_path = os.path.join(BASE_DIR, 'evaluation_charts.png')
plt.savefig(save_path, dpi=120, bbox_inches='tight')
plt.show()
print(f"    저장 완료: {save_path}")

# =============================================================
#  4. Feature Importance 인사이트
# =============================================================
feat_series = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

print("\n[3] 피처 중요도 분석 결과")
print("-" * 50)
for feat, imp in feat_series.items():
    bar = '█' * int(imp * 50)
    print(f"  {feat:<18} {bar:<25} {imp:.4f} ({imp*100:.1f}%)")

print("\n[4] 비즈니스 인사이트")
print("-" * 50)
top1 = feat_series.index[0]
top2 = feat_series.index[1]
top3 = feat_series.index[2]
print(f"  ▶ 가장 중요한 변수: {top1} ({feat_series[top1]*100:.1f}%)")
print(f"  ▶ 2위: {top2} ({feat_series[top2]*100:.1f}%)")
print(f"  ▶ 3위: {top3} ({feat_series[top3]*100:.1f}%)")
print()

# 해석 자동 생성
insights = {
    'prev_sales':    "→ 전주 매출 패턴이 강하게 반복됨. 주간 계절성 뚜렷.",
    'temp':          "→ 날씨/계절이 매출에 직접 영향. 기온 API 연동 시 정확도 UP.",
    'month':         "→ 계절성(월별 패턴)이 중요. 여름/연말 특수 반영 필요.",
    'is_weekend':    "→ 주말 프리미엄 존재. 주말 재고/인력 배치 차별화 권장.",
    'event':         "→ 행사 여부가 매출에 영향. 지역 이벤트 캘린더 연동 권장.",
    'holiday':       "→ 공휴일 효과 있음. 공휴일 전날 발주 증량 전략 유효.",
    'weather_sunny': "→ 맑은 날 외부 활동 증가 → 편의점 방문 증가.",
    'weather_rain':  "→ 비 오는 날 우산·음료 판매 증가 패턴 확인.",
    'weather_snow':  "→ 눈 오는 날 방문객 감소. 입지별 차이 있음.",
}
for feat in [top1, top2, top3]:
    if feat in insights:
        print(f"  [{feat}] {insights[feat]}")

# =============================================================
#  5. 오차 케이스 분석 (큰 오차가 난 날)
# =============================================================
print("\n[5] 오차 큰 케이스 TOP 5 분석")
print("-" * 50)
test_result = X_test.copy()
test_result['actual']   = y_test.values
test_result['predicted'] = y_pred
test_result['error']    = np.abs(errors)
test_result['error_pct'] = (np.abs(errors) / y_test.values * 100)

worst5 = test_result.nlargest(5, 'error')
print(worst5[['actual','predicted','error','error_pct','event','holiday','weather_snow']].round(2).to_string())
print("\n  → 오차가 큰 날의 패턴을 분석해 추가 피처를 발굴하세요!")

print("\n" + "=" * 55)
print("  ✅ Step 03 완료!")
print("  모델 성능 확인 및 인사이트 도출 완료")
print("  다음: python Step04_Streamlit앱.py")
print("=" * 55)

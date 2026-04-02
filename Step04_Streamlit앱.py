# =============================================================
#  Step 04 — Streamlit 예측 앱
#  강의: 통계에 의한 AI 활용 예측 및 분석 솔루션 개발
#  목표: 학습된 모델을 인터랙티브 웹 앱으로 배포
# =============================================================
#  실행 방법:
#    1. 로컬:        streamlit run Step04_Streamlit앱.py
#    2. antigravity: 해당 플랫폼의 Streamlit 실행 버튼 사용
#    3. Colab:       !streamlit run Step04_Streamlit앱.py &
#                    (ngrok 터널 필요)
#
#  주의: Step02가 먼저 실행되어 sales_model.pkl 이 있어야 합니다.
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from datetime import date

# ── 페이지 설정 ───────────────────────────────────────
st.set_page_config(
    page_title="편의점 매출 예측 AI",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 색상 테마 (CSS) ───────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        color: #1B4F72; text-align: center;
        padding: 0.5rem 0 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem; color: #666;
        text-align: center; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1B4F72, #2E86C1);
        border-radius: 12px; padding: 1.2rem;
        color: white; text-align: center;
    }
    .metric-value {
        font-size: 2.4rem; font-weight: 800; color: #28B463;
    }
    .metric-label {
        font-size: 0.9rem; opacity: 0.9; margin-top: 0.3rem;
    }
    .insight-box {
        background: #EBF5FB; border-left: 5px solid #1B4F72;
        padding: 1rem; border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #FEF9E7; border-left: 5px solid #F47920;
        padding: 0.8rem; border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── 헬퍼: 모델 로드 or 자동 학습 (캐시) ──────────────
@st.cache_resource
def load_model():
    from sklearn.ensemble import GradientBoostingRegressor
    base  = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    mpath = os.path.join(base, 'sales_model.pkl')
    fpath = os.path.join(base, 'features.pkl')

    # pkl 파일이 있으면 바로 로드
    if os.path.exists(mpath) and os.path.exists(fpath):
        return joblib.load(mpath), joblib.load(fpath)

    # ── pkl 없으면 데이터로 즉시 학습 ──────────────────
    for fname in ['data/sales_clean.csv', 'data/sample_sales.csv']:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    else:
        return None, None   # 데이터도 없으면 포기

    # 결측치 처리
    df['temp']       = df['temp'].fillna(df['temp'].median())
    df['prev_sales'] = df['prev_sales'].fillna(df['prev_sales'].median())

    # 피처 엔지니어링
    df['weather_sunny'] = (df['weather'] == '맑음').astype(int)
    df['weather_rain']  = (df['weather'] == '비').astype(int)
    df['weather_snow']  = (df['weather'] == '눈').astype(int)
    df['month']         = pd.to_datetime(df['date']).dt.month

    FEATURES = ['temp','is_weekend','holiday','event','prev_sales',
                'month','weather_sunny','weather_rain','weather_snow']

    X = df[FEATURES]
    y = df['sales']

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                      max_depth=4, random_state=42)
    model.fit(X, y)

    # 저장 시도 (권한 없어도 앱은 동작)
    try:
        joblib.dump(model,    mpath)
        joblib.dump(FEATURES, fpath)
    except Exception:
        pass

    return model, FEATURES

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    for fname in ['data/sales_clean.csv', 'data/sample_sales.csv']:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['date'] = pd.to_datetime(df['date'])
            return df
    return None

# ── 헤더 ─────────────────────────────────────────────
st.markdown('<div class="main-title">🏪 편의점 매출 예측 AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">통계 + 머신러닝으로 만드는 예측 솔루션 | AI 앱크리에이터 과정 실습</div>',
            unsafe_allow_html=True)
st.markdown("---")

# ── 모델 로드 ─────────────────────────────────────────
model, FEATURES = load_model()
df_all = load_data()

if model is None:
    st.error("""
    ⚠️ **데이터 파일(sample_sales.csv)을 찾을 수 없습니다.**

    GitHub 저장소에 `data/sample_sales.csv` 파일이 포함되어 있는지 확인하세요.
    """)
    st.stop()

# ── 사이드바: 입력 ────────────────────────────────────
with st.sidebar:
    st.header("📊 예측 조건 입력")
    st.markdown("---")

    temp    = st.slider("🌡️ 기온 (℃)",     min_value=-15, max_value=40, value=20, step=1)
    prev    = st.number_input("📅 전주 동일 요일 매출 (만원)",
                               min_value=80.0, max_value=400.0, value=200.0, step=5.0)
    month   = st.selectbox("📆 월",
                            options=list(range(1,11)),
                            index=date.today().month - 1 if date.today().month <= 10 else 0,
                            format_func=lambda x: f"{x}월")
    weather = st.selectbox("🌤️ 날씨",
                            options=['맑음', '흐림', '비', '눈'])
    st.markdown("---")
    weekend = st.toggle("🗓️ 주말 여부",       value=False)
    holiday = st.toggle("🎌 공휴일 여부",      value=False)
    event   = st.toggle("🎪 인근 행사 여부",   value=False)
    st.markdown("---")
    predict_btn = st.button("🔮 매출 예측하기", type="primary", use_container_width=True)

# ── 메인: 예측 결과 ───────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 1])

# 입력값 구성
input_dict = {
    'temp':          temp,
    'is_weekend':    int(weekend),
    'holiday':       int(holiday),
    'event':         int(event),
    'prev_sales':    prev,
    'month':         month,
    'weather_sunny': 1 if weather == '맑음' else 0,
    'weather_rain':  1 if weather == '비'   else 0,
    'weather_snow':  1 if weather == '눈'   else 0,
}
X_input = pd.DataFrame([{f: input_dict.get(f, 0) for f in FEATURES}])
pred_val = model.predict(X_input)[0]
delta    = pred_val - prev

# 예측 결과 표시 (버튼 클릭 또는 실시간)
with col1:
    st.subheader("📈 예측 결과")
    st.metric(
        label="예측 매출",
        value=f"{pred_val:,.0f} 만원",
        delta=f"{delta:+.0f} 만원 (전주 대비)"
    )

    # 신뢰 구간 (±RMSE 기준 약 ±11만원 가정)
    rmse_est = 11.0
    st.markdown(f"""
    <div class="insight-box">
    <b>예측 범위 (±RMSE)</b><br>
    최소 <b>{max(80, pred_val-rmse_est):.0f}만원</b> ~
    최대 <b>{pred_val+rmse_est:.0f}만원</b>
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        st.balloons()

with col2:
    st.subheader("📋 입력 조건 요약")
    conditions = {
        "기온":        f"{temp}℃",
        "날씨":        weather,
        "요일":        "주말" if weekend else "평일",
        "공휴일":      "예" if holiday else "아니오",
        "행사":        "있음" if event else "없음",
        "월":          f"{month}월",
        "전주 매출":   f"{prev:.0f}만원",
    }
    cdf = pd.DataFrame(list(conditions.items()), columns=['조건', '값'])
    st.dataframe(cdf, use_container_width=True, hide_index=True)

with col3:
    st.subheader("💡 매출 등급")
    if pred_val >= 260:
        grade = "🟢 최고 매출"
        color = "#28B463"
    elif pred_val >= 220:
        grade = "🔵 우수 매출"
        color = "#2E86C1"
    elif pred_val >= 180:
        grade = "🟡 평균 매출"
        color = "#F4D03F"
    else:
        grade = "🔴 저조 매출"
        color = "#E74C3C"

    st.markdown(f"""
    <div style="background:{color}22; border:2px solid {color};
                border-radius:12px; padding:1rem; text-align:center;">
        <div style="font-size:1.6rem; font-weight:800; color:{color};">
            {pred_val:,.0f}<br><small>만원</small>
        </div>
        <div style="font-size:1.0rem; margin-top:0.5rem;">{grade}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Feature Importance 차트 ───────────────────────────
col_fi, col_trend = st.columns(2)

with col_fi:
    st.subheader("🔍 피처 중요도 (Feature Importance)")
    feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax  = plt.subplots(figsize=(7, 4))
    colors   = ['#F47920' if v == feat_imp.max() else '#2E86C1' for v in feat_imp.values]
    ax.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor='white')
    ax.set_xlabel('중요도')
    ax.set_title('어떤 변수가 예측에 가장 영향을 줬나?', fontweight='bold')
    for i, v in enumerate(feat_imp.values):
        ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

with col_trend:
    st.subheader("📊 월별 예측 매출 시뮬레이션")
    months     = list(range(1, 11))
    month_pred = []
    for m in months:
        x_sim = pd.DataFrame([{
            'temp': 5 + 20 * np.sin((m/12)*2*np.pi - 1.5),
            'is_weekend': int(weekend), 'holiday': 0,
            'event': int(event), 'prev_sales': prev,
            'month': m,
            'weather_sunny': 1 if m in [4,5,6,7,8,9] else 0,
            'weather_rain':  1 if m in [6,7] else 0,
            'weather_snow':  1 if m in [1,2] else 0,
        }])
        x_sim = x_sim.reindex(columns=FEATURES, fill_value=0)
        month_pred.append(model.predict(x_sim)[0])

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(months, month_pred, 'o-', color='#1B4F72', linewidth=2.5, markersize=8)
    ax2.fill_between(months, month_pred, min(month_pred)*0.95, alpha=0.15, color='#1B4F72')
    ax2.axvline(month, color='#F47920', linestyle='--', linewidth=1.5, label=f'현재 선택: {month}월')
    ax2.set_xlabel('월')
    ax2.set_ylabel('예측 매출 (만원)')
    ax2.set_title('현재 조건 기준 월별 예측 추이', fontweight='bold')
    ax2.legend()
    ax2.set_xticks(months)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

# ── 실제 데이터 탐색 ──────────────────────────────────
if df_all is not None:
    st.markdown("---")
    with st.expander("📂 원본 데이터 탐색 (sample_sales.csv)", expanded=False):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.subheader("데이터 미리보기")
            st.dataframe(df_all.head(20), use_container_width=True)
        with col_d2:
            st.subheader("기술 통계")
            st.dataframe(df_all[['temp','prev_sales','sales']].describe().round(2),
                         use_container_width=True)
            st.subheader("결측치 현황")
            missing = df_all.isnull().sum()
            st.dataframe(missing[missing > 0].rename("결측치 수").to_frame(),
                         use_container_width=True)

# ── 푸터 ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding:0.5rem;">
    AI 앱크리에이터 과정 실습 | 통계에 의한 AI 활용 예측 및 분석 솔루션 개발<br>
    강사: 제조혁신 길라잡이 김사부 | RandomForest 기반 편의점 매출 예측 모델
</div>
""", unsafe_allow_html=True)

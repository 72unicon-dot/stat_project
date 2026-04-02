import streamlit as st, joblib, numpy as np

# 모델 로드
model = joblib.load("sales_model.pkl")

st.title("편의점 매출 예측 AI")
st.markdown("---")

# 입력 슬라이더
temp    = st.slider("기온 (°C)", -10, 40, 20)
weekend = st.toggle("주말 여부")
holiday = st.toggle("공휴일 여부")
prev    = st.number_input("전주 동일 요일 매출 (만원)", 100, 500, 200)

# 예측
if st.button("매출 예측하기"):
    import datetime
    import pandas as pd
    features = joblib.load("features.pkl")
    input_df = pd.DataFrame([[temp, int(weekend), int(holiday), 0, prev, datetime.date.today().month, 1, 0, 0]], columns=features)
    pred = model.predict(input_df)
    st.metric("예측 매출", f"{pred[0]:,.0f} 만원",
              delta=f"{pred[0]-prev:+.0f} 만원 (전주 대비)")

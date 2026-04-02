# =============================================================
#  Step 01 — 탐색적 데이터 분석 (EDA)
#  강의: 통계에 의한 AI 활용 예측 및 분석 솔루션 개발
#  목표: 데이터 구조 파악 → 분포 확인 → 상관관계 분석
# =============================================================
# antigravity / Jupyter / Google Colab 모두 실행 가능
# 실행: python Step01_데이터탐색_EDA.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 설정 (시스템별 자동 감지) ─────────────────
def set_korean_font():
    """운영체제별 한글 폰트 자동 설정"""
    import platform
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    else:  # Linux / Colab / antigravity
        try:
            # Google Colab / Ubuntu 환경
            import subprocess
            subprocess.run(['apt-get', 'install', '-y', 'fonts-nanum'],
                          capture_output=True)
            fm._load_fontmanager(try_read_cache=False)
            plt.rcParams['font.family'] = 'NanumGothic'
        except Exception:
            plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# =============================================================
#  1. 데이터 로드
# =============================================================
print("=" * 55)
print("  STEP 01 — 탐색적 데이터 분석 (EDA)")
print("=" * 55)

# 파일 경로 (같은 폴더의 data 서브폴더)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
CSV_PATH = os.path.join(BASE_DIR, 'data', 'sample_sales.csv')

df = pd.read_csv(CSV_PATH)
df['date'] = pd.to_datetime(df['date'])

print(f"\n[1] 데이터 로드 완료")
print(f"    파일: {CSV_PATH}")
print(f"    행(rows): {df.shape[0]}, 열(columns): {df.shape[1]}")

# =============================================================
#  2. 기본 정보 확인
# =============================================================
print("\n[2] 컬럼 목록 및 데이터 타입")
print("-" * 40)
print(df.dtypes)

print("\n[3] 샘플 데이터 (상위 5행)")
print("-" * 40)
print(df.head().to_string())

print("\n[4] 기술 통계량")
print("-" * 40)
print(df.describe().round(2).to_string())

# =============================================================
#  3. 결측치 확인 및 처리
# =============================================================
print("\n[5] 결측치 현황")
print("-" * 40)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_info = pd.DataFrame({'결측치수': missing, '비율(%)': missing_pct})
print(missing_info[missing_info['결측치수'] > 0].to_string())

# 결측치 처리
print("\n[6] 결측치 처리 (수치형 → 평균값 대체)")
df['temp']       = df['temp'].fillna(df['temp'].mean())
df['prev_sales'] = df['prev_sales'].fillna(df['prev_sales'].mean())
print(f"    처리 후 결측치: {df.isnull().sum().sum()}개")

# =============================================================
#  4. 분포 시각화
# =============================================================
print("\n[7] 분포 시각화 (4개 차트) — 저장: eda_distribution.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('편의점 매출 예측 — 주요 변수 분포', fontsize=16, fontweight='bold', y=1.01)

# (1) 매출 분포 히스토그램
axes[0,0].hist(df['sales'], bins=30, color='#1B4F72', edgecolor='white', alpha=0.85)
axes[0,0].axvline(df['sales'].mean(), color='#F47920', linestyle='--', linewidth=2,
                   label=f"평균: {df['sales'].mean():.1f}만원")
axes[0,0].set_title('일별 매출 분포 (만원)', fontweight='bold')
axes[0,0].set_xlabel('매출 (만원)')
axes[0,0].set_ylabel('빈도')
axes[0,0].legend()

# (2) 요일별 평균 매출
dow_labels = ['월','화','수','목','금','토','일']
dow_avg = df.groupby('dayofweek')['sales'].mean()
colors  = ['#2E86C1']*5 + ['#F47920']*2
axes[0,1].bar(dow_labels, dow_avg.values, color=colors, edgecolor='white')
axes[0,1].set_title('요일별 평균 매출 (만원)', fontweight='bold')
axes[0,1].set_xlabel('요일')
axes[0,1].set_ylabel('평균 매출 (만원)')
for i, v in enumerate(dow_avg.values):
    axes[0,1].text(i, v+1, f'{v:.0f}', ha='center', fontsize=10)

# (3) 기온 vs 매출 산점도
sc = axes[1,0].scatter(df['temp'], df['sales'],
                        c=df['is_weekend'], cmap='RdYlGn',
                        alpha=0.55, s=22)
axes[1,0].set_title('기온 vs 매출 (색=주말여부)', fontweight='bold')
axes[1,0].set_xlabel('기온 (℃)')
axes[1,0].set_ylabel('매출 (만원)')
plt.colorbar(sc, ax=axes[1,0], label='주말(1)/평일(0)')

# (4) 월별 평균 매출 추이
df['month'] = df['date'].dt.month
monthly = df.groupby('month')['sales'].mean()
axes[1,1].plot(monthly.index, monthly.values, 'o-',
               color='#117A65', linewidth=2.5, markersize=8)
axes[1,1].fill_between(monthly.index, monthly.values, alpha=0.15, color='#117A65')
axes[1,1].set_title('월별 평균 매출 추이 (만원)', fontweight='bold')
axes[1,1].set_xlabel('월')
axes[1,1].set_ylabel('평균 매출 (만원)')
axes[1,1].set_xticks(range(1, 11))

plt.tight_layout()
save_path = os.path.join(BASE_DIR, 'eda_distribution.png')
plt.savefig(save_path, dpi=120, bbox_inches='tight')
plt.show()
print(f"    저장 완료: {save_path}")

# =============================================================
#  5. 상관관계 분석
# =============================================================
print("\n[8] 상관관계 분석 (sales와의 상관계수)")
print("-" * 40)
numeric_cols = ['temp', 'is_weekend', 'holiday', 'event', 'prev_sales', 'sales']
corr_matrix  = df[numeric_cols].corr()
corr_with_sales = corr_matrix['sales'].drop('sales').sort_values(ascending=False)
print(corr_with_sales.round(4).to_string())

print("\n[9] 상관관계 히트맵 — 저장: eda_correlation.png")
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
ax.set_yticklabels(numeric_cols)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, f'{corr_matrix.iloc[i,j]:.2f}',
                ha='center', va='center', fontsize=9,
                color='black' if abs(corr_matrix.iloc[i,j]) < 0.7 else 'white')
ax.set_title('변수 간 상관관계 히트맵', fontweight='bold', fontsize=13)
plt.tight_layout()
save_path2 = os.path.join(BASE_DIR, 'eda_correlation.png')
plt.savefig(save_path2, dpi=120, bbox_inches='tight')
plt.show()
print(f"    저장 완료: {save_path2}")

# =============================================================
#  6. EDA 인사이트 요약
# =============================================================
top_var = corr_with_sales.idxmax()
print("\n" + "=" * 55)
print("  EDA 인사이트 요약")
print("=" * 55)
print(f"  ▶ 데이터 기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"  ▶ 일 평균 매출: {df['sales'].mean():.1f}만원")
print(f"  ▶ 최고 매출일: {df.loc[df['sales'].idxmax(), 'date'].date()} "
      f"({df['sales'].max():.1f}만원)")
print(f"  ▶ 최저 매출일: {df.loc[df['sales'].idxmin(), 'date'].date()} "
      f"({df['sales'].min():.1f}만원)")
print(f"  ▶ 매출에 가장 큰 영향: {top_var} "
      f"(상관계수: {corr_with_sales.max():.3f})")
print(f"  ▶ 주말 평균 매출: {df[df['is_weekend']==1]['sales'].mean():.1f}만원 "
      f"(평일 {df[df['is_weekend']==0]['sales'].mean():.1f}만원)")
print(f"\n  ✅ Step 01 완료! 다음: python Step02_전처리_모델학습.py")
print("=" * 55)

# 전처리된 데이터 저장 (Step02에서 사용)
clean_path = os.path.join(BASE_DIR, 'data', 'sales_clean.csv')
df.to_csv(clean_path, index=False, encoding='utf-8-sig')
print(f"\n  전처리 데이터 저장: {clean_path}")

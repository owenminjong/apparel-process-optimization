# src/preprocessing/validate_results.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 결과 디렉토리 생성
os.makedirs('results', exist_ok=True)

def validate_preprocessing_results():
    """전처리된 데이터의 결과를 검증합니다."""
    # 전처리된 데이터 로드
    df = pd.read_csv('data/preprocessed_data.csv')

    # 기본 정보 확인
    print(f"데이터 크기: {df.shape[0]}행, {df.shape[1]}열")
    print(f"결측치 개수: {df.isnull().sum().sum()}")

    # 컬럼 목록
    print("\n컬럼 목록:")
    for col in df.columns:
        print(f"- {col}")

    # 요약 통계
    print("\n기술 통계량:")
    print(df.describe().round(2))

    # 상관관계 분석 (수치형 변수만)
    numeric_cols = df.select_dtypes(include=['number']).columns
    correlation = df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, cmap='coolwarm',
                vmin=-1, vmax=1, annot_kws={"size": 8})
    plt.title('변수 간 상관관계')
    plt.tight_layout()
    plt.savefig('results/correlation_heatmap.png')
    print("\n상관관계 히트맵이 'results/correlation_heatmap.png'에 저장되었습니다.")

    # 종속변수 분포
    plt.figure(figsize=(10, 6))
    sns.histplot(df['염색색차 DE'], kde=True)
    plt.title('염색색차 DE 분포')
    plt.savefig('results/target_distribution.png')
    print("종속변수 분포가 'results/target_distribution.png'에 저장되었습니다.")

    print("\n데이터 검증 완료!")

if __name__ == "__main__":
    validate_preprocessing_results()
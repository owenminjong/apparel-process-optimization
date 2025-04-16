import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

def load_data():
    """
    데이터 파일을 로드합니다.
    """
    try:
        # SET1: LOT 물량 정보
        workload = pd.read_excel('data/LOT 물량.xlsx')
        print("LOT 물량 데이터 로드 성공")

        # SET2: 시계열 설비데이터
        operation = pd.read_csv('data/PRODUCTION_TREND.csv', encoding='CP949')
        print("PRODUCTION_TREND 데이터 로드 성공")

        # SET3: CCM 검사결과
        ccm = pd.read_excel('data/CCM 측정값.xlsx')
        print("CCM 측정값 데이터 로드 성공")

        return workload, operation, ccm

    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return None, None, None

def preprocess_data(workload, operation, ccm):
    """
    데이터 전처리를 수행합니다.
    """
    if workload is None or operation is None or ccm is None:
        return None

    # TODO: 가이드북에 따른 데이터 전처리 코드 구현
    # 1. 데이터 형식 통일
    # 2. 중복데이터 처리
    # 3. 비유일성 데이터 처리
    # 4. 파생변수 생성
    # 5. 데이터셋 결합
    # 6. 이상치 제거
    # 7. 불필요한 열 제거

    print("데이터 전처리 완료")
    return None  # 전처리된 데이터 반환

if __name__ == "__main__":
    workload, operation, ccm = load_data()
    processed_data = preprocess_data(workload, operation, ccm)
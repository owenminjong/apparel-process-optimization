import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_workload_data(file_path='data/LOT 물량.xlsx'):
    """LOT 물량 데이터를 로드합니다."""
    try:
        workload = pd.read_excel(file_path)
        print(f"LOT 물량 데이터 로드 성공: {workload.shape[0]}행, {workload.shape[1]}열")
        return workload
    except Exception as e:
        print(f"LOT 물량 데이터 로드 실패: {e}")
        return None

def load_operation_data(file_path='data/PRODUCTION_TREND.csv'):
    """시계열 설비데이터를 로드합니다."""
    try:
        operation = pd.read_csv(file_path, encoding='CP949')
        print(f"PRODUCTION_TREND 데이터 로드 성공: {operation.shape[0]}행, {operation.shape[1]}열")
        return operation
    except Exception as e:
        print(f"PRODUCTION_TREND 데이터 로드 실패: {e}")
        return None

def load_ccm_data(file_path='data/CCM 측정값.xlsx'):
    """CCM 측정값 데이터를 로드합니다."""
    try:
        ccm = pd.read_excel(file_path)
        print(f"CCM 측정값 데이터 로드 성공: {ccm.shape[0]}행, {ccm.shape[1]}열")
        return ccm
    except Exception as e:
        print(f"CCM 측정값 데이터 로드 실패: {e}")
        return None

def load_all_data():
    """모든 데이터셋을 로드합니다."""
    workload = load_workload_data()
    operation = load_operation_data()
    ccm = load_ccm_data()

    return workload, operation, ccm

if __name__ == "__main__":
    workload, operation, ccm = load_all_data()

    # 각 데이터셋의 기본 정보 출력
    if workload is not None:
        print("\nLOT 물량 데이터 미리보기:")
        print(workload.head(2))

    if operation is not None:
        print("\n시계열 설비데이터 미리보기:")
        print(operation.head(2))

    if ccm is not None:
        print("\nCCM 측정값 데이터 미리보기:")
        print(ccm.head(2))
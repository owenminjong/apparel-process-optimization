import pandas as pd
import numpy as np

def merge_datasets(workload, operation, ccm):
    """전처리된 세 데이터셋을 병합합니다."""
    if workload is None or operation is None or ccm is None:
        print("데이터셋 병합 실패: 하나 이상의 데이터셋이 None입니다.")
        return None

    # LOT 물량 데이터 + CCM 데이터 결합 - 'LOT번호' 기준
    df = ccm.merge(workload, how='inner', on='LOT번호')
    print(f"물량 + CCM 데이터 병합 결과: {df.shape[0]}행, {df.shape[1]}열")

    # 설비데이터(시계열) + 그 외 데이터(배치성) 결합 - 'LOT번호', '공정코드' 기준
    df2 = df.merge(operation, how='inner', on=['LOT번호', '공정코드'])
    print(f"최종 병합 결과: {df2.shape[0]}행, {df2.shape[1]}열")

    # 변수 순서 변경 (독립변수-종속변수 순)
    df2 = df2[['LOT번호', '검사차수', '작업명', '공정코드', '설비번호',
               '단위중량(kg)', '투입중량(kg)', '투입액량(L)', '염색길이(m)', '투입중량/길이', '투입중량/액량',
               '공정진행시간(%)', '목표온도', '지시온도', '진행온도', '포속1', '포속2', '포속3', '포속4',
               '염색색차 DE']]

    return df2

if __name__ == "__main__":
    from data_loader import load_all_data
    from data_cleaner import clean_workload_data, clean_operation_data, clean_ccm_data
    from feature_engineering import create_workload_features, create_operation_features

    # 데이터 로드
    workload, operation, ccm = load_all_data()

    # 데이터 정제
    workload_cleaned = clean_workload_data(workload)
    operation_cleaned = clean_operation_data(operation)
    ccm_cleaned = clean_ccm_data(ccm)

    # 파생변수 생성
    workload_with_features = create_workload_features(workload_cleaned)
    operation_with_features = create_operation_features(operation_cleaned)

    # 데이터셋 병합
    merged_data = merge_datasets(workload_with_features, operation_with_features, ccm_cleaned)

    if merged_data is not None:
        print("\n병합된 데이터 미리보기:")
        print(merged_data.head(2))
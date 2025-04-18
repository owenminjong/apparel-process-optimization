import pandas as pd
import numpy as np

def clean_workload_data(workload):
    """LOT 물량 데이터를 정제합니다."""
    if workload is None:
        return None

    # 필요 변수만 추출 및 변수 순서 변경
    workload = workload[['PRODT_ORDER_NO', 'JOB_CD', 'EXT1_QTY(투입중량 (KG))',
                         'EXT2_QTY (액량 (LITER))', '단위중량', '염색 가동 길이']]

    # 컬럼명 변경
    workload.columns = ['LOT번호', '공정코드', '투입중량(kg)', '투입액량(L)', '단위중량(kg)', '염색길이(m)']

    # 중복데이터 제거
    workload = workload[~workload.duplicated(keep='first')]
    workload.reset_index(drop=True, inplace=True)

    # 비유일성 데이터 처리 - LOT번호 기준으로 유일해야 함
    duplicated_lots = workload[workload.duplicated(subset='LOT번호', keep=False)]
    if len(duplicated_lots) > 0:
        print(f"중복 LOT번호 발견: {len(duplicated_lots)}개")
        # 중복 LOT번호 중 염색길이가 1m인 행 제거 (예시)
        workload = workload[~((workload.duplicated(subset='LOT번호', keep=False)) &
                              (workload['염색길이(m)'] == 1))]
        workload.reset_index(drop=True, inplace=True)

    print("LOT 물량 데이터 정제 완료")
    return workload

def clean_operation_data(operation):
    """시계열 설비데이터를 정제합니다."""
    if operation is None:
        return None

    # 필요한 변수만 추출 및 변수 순서 변경
    try:
        operation = operation[['LOT_NO', 'WC_CD', 'RESOURCE_CD', 'INSRT_DT', 'SEQ_NO', 'CR_TEMP',
                               'TRD_TEMP_SP', 'TRD_TEMP_PV', 'TRD_SPEED1', 'TRD_SPEED2',
                               'TRD_SPEED3', 'TRD_SPEED4']]

        # 변수명 변경
        operation.columns = ['LOT번호', '공정코드', '설비번호', '공정일시', '공정진행시간', '목표온도',
                             '지시온도', '진행온도', '포속1', '포속2', '포속3', '포속4']

        # 데이터 타입 확인 및 변환
        # 숫자형 변수 변환
        numeric_cols = ['공정진행시간', '목표온도', '지시온도', '진행온도', '포속1', '포속2', '포속3', '포속4']
        for col in numeric_cols:
            operation[col] = pd.to_numeric(operation[col], errors='coerce')

        # 문자열 변수 변환
        str_cols = ['LOT번호', '공정코드', '설비번호', '공정일시']
        for col in str_cols:
            operation[col] = operation[col].astype(str)

        # 결측치 처리
        operation = operation.dropna(subset=['LOT번호', '공정코드', '공정진행시간'])

        # 중복데이터 제거 - 수정된 방식
        operation = operation.drop_duplicates(keep='first')
        operation.reset_index(drop=True, inplace=True)

        # 비유일성 데이터 처리 - LOT번호, 공정코드, 공정진행시간 기준으로 유일해야 함
        operation = operation.drop_duplicates(subset=['LOT번호', '공정코드', '공정진행시간'],
                                              keep='last')
        operation.reset_index(drop=True, inplace=True)

        print("시계열 설비데이터 정제 완료")
        return operation

    except Exception as e:
        print(f"시계열 설비데이터 정제 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def clean_ccm_data(ccm):
    """CCM 측정값 데이터를 정제합니다."""
    if ccm is None:
        return None

    # 필요한 변수만 추출
    ccm = ccm[['lot_no', 'seq', 'oper_id', '염색 색차 DE']]

    # 변수명 변경
    ccm.columns = ['LOT번호', '검사차수', '작업명', '염색색차 DE']

    # 첫번째 글자가 소문자로 시작하는 LOT번호 처리
    ccm.loc[:, 'LOT번호'] = ccm['LOT번호'].str.capitalize()

    # 중복데이터 제거
    ccm = ccm[~ccm.duplicated(keep='first')]
    ccm.reset_index(drop=True, inplace=True)

    # LOT번호 기준으로 가장 마지막 검사차수 값만 추출
    ccm = ccm.groupby(['LOT번호']).last().reset_index()

    print("CCM 측정값 데이터 정제 완료")
    return ccm

def remove_outliers(df):
    """이상치를 제거합니다."""
    if df is None:
        return None

    # 단위중량 이상치 처리 (0인 값 제거)
    if '단위중량(kg)' in df.columns:
        df = df[df['단위중량(kg)'] > 0]
        df.reset_index(drop=True, inplace=True)

    # 염색길이 이상치 처리 (1인 값 제거)
    if '염색길이(m)' in df.columns:
        df = df[df['염색길이(m)'] > 1]
        df.reset_index(drop=True, inplace=True)

    print("이상치 제거 완료")
    return df

def remove_unnecessary_columns(df, excluded_columns=None):
    """불필요한 열을 제거합니다."""
    if df is None:
        return None

    if excluded_columns:
        df = df.drop(excluded_columns, axis=1, errors='ignore')
        print(f"불필요한 열 제거 완료: {', '.join(excluded_columns)}")

    return df

if __name__ == "__main__":
    from data_loader import load_all_data

    # 데이터 로드
    workload, operation, ccm = load_all_data()

    # 데이터 정제
    workload_cleaned = clean_workload_data(workload)
    operation_cleaned = clean_operation_data(operation)
    ccm_cleaned = clean_ccm_data(ccm)

    print("\n정제된 데이터 정보:")
    print(f"LOT 물량 데이터: {workload_cleaned.shape[0]}행, {workload_cleaned.shape[1]}열")
    print(f"시계열 설비데이터: {operation_cleaned.shape[0]}행, {operation_cleaned.shape[1]}열")
    print(f"CCM 측정값 데이터: {ccm_cleaned.shape[0]}행, {ccm_cleaned.shape[1]}열")
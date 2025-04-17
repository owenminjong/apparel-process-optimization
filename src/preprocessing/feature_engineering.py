import pandas as pd
import numpy as np

def create_workload_features(workload):
    """LOT 물량 데이터에서 파생변수를 생성합니다."""
    if workload is None:
        return None

    print("LOT 물량 데이터 파생변수 생성 중...")

    # 파생변수 생성
    workload['투입중량/길이'] = workload['투입중량(kg)'] / workload['염색길이(m)']
    workload['투입중량/액량'] = workload['투입중량(kg)'] / workload['투입액량(L)']

    print("LOT 물량 데이터 파생변수 생성 완료")
    return workload

def create_operation_features(operation):
    """시계열 설비데이터에서 공정진행시간(%) 파생변수를 생성합니다."""
    if operation is None:
        return None

    print("공정진행시간(%) 파생변수 생성 중...")

    # 더 안전한 방법: 그룹별로 처리
    result_df = pd.DataFrame()

    # 각 LOT번호 그룹에 대해 처리
    lot_groups = operation.groupby('LOT번호')
    total_groups = len(lot_groups)

    print(f"총 {total_groups}개의 LOT번호 그룹 처리 중...")

    for i, (lot_no, group) in enumerate(lot_groups):
        if i % 100 == 0:  # 100개 그룹마다 진행 상황 출력
            print(f"LOT번호 그룹 처리 중: {i}/{total_groups}")

        # 정렬
        group = group.sort_values('공정진행시간')

        # 최대값 계산
        max_time = group['공정진행시간'].max()

        # 공정진행시간(%) 계산
        if max_time > 0:
            group['공정진행시간(%)'] = (group['공정진행시간'] / max_time * 100).round(2)
        else:
            group['공정진행시간(%)'] = 0

        # 결과에 추가
        result_df = pd.concat([result_df, group])

    # 인덱스 재설정
    result_df = result_df.reset_index(drop=True)

    # 필요한 변수만 선택
    result_df = result_df[['LOT번호', '공정코드', '설비번호', '공정진행시간(%)',
                           '목표온도', '지시온도', '진행온도', '포속1', '포속2', '포속3', '포속4']]

    print("시계열 설비데이터 파생변수 생성 완료")
    return result_df
def create_operation_features_chunked(operation, chunk_size=100000):
    """
    대용량 시계열 설비데이터를 청크 단위로 처리하여 공정진행시간(%) 파생변수를 생성합니다.
    큰 데이터셋에 더 적합한 방법입니다.
    """
    if operation is None:
        return None

    print("공정진행시간(%) 파생변수 생성 중 (청크 처리 방식)...")

    # 각 LOT번호별 최대 공정진행시간 계산
    max_times = operation.groupby('LOT번호')['공정진행시간'].max()

    # 결과를 저장할 빈 데이터프레임
    result_df = pd.DataFrame()

    # 데이터를 청크로 나누어 처리
    total_chunks = len(operation) // chunk_size + (1 if len(operation) % chunk_size != 0 else 0)

    # 진행 상황 출력 (tqdm 대신 간단한 출력 사용)
    for i in range(total_chunks):
        print(f"청크 처리 중: {i+1}/{total_chunks}")

        # 청크 데이터 추출
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(operation))
        chunk = operation.iloc[start_idx:end_idx].copy()

        # 공정진행시간(%) 계산
        for lot_no, group in chunk.groupby('LOT번호'):
            if lot_no in max_times and max_times[lot_no] > 0:
                chunk.loc[group.index, '공정진행시간(%)'] = round(group['공정진행시간'] / max_times[lot_no] * 100, 2)
            else:
                chunk.loc[group.index, '공정진행시간(%)'] = 0

        # 결과에 추가
        result_df = pd.concat([result_df, chunk])

    # 필요한 변수만 선택
    result_df = result_df[['LOT번호', '공정코드', '설비번호', '공정진행시간(%)',
                           '목표온도', '지시온도', '진행온도', '포속1', '포속2', '포속3', '포속4']]

    print("시계열 설비데이터 파생변수 생성 완료")
    return result_df

def select_optimal_method(operation):
    """
    데이터 크기에 따라 최적의 처리 방법을 선택합니다.
    """
    # 메모리 사용량 추정 (단순 근사치)
    estimated_memory_mb = operation.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"추정 메모리 사용량: {estimated_memory_mb:.2f} MB")

    # 메모리 기반 방법 선택
    if estimated_memory_mb < 500:  # 500MB 미만
        print("표준 벡터화 방식으로 처리합니다.")
        return create_operation_features(operation)
    else:
        print("청크 처리 방식으로 처리합니다.")
        return create_operation_features_chunked(operation)

if __name__ == "__main__":
    from data_loader import load_all_data
    from data_cleaner import clean_workload_data, clean_operation_data

    # 데이터 로드
    workload, operation, _ = load_all_data()

    # 데이터 정제
    workload_cleaned = clean_workload_data(workload)
    operation_cleaned = clean_operation_data(operation)

    # 파생변수 생성
    workload_with_features = create_workload_features(workload_cleaned)

    # 데이터 크기에 따라 최적의 방법 선택
    operation_with_features = select_optimal_method(operation_cleaned)

    print("\n파생변수가 추가된 데이터 미리보기:")
    if workload_with_features is not None:
        print("\nLOT 물량 데이터:")
        print(workload_with_features.head(2))

    if operation_with_features is not None:
        print("\n시계열 설비데이터:")
        print(operation_with_features.head(2))
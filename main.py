from src.preprocessing import download_data, data_preprocessor
from src.modeling import model_builder
from src.optimization import genetic_optimizer

def main():
    # 1. 데이터 다운로드
    print("1. 데이터 다운로드 중...")
    download_data.download_data()

    # 2. 데이터 로드 및 전처리
    print("\n2. 데이터 로드 및 전처리 중...")
    workload, operation, ccm = data_preprocessor.load_data()
    processed_data = data_preprocessor.preprocess_data(workload, operation, ccm)

    # TODO: 나머지 코드 구현
    # 3. 모델 구축
    # 4. 최적화 수행
    # 5. 결과 시각화 및 저장

    print("\n프로세스 완료!")

if __name__ == "__main__":
    main()
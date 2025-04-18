#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
의류 염색 공정 최적화 AI 시스템 (DyeOptimAI)
"""

import os
import argparse
import pandas as pd
import time
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dyeoptim.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DyeOptimAI")

# 디렉토리 생성
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def parse_arguments():
    """명령행 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description='의류 염색 공정 최적화 AI 시스템')

    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'download', 'preprocess', 'model', 'optimize', 'validate'],
                        help='실행 모드 (기본값: full)')

    parser.add_argument('--force-download', action='store_true',
                        help='이미 존재하는 데이터 파일도 강제로 다시 다운로드')

    parser.add_argument('--target', type=float, default=1.5,
                        help='목표 염색색차 DE 값 (기본값: 1.5)')

    parser.add_argument('--no-download', action='store_true',
                        help='데이터 다운로드 단계 건너뛰기')

    parser.add_argument('--no-preprocess', action='store_true',
                        help='데이터 전처리 단계 건너뛰기')

    parser.add_argument('--no-model', action='store_true',
                        help='모델링 단계 건너뛰기')

    parser.add_argument('--no-optimize', action='store_true',
                        help='최적화 단계 건너뛰기')

    return parser.parse_args()

def download_data(force=False):
    """데이터를 다운로드합니다."""
    from src.preprocessing.download_data import download_data

    logger.info("데이터 다운로드 중...")
    start_time = time.time()

    download_data(force_download=force)

    elapsed_time = time.time() - start_time
    logger.info(f"데이터 다운로드 완료! (소요 시간: {elapsed_time:.2f}초)")

def preprocess_data():
    """데이터를 전처리합니다."""
    from src.preprocessing.data_preprocessor import preprocess_data

    logger.info("데이터 전처리 중...")
    start_time = time.time()

    preprocessed_data = preprocess_data()

    if preprocessed_data is None:
        logger.error("데이터 전처리 실패!")
        return False

    elapsed_time = time.time() - start_time
    logger.info(f"데이터 전처리 완료! (소요 시간: {elapsed_time:.2f}초)")

    return True

def build_model():
    """모델을 구축합니다."""
    from src.modeling.model_builder import run_modeling_pipeline

    logger.info("모델 구축 중...")
    start_time = time.time()

    model_results = run_modeling_pipeline()

    if model_results is None:
        logger.error("모델 구축 실패!")
        return False

    elapsed_time = time.time() - start_time
    logger.info(f"모델 구축 완료! (소요 시간: {elapsed_time:.2f}초)")
    logger.info(f"모델 성능: Adjusted R² = {model_results['metrics']['adjusted_r2']:.3f}, "
                f"RMSE = {model_results['metrics']['rmse']:.3f}, "
                f"MAE = {model_results['metrics']['mae']:.3f}")

    return True

def optimize_process(target_value=1.5):
    """공정 최적화를 수행합니다."""
    from src.optimization.genetic_optimizer import run_optimization_pipeline

    logger.info(f"공정 최적화 중... (목표값: {target_value})")
    start_time = time.time()

    optimization_result = run_optimization_pipeline(target_value=target_value)

    if optimization_result is None:
        logger.error("공정 최적화 실패!")
        return False

    elapsed_time = time.time() - start_time
    logger.info(f"공정 최적화 완료! (소요 시간: {elapsed_time:.2f}초)")
    logger.info(f"목표값: {target_value:.4f}, 예측값: {optimization_result['predicted_value']:.4f}")

    # 최적화 결과 요약
    logger.info("최적 공정 변수:")
    for name, value in optimization_result['optimal_parameters'].items():
        logger.info(f"  {name}: {value:.4f}")

    return True

def visualize_results():
    """결과를 시각화합니다."""
    logger.info("결과 시각화 중...")

    # 모델링 결과 확인
    prediction_plot = Path('results/prediction_vs_actual.png')
    feature_importance_plot = Path('results/shap_feature_importance.png')

    # 최적화 결과 확인
    convergence_plot = Path('results/optimization_convergence.png')

    # 결과 요약 보고서 생성
    generate_summary_report()

    logger.info("결과 시각화 완료!")
    logger.info(f"모든 결과는 'results/' 폴더에서 확인할 수 있습니다.")

def generate_summary_report():
    """결과 요약 보고서를 생성합니다."""
    from datetime import datetime
    import json
    import os

    try:
        # 결과 디렉토리 확인 및 생성
        if not os.path.exists('results'):
            os.makedirs('results')

        # optimization_result.json 파일 존재 확인
        json_path = 'results/optimization_result.json'
        if not os.path.exists(json_path):
            logger.warning("최적화 결과 파일이 없습니다. 기본 보고서를 생성합니다.")

            # 기본 보고서 생성
            report = [
                "# 의류 염색 공정 최적화 AI 시스템 - 결과 요약 보고서",
                f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## 알림",
                "최적화 과정이 완료되지 않았거나 결과를 저장하지 못했습니다.",
                "파이프라인을 다시 실행하여 최적화를 진행해주세요."
            ]

            # 보고서 저장
            with open('results/summary_report.md', 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))

            logger.info("기본 요약 보고서가 'results/summary_report.md'에 저장되었습니다.")
            return True

        # 파일 로드
        with open(json_path, 'r') as f:
            optimization_result = json.load(f)

        # 오류 확인
        if 'error' in optimization_result:
            logger.warning(f"최적화 중 오류 발생: {optimization_result['error']}")
            report = [
                "# 의류 염색 공정 최적화 AI 시스템 - 결과 요약 보고서",
                f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## 오류 발생",
                f"최적화 중 오류가 발생했습니다: {optimization_result['error']}",
                "",
                "## 목표값",
                f"목표 염색색차 DE: {optimization_result['target_value']:.4f}"
            ]

            # 추가 정보가 있으면 추가
            if 'optimal_parameters' in optimization_result and isinstance(optimization_result['optimal_parameters'], dict):
                report.append("")
                report.append("## 임시 공정 변수")
                for name, value in optimization_result['optimal_parameters'].items():
                    if isinstance(value, (int, float)):
                        report.append(f"- {name}: {value:.4f}")
                    else:
                        report.append(f"- {name}: {value}")
        else:
            # 정상 요약 보고서 생성
            report = [
                "# 의류 염색 공정 최적화 AI 시스템 - 결과 요약 보고서",
                f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## 최적화 결과",
                f"목표 염색색차 DE: {optimization_result['target_value']:.4f}",
                f"예측 염색색차 DE: {optimization_result['predicted_value']:.4f}",
                "",
                "## 최적 공정 변수"
            ]

            # 최적 공정 변수 추가
            for name, value in optimization_result['optimal_parameters'].items():
                report.append(f"- {name}: {value:.4f}")

        # 보고서 저장
        with open('results/summary_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        logger.info("요약 보고서가 'results/summary_report.md'에 저장되었습니다.")
        return True
    except Exception as e:
        logger.error(f"요약 보고서 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
def full_pipeline(args):
    """전체 파이프라인을 실행합니다."""
    logger.info("=== 의류 염색 공정 최적화 AI 시스템 실행 ===")
    start_time = time.time()

    # 1. 데이터 다운로드
    if not args.no_download:
        download_data(args.force_download)
    else:
        logger.info("데이터 다운로드 단계 건너뜀")

    # 2. 데이터 전처리
    if not args.no_preprocess:
        if not preprocess_data():
            return
    else:
        logger.info("데이터 전처리 단계 건너뜀")

    # 3. 모델 구축
    if not args.no_model:
        if not build_model():
            return
    else:
        logger.info("모델 구축 단계 건너뜀")

    # 4. 공정 최적화
    if not args.no_optimize:
        if not optimize_process(args.target):
            return
    else:
        logger.info("공정 최적화 단계 건너뜀")

    # 5. 결과 시각화
    visualize_results()

    elapsed_time = time.time() - start_time
    logger.info(f"=== 전체 파이프라인 완료! (총 소요 시간: {elapsed_time:.2f}초) ===")

def main():
    """메인 함수"""
    args = parse_arguments()

    if args.mode == 'full':
        full_pipeline(args)
    elif args.mode == 'download':
        download_data(args.force_download)
    elif args.mode == 'preprocess':
        preprocess_data()
    elif args.mode == 'model':
        build_model()
    elif args.mode == 'optimize':
        optimize_process(args.target)
    elif args.mode == 'validate':
        from src.preprocessing.validate_results import validate_preprocessing_results
        validate_preprocessing_results()

if __name__ == "__main__":
    main()
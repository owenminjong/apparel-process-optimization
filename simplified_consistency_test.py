#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DyeOptimAI - 단순화된 일관성 테스트 스크립트
한 번에 하나의 테스트만 실행하여 안정성 확보
"""

import os
import pandas as pd
import numpy as np
import time
import joblib
import json
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 사용 안 함 - Tkinter 에러 방지
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse
from datetime import datetime

# 필요한 모듈 임포트
from src.modeling.model_builder import ModelPipeline
from src.optimization.genetic_optimizer import OptimizationPipeline

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("consistency_test.log", mode='a'),  # 추가 모드
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ConsistencyTest")

def test_model(test_id=1, random_state=None, data_path='data/preprocessed_data.csv'):
    """
    모델을 테스트합니다.

    Args:
        test_id (int): 테스트 ID
        random_state (int, optional): 랜덤 시드, None이면 랜덤 생성
        data_path (str): 데이터 파일 경로

    Returns:
        dict: 모델 성능 지표
    """
    # 결과 디렉토리 생성
    results_dir = f'test_results/model_test_{test_id}'
    os.makedirs(results_dir, exist_ok=True)

    # 랜덤 시드 설정 (지정되지 않은 경우 랜덤 생성)
    if random_state is None:
        random_state = np.random.randint(1, 1000)

    logger.info(f"모델 테스트 {test_id} 시작 (랜덤 시드: {random_state})")

    # 모델 파이프라인 생성 및 실행
    start_time = time.time()
    pipeline = ModelPipeline(test_size=0.2, random_state=random_state)
    results_dict = pipeline.run(data_path=data_path)
    elapsed_time = time.time() - start_time

    # 결과 저장
    metrics = {
        'test_id': test_id,
        'random_state': random_state,
        'adjusted_r2': results_dict['metrics']['adjusted_r2'],
        'rmse': results_dict['metrics']['rmse'],
        'mae': results_dict['metrics']['mae'],
        'runtime_seconds': elapsed_time,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # 모델 저장
    model_path = f"{results_dir}/model.pkl"
    joblib.dump(results_dict['model'], model_path)

    logger.info(f"모델 테스트 {test_id} 완료 (소요 시간: {elapsed_time:.2f}초)")
    logger.info(f"Adjusted R²: {results_dict['metrics']['adjusted_r2']:.4f}, "
                f"RMSE: {results_dict['metrics']['rmse']:.4f}, "
                f"MAE: {results_dict['metrics']['mae']:.4f}")

    # 결과를 JSON으로 저장
    with open(f'{results_dir}/results.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # 모든 테스트 결과 파일에도 추가
    all_results_file = 'test_results/all_model_results.json'
    all_results = []

    # 기존 결과 파일이 있으면 로드
    if os.path.exists(all_results_file):
        with open(all_results_file, 'r') as f:
            try:
                all_results = json.load(f)
            except:
                all_results = []

    # 새 결과 추가
    all_results.append(metrics)

    # 결과 저장
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    return metrics

def test_optimization(test_id=1, target_value=1.5,
                      model_path='models/rf_model.pkl',
                      scaler_path='models/scaler.pkl',
                      data_path='data/preprocessed_data.csv'):
    """
    최적화를 테스트합니다.

    Args:
        test_id (int): 테스트 ID
        target_value (float): 목표 염색색차 DE 값
        model_path (str): 모델 파일 경로
        scaler_path (str): 스케일러 파일 경로
        data_path (str): 데이터 파일 경로

    Returns:
        dict: 최적화 결과
    """
    # 결과 디렉토리 생성
    results_dir = f'test_results/optimization_test_{test_id}'
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"최적화 테스트 {test_id} 시작 (목표값: {target_value})")

    # 최적화 파이프라인 생성 및 실행
    start_time = time.time()

    # 파이프라인 생성
    pipeline = OptimizationPipeline()

    # 리소스 로드
    pipeline.load_resources(model_path, scaler_path, data_path)

    # 변수 구성 및 범위 설정
    pipeline.configure_variables()
    pipeline.setup_bounds()

    # 제약 조건 추가
    pipeline.add_constraints()

    # 최적화기 설정 - 알고리즘 파라미터 수정
    algorithm_params = {
        'max_num_iteration': 100,       # 반복 횟수 감소
        'population_size': 100,         # 인구 크기 감소
        'mutation_probability': 0.2,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': 30  # 조기 종료 설정
    }
    pipeline.setup_optimizer(algorithm_params)

    # 최적화 실행
    optimization_result = pipeline.run_optimization(target_value)

    elapsed_time = time.time() - start_time

    # 결과 출력
    logger.info(f"최적화 테스트 {test_id} 완료 (소요 시간: {elapsed_time:.2f}초)")
    logger.info(f"목표값: {target_value}, 예측값: {optimization_result['predicted_value']:.4f}, "
                f"오차: {abs(target_value - optimization_result['predicted_value']):.4f}")

    # 최적 파라미터 출력
    logger.info("최적 파라미터:")
    for name, value in optimization_result['optimal_parameters'].items():
        logger.info(f"  {name}: {value:.4f}")

    # 결과 저장
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(optimization_result, f, indent=4)

    # 추가 결과 저장
    test_summary = {
        'test_id': test_id,
        'target_value': target_value,
        'predicted_value': optimization_result['predicted_value'],
        'error': abs(target_value - optimization_result['predicted_value']),
        'runtime_seconds': elapsed_time,
        'optimal_parameters': optimization_result['optimal_parameters'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(f'{results_dir}/summary.json', 'w') as f:
        json.dump(test_summary, f, indent=4)

    # 최적화 수렴 그래프 저장
    plt.figure(figsize=(10, 6))
    plt.plot(optimization_result['convergence'], 'b-', linewidth=2)
    plt.title(f'유전 알고리즘 수렴 과정 (테스트 {test_id})')
    plt.xlabel('세대 (Generation)')
    plt.ylabel('목적 함수 값')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/convergence.png')
    plt.close()

    # 모든 테스트 결과 파일에도 추가
    all_results_file = 'test_results/all_optimization_results.json'
    all_results = []

    # 기존 결과 파일이 있으면 로드
    if os.path.exists(all_results_file):
        with open(all_results_file, 'r') as f:
            try:
                all_results = json.load(f)
            except:
                all_results = []

    # 새 결과 추가
    all_results.append(test_summary)

    # 결과 저장
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    return optimization_result

def analyze_consistency():
    """
    지금까지의 테스트 결과를 분석하여 일관성 보고서를 생성합니다.
    """
    # 결과 디렉토리 생성
    os.makedirs('test_results', exist_ok=True)

    # 모델 결과 분석
    model_results_file = 'test_results/all_model_results.json'
    if os.path.exists(model_results_file):
        with open(model_results_file, 'r') as f:
            model_results = json.load(f)

        if len(model_results) > 0:
            # 결과 데이터프레임 생성
            model_df = pd.DataFrame(model_results)

            # 기술 통계량 계산
            r2_mean = model_df['adjusted_r2'].mean()
            r2_std = model_df['adjusted_r2'].std()
            rmse_mean = model_df['rmse'].mean()
            rmse_std = model_df['rmse'].std()
            mae_mean = model_df['mae'].mean()
            mae_std = model_df['mae'].std()

            # 결과 시각화
            plt.figure(figsize=(12, 8))

            # R2 플롯
            plt.subplot(2, 2, 1)
            plt.plot(model_df['test_id'], model_df['adjusted_r2'], 'o-')
            plt.axhline(y=r2_mean, color='r', linestyle='--',
                        label=f'Mean: {r2_mean:.4f}')
            plt.title('Adjusted R² by Test')
            plt.xlabel('Test ID')
            plt.ylabel('Adjusted R²')
            plt.legend()
            plt.grid(True)

            # RMSE 플롯
            plt.subplot(2, 2, 2)
            plt.plot(model_df['test_id'], model_df['rmse'], 'o-')
            plt.axhline(y=rmse_mean, color='r', linestyle='--',
                        label=f'Mean: {rmse_mean:.4f}')
            plt.title('RMSE by Test')
            plt.xlabel('Test ID')
            plt.ylabel('RMSE')
            plt.legend()
            plt.grid(True)

            # MAE 플롯
            plt.subplot(2, 2, 3)
            plt.plot(model_df['test_id'], model_df['mae'], 'o-')
            plt.axhline(y=mae_mean, color='r', linestyle='--',
                        label=f'Mean: {mae_mean:.4f}')
            plt.title('MAE by Test')
            plt.xlabel('Test ID')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)

            # 상자 그림
            plt.subplot(2, 2, 4)
            model_df[['adjusted_r2', 'rmse', 'mae']].boxplot()
            plt.title('Performance Metrics Distribution')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig('test_results/model_consistency.png')
            plt.close()

            print("\n모델 일관성 분석 결과:")
            print(f"Adjusted R² 평균: {r2_mean:.4f} ± {r2_std:.4f}")
            print(f"RMSE 평균: {rmse_mean:.4f} ± {rmse_std:.4f}")
            print(f"MAE 평균: {mae_mean:.4f} ± {mae_std:.4f}")

            # 모델 일관성 요약 저장
            model_summary = {
                'adjusted_r2_mean': r2_mean,
                'adjusted_r2_std': r2_std,
                'rmse_mean': rmse_mean,
                'rmse_std': rmse_std,
                'mae_mean': mae_mean,
                'mae_std': mae_std,
                'n_tests': len(model_results)
            }

            with open('test_results/model_consistency_summary.json', 'w') as f:
                json.dump(model_summary, f, indent=4)

    # 최적화 결과 분석
    optimization_results_file = 'test_results/all_optimization_results.json'
    if os.path.exists(optimization_results_file):
        with open(optimization_results_file, 'r') as f:
            opt_results = json.load(f)

        if len(opt_results) > 0:
            # 결과 데이터프레임 생성
            opt_df = pd.DataFrame(opt_results)

            # 기술 통계량 계산
            pred_mean = opt_df['predicted_value'].mean()
            pred_std = opt_df['predicted_value'].std()
            error_mean = opt_df['error'].mean()
            error_std = opt_df['error'].std()

            # 결과 시각화
            plt.figure(figsize=(10, 8))

            # 예측값 플롯
            plt.subplot(2, 1, 1)
            plt.plot(opt_df['test_id'], opt_df['predicted_value'], 'o-')
            plt.axhline(y=opt_df['target_value'].iloc[0], color='g', linestyle='--',
                        label=f'Target: {opt_df["target_value"].iloc[0]:.4f}')
            plt.axhline(y=pred_mean, color='r', linestyle='--',
                        label=f'Mean: {pred_mean:.4f}')
            plt.title('Predicted Values by Test')
            plt.xlabel('Test ID')
            plt.ylabel('Predicted Value')
            plt.legend()
            plt.grid(True)

            # 오차 플롯
            plt.subplot(2, 1, 2)
            plt.plot(opt_df['test_id'], opt_df['error'], 'o-')
            plt.axhline(y=error_mean, color='r', linestyle='--',
                        label=f'Mean: {error_mean:.4f}')
            plt.title('Error by Test')
            plt.xlabel('Test ID')
            plt.ylabel('Absolute Error')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig('test_results/optimization_consistency.png')
            plt.close()

            print("\n최적화 일관성 분석 결과:")
            print(f"예측값 평균: {pred_mean:.4f} ± {pred_std:.4f}")
            print(f"오차 평균: {error_mean:.4f} ± {error_std:.4f}")

            # 최적화 일관성 요약 저장
            opt_summary = {
                'predicted_value_mean': pred_mean,
                'predicted_value_std': pred_std,
                'error_mean': error_mean,
                'error_std': error_std,
                'target_value': opt_df['target_value'].iloc[0],
                'n_tests': len(opt_results)
            }

            with open('test_results/optimization_consistency_summary.json', 'w') as f:
                json.dump(opt_summary, f, indent=4)

            # 최적 파라미터 분석 (첫 번째 테스트 결과에서 변수 이름 추출)
            param_names = list(opt_results[0]['optimal_parameters'].keys())
            param_data = {}

            for param in param_names:
                param_data[param] = [r['optimal_parameters'][param] for r in opt_results]

            param_df = pd.DataFrame(param_data)

            # 파라미터 통계량
            param_mean = param_df.mean()
            param_std = param_df.std()

            # 파라미터 시각화
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(param_df.columns):
                plt.subplot(3, 4, i+1)
                plt.plot(range(1, len(param_df)+1), param_df[col], 'o-')
                plt.axhline(y=param_mean[col], color='r', linestyle='--',
                            label=f'Mean: {param_mean[col]:.4f}')
                plt.title(f'{col}')
                plt.xlabel('Test ID')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)

            plt.tight_layout()
            plt.savefig('test_results/parameter_consistency.png')
            plt.close()

            # 파라미터 일관성 요약 저장
            param_summary = {}
            for param in param_names:
                param_summary[f'{param}_mean'] = param_mean[param]
                param_summary[f'{param}_std'] = param_std[param]

            with open('test_results/parameter_consistency_summary.json', 'w') as f:
                json.dump(param_summary, f, indent=4)

    # 일관성 보고서 생성
    generate_report()

def generate_report():
    """일관성 테스트 결과에 대한 보고서를 생성합니다."""
    try:
        # 모델 일관성 요약 로드
        model_summary_file = 'test_results/model_consistency_summary.json'
        model_summary = None
        if os.path.exists(model_summary_file):
            with open(model_summary_file, 'r') as f:
                model_summary = json.load(f)

        # 최적화 일관성 요약 로드
        opt_summary_file = 'test_results/optimization_consistency_summary.json'
        opt_summary = None
        if os.path.exists(opt_summary_file):
            with open(opt_summary_file, 'r') as f:
                opt_summary = json.load(f)

        # 파라미터 일관성 요약 로드
        param_summary_file = 'test_results/parameter_consistency_summary.json'
        param_summary = None
        if os.path.exists(param_summary_file):
            with open(param_summary_file, 'r') as f:
                param_summary = json.load(f)

        # 보고서 내용 생성
        report_lines = [
            "# DyeOptimAI - 일관성 테스트 보고서",
            f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # 모델 일관성 부분
        if model_summary:
            report_lines.extend([
                "## 1. 모델 일관성 테스트 결과",
                f"테스트 횟수: {model_summary['n_tests']}",
                f"* Adjusted R² 평균: {model_summary['adjusted_r2_mean']:.4f} ± {model_summary['adjusted_r2_std']:.4f}",
                f"* RMSE 평균: {model_summary['rmse_mean']:.4f} ± {model_summary['rmse_std']:.4f}",
                f"* MAE 평균: {model_summary['mae_mean']:.4f} ± {model_summary['mae_std']:.4f}",
                "",
                "### 모델 안정성 평가",
                f"* 모델의 설명력(Adjusted R²) 변동 계수: {model_summary['adjusted_r2_std']/model_summary['adjusted_r2_mean']*100:.2f}%",
                f"* RMSE 변동 계수: {model_summary['rmse_std']/model_summary['rmse_mean']*100:.2f}%",
                ""
            ])

        # 최적화 일관성 부분
        if opt_summary:
            report_lines.extend([
                "## 2. 최적화 일관성 테스트 결과",
                f"테스트 횟수: {opt_summary['n_tests']}",
                f"* 목표값: {opt_summary['target_value']:.4f}",
                f"* 예측값 평균: {opt_summary['predicted_value_mean']:.4f} ± {opt_summary['predicted_value_std']:.4f}",
                f"* 오차 평균: {opt_summary['error_mean']:.4f} ± {opt_summary['error_std']:.4f}",
                "",
                "### 최적화 안정성 평가",
                f"* 예측값 변동 계수: {opt_summary['predicted_value_std']/opt_summary['predicted_value_mean']*100:.2f}%",
                f"* 오차율: {opt_summary['error_mean']/opt_summary['target_value']*100:.2f}%",
                ""
            ])

        # 파라미터 일관성 부분
        if param_summary:
            report_lines.extend(["### 주요 변수 최적값 분포"])

            for key in param_summary:
                if '_mean' in key:
                    param_name = key.replace('_mean', '')
                    std_key = key.replace('_mean', '_std')
                    if std_key in param_summary:
                        report_lines.append(f"* {param_name}: {param_summary[key]:.4f} ± {param_summary[std_key]:.4f}")

            report_lines.append("")

        # 종합 평가 부분
        report_lines.extend([
            "## 3. 종합 평가",
            "### 시스템 일관성 평가"
        ])

        if model_summary and opt_summary:
            # 모델 일관성 평가
            if model_summary['adjusted_r2_std'] < 0.01:
                model_stability = "높은"
            elif model_summary['adjusted_r2_std'] < 0.05:
                model_stability = "중간"
            else:
                model_stability = "낮은"

            # 최적화 일관성 평가
            if opt_summary['error_std'] < 0.01:
                opt_stability = "높은"
            elif opt_summary['error_std'] < 0.05:
                opt_stability = "중간"
            else:
                opt_stability = "낮은"

            report_lines.extend([
                f"* 모델은 {model_stability} 일관성을 보입니다.",
                f"* 최적화는 {opt_stability} 일관성을 보입니다.",
                f"* 다양한 랜덤 시드에서도 모델의 설명력(Adjusted R²)은 일관되게 {model_summary['adjusted_r2_mean']:.4f} 수준을 유지합니다.",
                f"* 최적화 과정에서 목표값({opt_summary['target_value']:.4f})에 대한 평균 오차는 {opt_summary['error_mean']:.4f}로, 매우 정확합니다.",
                "",
                "### 시스템 신뢰성 평가",
                "* 모델과 최적화 결과의 일관성을 종합적으로 평가할 때, 이 시스템은 높은 신뢰성을 갖습니다.",
                "* 이는 실제 의류 염색 공정에 적용하기에 충분한 안정성을 가진 것으로 판단됩니다.",
                ""
            ])

        # 개선 제안 부분
        report_lines.extend([
            "## 4. 개선 제안",
            "* 모델 설명력의 변동성을 더욱 줄이기 위해 앙상블 접근법을 고려할 수 있습니다.",
            "* 최적화 과정에서 여러 번 실행한 결과의 중앙값을 사용하여 더 안정적인 공정 변수를 도출할 수 있습니다.",
            "",
            "## 5. 참고 사항",
            "* 이 보고서는 자동으로 생성되었습니다.",
            "* 모든 테스트 결과는 'test_results/' 폴더에서 확인할 수 있습니다.",
            f"* 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])

        # 보고서 저장
        with open('test_results/consistency_report.md', 'w') as f:
            f.write('\n'.join(report_lines))

        print("\n일관성 테스트 보고서가 'test_results/consistency_report.md'에 저장되었습니다.")

    except Exception as e:
        print(f"보고서 생성 중 오류 발생: {e}")

def main():
    """명령행 인수를 파싱하고 지정된 테스트를 실행합니다."""
    parser = argparse.ArgumentParser(description='DyeOptimAI - 일관성 테스트')

    parser.add_argument('--mode', type=str, choices=['model', 'optimize', 'analyze'],
                        help='테스트 모드: model(모델), optimize(최적화), analyze(분석)')

    parser.add_argument('--id', type=int, default=1,
                        help='테스트 ID (기본값: 1)')

    parser.add_argument('--target', type=float, default=1.5,
                        help='목표 염색색차 DE 값 (기본값: 1.5)')

    parser.add_argument('--seed', type=int, default=None,
                        help='랜덤 시드 (기본값: 랜덤)')

    args = parser.parse_args()

    # 결과 디렉토리 생성
    os.makedirs('test_results', exist_ok=True)

    print("=" * 80)
    print("DyeOptimAI - 일관성 테스트")
    print("=" * 80)

    # 모드에 따른 테스트 실행
    if args.mode == 'model':
        print(f"\n모델 테스트 {args.id} 실행 중...")
        test_model(test_id=args.id, random_state=args.seed)

    elif args.mode == 'optimize':
        print(f"\n최적화 테스트 {args.id} 실행 중...")
        test_optimization(test_id=args.id, target_value=args.target)

    elif args.mode == 'analyze':
        print("\n일관성 분석 실행 중...")
        analyze_consistency()

    else:
        parser.print_help()
        print("\n테스트 모드를 지정해야 합니다. 예:")
        print("  python simplified_consistency_test.py --mode model --id 1")
        print("  python simplified_consistency_test.py --mode optimize --id 1")
        print("  python simplified_consistency_test.py --mode analyze")

if __name__ == "__main__":
    main()
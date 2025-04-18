# summary_generator.py
from datetime import datetime
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SummaryGenerator")

try:
    # 결과 디렉토리 확인 및 생성
    if not os.path.exists('results'):
        os.makedirs('results')
        logger.info("'results' 디렉토리 생성됨")

    # optimization_result.json 파일 존재 확인
    json_path = 'results/optimization_result.json'
    if not os.path.exists(json_path):
        logger.error(f"최적화 결과 파일({json_path})이 존재하지 않습니다.")
        exit(1)

    logger.info(f"'{json_path}' 파일 존재함, 로드 중...")

    # 파일 로드
    with open(json_path, 'r') as f:
        optimization_result = json.load(f)

    logger.info("JSON 파일 로드 완료")

    # 요약 보고서 생성
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
except Exception as e:
    logger.error(f"요약 보고서 생성 실패: {e}")
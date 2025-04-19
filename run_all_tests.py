#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DyeOptimAI - 일괄 일관성 테스트 스크립트
모든 테스트를 순차적으로 실행하고 결과를 분석합니다.
"""

import os
import subprocess
import time
import sys

def run_command(command):
    """
    명령어를 실행하고 결과를 반환합니다.

    Args:
        command (str): 실행할 명령어

    Returns:
        bool: 성공 여부
    """
    print(f"\n실행 중: {command}")
    print("=" * 80)

    try:
        # 명령어 실행
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # 실시간으로 출력 표시
        for line in process.stdout:
            print(line, end='')

        # 프로세스 종료 대기
        process.wait()

        # 종료 코드 확인
        if process.returncode == 0:
            print("\n명령어 실행 성공!")
            return True
        else:
            print(f"\n명령어 실행 실패! 종료 코드: {process.returncode}")
            return False

    except Exception as e:
        print(f"\n명령어 실행 중 오류 발생: {e}")
        return False

def main():
    """
    모든 테스트를 순차적으로 실행합니다.
    """
    print("=" * 80)
    print("DyeOptimAI - 일괄 일관성 테스트")
    print("=" * 80)

    start_time = time.time()

    # 테스트 횟수 설정
    num_tests = 3

    # 결과 디렉토리 생성
    os.makedirs('test_results', exist_ok=True)

    # 1. 모델 테스트 실행
    model_success = 0
    for i in range(1, num_tests + 1):
        print(f"\n[{i}/{num_tests}] 모델 테스트 실행")
        if run_command(f"python simplified_consistency_test.py --mode model --id {i}"):
            model_success += 1
        else:
            print(f"모델 테스트 {i} 실패! 다음 테스트로 진행합니다.")

    print(f"\n모델 테스트 결과: {model_success}/{num_tests} 성공")

    # 2. 최적화 테스트 실행
    opt_success = 0
    for i in range(1, num_tests + 1):
        print(f"\n[{i}/{num_tests}] 최적화 테스트 실행")
        if run_command(f"python simplified_consistency_test.py --mode optimize --id {i}"):
            opt_success += 1
        else:
            print(f"최적화 테스트 {i} 실패! 다음 테스트로 진행합니다.")

    print(f"\n최적화 테스트 결과: {opt_success}/{num_tests} 성공")

    # 3. 결과 분석
    if model_success > 0 or opt_success > 0:
        print("\n일관성 분석 실행")
        run_command("python simplified_consistency_test.py --mode analyze")
    else:
        print("\n모든 테스트가 실패하여 분석을 건너뜁니다.")

    elapsed_time = time.time() - start_time
    print(f"\n모든 테스트 완료! (총 소요 시간: {elapsed_time:.2f}초)")
    print(f"모델 테스트: {model_success}/{num_tests} 성공")
    print(f"최적화 테스트: {opt_success}/{num_tests} 성공")
    print("결과는 'test_results/' 폴더에서 확인할 수 있습니다.")

if __name__ == "__main__":
    main()
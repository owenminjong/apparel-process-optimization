# src/optimization/__init__.py
# 의류 염색 공정 최적화 AI 시스템 - 최적화 모듈

from . import genetic_optimizer

# 모듈 편의 함수 노출
from .genetic_optimizer import run_optimization_pipeline
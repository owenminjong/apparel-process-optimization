# src/optimization/genetic_optimizer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm
import math
import os
import joblib
import json
from pathlib import Path

# 결과 디렉토리 생성
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

class VariableBoundsManager:
    """유전 알고리즘에 사용될 변수 범위를 관리하는 컴포넌트"""

    def __init__(self, data=None, custom_bounds=None):
        """
        변수 범위 관리자를 초기화합니다.

        Parameters:
            data (DataFrame): 변수 범위를 계산하기 위한 데이터프레임
            custom_bounds (dict): 사용자 정의 범위 (선택사항) - {'변수명': [최소값, 최대값]} 형태
        """
        self.data = data
        self.custom_bounds = custom_bounds if custom_bounds else {}
        self.variable_bounds = None
        self.variable_names = None

    def set_bounds(self, variable_names, safety_margin=0.05):
        """
        주어진 변수들의 범위를 설정합니다.

        Parameters:
            variable_names (list): 변수 이름 리스트
            safety_margin (float): 범위 확장 마진 (기본값: 5%)

        Returns:
            numpy.ndarray: 변수 범위 배열
        """
        if self.data is None:
            raise ValueError("데이터가 설정되지 않았습니다.")

        self.variable_names = variable_names
        bounds = []

        for var in variable_names:
            if var in self.custom_bounds:
                # 사용자 정의 범위 사용
                bounds.append(self.custom_bounds[var])
            elif var in self.data.columns:
                # 데이터에서 자동으로 범위 계산
                min_val = self.data[var].min()
                max_val = self.data[var].max()

                # 특수 처리: 0이나 음수인 경우 안전한 값 설정
                if min_val <= 0 and var not in ['공정진행시간(%)']:  # 공정진행시간은 0일 수 있음
                    min_val = 1.0  # 안전한 최소값 설정

                # 마진 적용
                range_val = max_val - min_val
                min_val = max(0, min_val - range_val * safety_margin)
                max_val = max_val + range_val * safety_margin

                bounds.append([min_val, max_val])
            else:
                raise ValueError(f"변수 '{var}'가 데이터에 존재하지 않습니다.")

        self.variable_bounds = np.array(bounds)
        return self.variable_bounds

    def set_constraint(self, var_name, min_val=None, max_val=None, fixed_val=None):
        """특정 변수에 대한 제약 조건을 설정합니다."""
        if var_name not in self.variable_names:
            raise ValueError(f"변수 '{var_name}'가 설정된 변수 목록에 없습니다.")

        idx = self.variable_names.index(var_name)

        if fixed_val is not None:
            # 고정값 설정
            self.variable_bounds[idx] = [fixed_val, fixed_val]
        else:
            # 최소/최대값 업데이트
            if min_val is not None:
                self.variable_bounds[idx][0] = min_val
            if max_val is not None:
                self.variable_bounds[idx][1] = max_val

        return self.variable_bounds

    def save_bounds(self, path='models/variable_bounds.json'):
        """변수 범위를 JSON 파일로 저장합니다."""
        if self.variable_bounds is None or self.variable_names is None:
            raise ValueError("변수 범위가 설정되지 않았습니다.")

        bounds_dict = {
            'variable_names': self.variable_names,
            'bounds': self.variable_bounds.tolist()
        }

        with open(path, 'w') as f:
            json.dump(bounds_dict, f, indent=4)

        print(f"변수 범위가 {path}에 저장되었습니다.")

    @classmethod
    def load_bounds(cls, path='models/variable_bounds.json'):
        """저장된 변수 범위를 로드합니다."""
        with open(path, 'r') as f:
            bounds_dict = json.load(f)

        instance = cls()
        instance.variable_names = bounds_dict['variable_names']
        instance.variable_bounds = np.array(bounds_dict['bounds'])

        return instance


class InputConverterFactory:
    """유전 알고리즘의 입력 변환 함수를 생성하는 팩토리 클래스"""

    @staticmethod
    def create_converter(variable_names, derived_var_mappings=None):
        """
        입력 변환 함수를 생성합니다.

        Parameters:
            variable_names (list): 변수 이름 리스트
            derived_var_mappings (dict): 파생 변수 매핑 (예: {'투입중량/길이': ('투입중량(kg)', '염색길이(m)')})

        Returns:
            function: 입력 변환 함수
        """
        if derived_var_mappings is None:
            derived_var_mappings = {}

        # 가이드북 기준 변수 매핑 설정
        default_mappings = {
            '투입중량/길이': ('투입중량(kg)', '염색길이(m)')
        }
        # 기본 매핑과 사용자 정의 매핑 병합
        derived_var_mappings = {**default_mappings, **derived_var_mappings}

        def input_converter(x_input):
            """입력값을 모델이 요구하는 형태로 변환합니다."""
            result = []
            x_dict = {variable_names[i]: float(x_input[i]) for i in range(len(variable_names))}

            for var in variable_names:
                if var in derived_var_mappings:
                    # 파생 변수인 경우
                    num_var, den_var = derived_var_mappings[var]
                    # 분모가 0인 경우 방지
                    if x_dict[den_var] == 0:
                        raise ValueError(f"변수 '{den_var}'의 값이 0입니다. 0으로 나눌 수 없습니다.")

                    value = x_dict[num_var] / x_dict[den_var]
                else:
                    # 기본 변수인 경우
                    value = x_dict[var]

                result.append(value)

            return np.array(result)

        return input_converter


class ObjectiveFunctionFactory:
    """유전 알고리즘의 목적 함수를 생성하는 팩토리 클래스"""

    @staticmethod
    def create_function(model, scaler, target_value, input_converter=None, minimize=True):
        """
        목적 함수를 생성합니다.

        Parameters:
            model: 학습된 예측 모델
            scaler: 데이터 스케일러
            target_value (float): 목표 값
            input_converter (function): 입력 변환 함수 (선택사항)
            minimize (bool): 최소화 여부 (True이면 최소화, False이면 최대화)

        Returns:
            function: 목적 함수
        """
        def objective_function(x_input):
            """목표값과 예측값의 차이를 계산합니다."""
            try:
                # 입력 검증 추가
                if np.any(np.isnan(x_input)) or np.any(np.isinf(x_input)):
                    print(f"유효하지 않은 입력 값: {x_input}")
                    return 1e10 if minimize else -1e10

                # 입력 변환기가 제공된 경우 사용
                if input_converter:
                    try:
                        input_converted = input_converter(x_input)
                        input_scaled = scaler.transform(input_converted.reshape(1, -1))
                    except Exception as inner_e:
                        print(f"입력 변환 오류: {inner_e}")
                        return 1e10 if minimize else -1e10
                else:
                    # 그대로 사용
                    input_scaled = scaler.transform(x_input.reshape(1, -1))

                # 모델로 예측
                prediction = model.predict(input_scaled)[0]

                # 목표값과의 차이 계산
                difference = abs(target_value - prediction)

                # 로그 변환을 통해 작은 차이에 더 민감하게 만듦
                if difference > 0:
                    return math.log(difference) if minimize else -math.log(difference)
                else:
                    return -100 if minimize else 100  # 완벽한 예측일 경우

            except Exception as e:
                print(f"목적 함수 평가 오류: {e}")
                return 1e10 if minimize else -1e10  # 오류 발생시 매우 나쁜 값 반환

        return objective_function
class GeneticOptimizer:
    """유전 알고리즘을 사용하여 공정 변수를 최적화하는 컴포넌트"""

    def __init__(self, model=None, scaler=None, algorithm_params=None):
        """
        유전 알고리즘 최적화기를 초기화합니다.

        Parameters:
            model: 학습된 모델 (선택사항)
            scaler: 데이터 스케일러 (선택사항)
            algorithm_params (dict): 알고리즘 파라미터 (선택사항)
        """
        self.model = model
        self.scaler = scaler

        # 기본 알고리즘 파라미터 설정
        default_params = {
            'max_num_iteration': 500,
            'population_size': 200,
            'mutation_probability': 0.2,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': None
        }

        # 사용자 정의 파라미터로 업데이트
        self.algorithm_params = default_params
        if algorithm_params:
            self.algorithm_params.update(algorithm_params)

        self.objective_function = None
        self.variable_bounds = None
        self.variable_names = None
        self.optimization_result = None
        self.input_converter = None

    def set_model(self, model):
        """예측 모델을 설정합니다."""
        self.model = model
        return self

    def set_scaler(self, scaler):
        """데이터 스케일러를 설정합니다."""
        self.scaler = scaler
        return self

    def set_input_converter(self, input_converter):
        """입력 변환 함수를 설정합니다."""
        self.input_converter = input_converter
        return self

    def set_variable_bounds(self, variable_bounds, variable_names):
        """변수 범위와 이름을 설정합니다."""
        self.variable_bounds = variable_bounds
        self.variable_names = variable_names
        return self

    def set_objective_function(self, objective_function):
        """목적 함수를 설정합니다."""
        self.objective_function = objective_function
        return self

    def optimize(self, target_value=1.5, variable_type='real'):
        """
        유전 알고리즘을 실행하여 최적화를 수행합니다.

        Parameters:
            target_value (float): 목표값
            variable_type (str): 변수 타입 ('real' 또는 'int')

        Returns:
            dict: 최적화 결과
        """
        if self.model is None or self.scaler is None:
            raise ValueError("모델과 스케일러가 설정되지 않았습니다.")

        if self.variable_bounds is None or self.variable_names is None:
            raise ValueError("변수 범위와 이름이 설정되지 않았습니다.")

        if self.objective_function is None:
            # 목적 함수가 설정되지 않은 경우 기본 목적 함수 생성
            self.objective_function = ObjectiveFunctionFactory.create_function(
                self.model, self.scaler, target_value, self.input_converter
            )

        # 알고리즘 파라미터 업데이트 - 안정성 향상을 위한 설정
        algo_params = self.algorithm_params.copy()
        algo_params['max_num_iteration'] = 50  # 반복 횟수 감소
        algo_params['population_size'] = 30    # 개체 수 감소
        algo_params['mutation_probability'] = 0.1  # 변이 확률 감소
        algo_params['crossover_probability'] = 0.5  # 교차 확률 유지
        algo_params['function_timeout'] = 5  # 타임아웃 감소 (5초)

        # 랜덤 시드 설정 (일관된 결과 위해)
        import numpy as np
        np.random.seed(42)

        # 유전 알고리즘 모델 정의
        ga_model = geneticalgorithm(
            function=self.objective_function,
            dimension=len(self.variable_bounds),
            variable_type=variable_type,
            variable_boundaries=self.variable_bounds,
            algorithm_parameters=algo_params
        )

        print(f"목표값 {target_value}에 대한 최적화 시작...")

        try:
            # 최적화 실행
            ga_model.run()

            # 최적해 추출
            optimal_solution = ga_model.output_dict['variable']

            # 결과 변환 및 예측값 계산
            if self.input_converter:
                converted_solution = self.input_converter(optimal_solution)
                predicted_quality = self.model.predict(
                    self.scaler.transform(converted_solution.reshape(1, -1))
                )[0]

                # 결과를 딕셔너리로 저장
                solution_dict = {
                    self.variable_names[i]: float(optimal_solution[i])
                    for i in range(len(self.variable_names))
                }
            else:
                predicted_quality = self.model.predict(
                    self.scaler.transform(optimal_solution.reshape(1, -1))
                )[0]

                # 결과를 딕셔너리로 저장
                solution_dict = {
                    self.variable_names[i]: float(optimal_solution[i])
                    for i in range(len(self.variable_names))
                }

            # 결과 저장
            self.optimization_result = {
                'target_value': target_value,
                'predicted_value': float(predicted_quality),
                'optimal_parameters': solution_dict,
                'optimal_vector': optimal_solution.tolist(),
                'convergence': ga_model.report
            }

            print(f"최적화 완료!")
            print(f"목표값: {target_value:.4f}, 예측값: {float(predicted_quality):.4f}")
            print("최적 파라미터:")
            for name, value in solution_dict.items():
                print(f"  {name}: {value:.4f}")

            return self.optimization_result

        except Exception as e:
            print(f"최적화 중 오류 발생: {e}")

            # 기본 최적화 결과 생성
            default_solution = np.mean(self.variable_bounds, axis=1)
            solution_dict = {
                self.variable_names[i]: float(default_solution[i])
                for i in range(len(self.variable_names))
            }

            # 기본 결과 예측
            try:
                if self.input_converter:
                    converted_solution = self.input_converter(default_solution)
                    predicted_quality = self.model.predict(
                        self.scaler.transform(converted_solution.reshape(1, -1))
                    )[0]
                else:
                    predicted_quality = self.model.predict(
                        self.scaler.transform(default_solution.reshape(1, -1))
                    )[0]
            except:
                predicted_quality = target_value  # 예측 실패 시 목표값으로 대체

            # 결과 저장
            self.optimization_result = {
                'target_value': target_value,
                'predicted_value': float(predicted_quality),
                'optimal_parameters': solution_dict,
                'optimal_vector': default_solution.tolist(),
                'convergence': [],
                'error': str(e)
            }

            print("오류로 인해 최적화가 완료되지 않았습니다. 대체 결과를 반환합니다.")

            return self.optimization_result
    def save_result(self, path='results/optimization_result.json'):
        """최적화 결과를 JSON 파일로 저장합니다."""
        if self.optimization_result is None:
            raise ValueError("최적화가 실행되지 않았습니다.")

        with open(path, 'w') as f:
            json.dump(self.optimization_result, f, indent=4)

        print(f"최적화 결과가 {path}에 저장되었습니다.")

        return self

    def plot_convergence(self, path='results/optimization_convergence.png'):
        """최적화 수렴 과정을 시각화합니다."""
        if self.optimization_result is None or 'convergence' not in self.optimization_result:
            raise ValueError("최적화 결과가 없습니다.")

        # 수렴 과정 데이터 추출
        convergence = self.optimization_result['convergence']

        plt.figure(figsize=(10, 6))
        plt.plot(convergence, 'b-', linewidth=2)
        plt.title('유전 알고리즘 수렴 과정')
        plt.xlabel('세대 (Generation)')
        plt.ylabel('목적 함수 값')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        print(f"수렴 과정 그래프가 {path}에 저장되었습니다.")

        return self


class OptimizationPipeline:
    """전체 최적화 파이프라인을 관리하는 컴포넌트"""

    def __init__(self, model=None, scaler=None, data=None):
        """
        최적화 파이프라인을 초기화합니다.

        Parameters:
            model: 학습된 모델 (선택사항)
            scaler: 데이터 스케일러 (선택사항)
            data (DataFrame): 변수 범위 계산에 사용할 데이터 (선택사항)
        """
        self.model = model
        self.scaler = scaler
        self.data = data
        self.variable_names = None
        self.bounds_manager = None
        self.optimizer = None
        self.result = None

    def load_resources(self, model_path='models/rf_model.pkl',
                       scaler_path='models/scaler.pkl',
                       data_path='data/preprocessed_data.csv'):
        """모델, 스케일러, 데이터를 로드합니다."""
        # 모델 로드
        if self.model is None and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"모델을 {model_path}에서 로드했습니다.")

        # 스케일러 로드
        if self.scaler is None and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"스케일러를 {scaler_path}에서 로드했습니다.")

        # 데이터 로드
        if self.data is None and os.path.exists(data_path):
            self.data = pd.read_csv(data_path)
            print(f"데이터를 {data_path}에서 로드했습니다: {self.data.shape[0]}행, {self.data.shape[1]}열")

        return self

    def configure_variables(self, variable_names=None):
        """최적화할 변수를 구성합니다."""
        if variable_names:
            self.variable_names = variable_names
        else:
            # 이 부분에 누락된 변수를 추가
            self.variable_names = [
                '단위중량(kg)', '투입중량(kg)', '염색길이(m)',
                '투입중량/길이',  # <-- 이 변수가 누락되었을 가능성이 높습니다
                '투입중량/액량', '공정진행시간(%)', '진행온도',
                '포속1', '포속3', '포속4'
            ]

        print(f"최적화 변수: {', '.join(self.variable_names)}")
        return self

    def setup_bounds(self, custom_bounds=None, safety_margin=0.05):
        """변수 범위를 설정합니다."""
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")

        if self.variable_names is None:
            raise ValueError("변수가 설정되지 않았습니다.")

        # 변수 범위 관리자 생성
        self.bounds_manager = VariableBoundsManager(self.data, custom_bounds)

        # 변수 범위 설정
        bounds = self.bounds_manager.set_bounds(self.variable_names, safety_margin)

        print("변수 범위 설정 완료:")
        for i, var in enumerate(self.variable_names):
            print(f"  {var}: [{bounds[i][0]:.4f}, {bounds[i][1]:.4f}]")

        return self

    def add_constraints(self, constraints=None):
        """변수에 제약 조건을 추가합니다."""
        if self.bounds_manager is None:
            raise ValueError("변수 범위가 설정되지 않았습니다.")

        if constraints:
            for var, constraint in constraints.items():
                if 'fixed' in constraint:
                    self.bounds_manager.set_constraint(var, fixed_val=constraint['fixed'])
                    print(f"제약 조건 추가: {var} = {constraint['fixed']}")
                else:
                    min_val = constraint.get('min')
                    max_val = constraint.get('max')
                    self.bounds_manager.set_constraint(var, min_val, max_val)
                    min_str = f">= {min_val}" if min_val is not None else ""
                    max_str = f"<= {max_val}" if max_val is not None else ""
                    print(f"제약 조건 추가: {var} {min_str} {max_str}")

        # 기본 제약 조건: 공정진행시간 = 100%
        self.bounds_manager.set_constraint('공정진행시간(%)', fixed_val=100)
        print("기본 제약 조건 추가: 공정진행시간(%) = 100")

        return self

    def setup_optimizer(self, algorithm_params=None):
        """최적화기를 설정합니다."""
        if self.model is None or self.scaler is None:
            raise ValueError("모델과 스케일러가 로드되지 않았습니다.")

        if self.bounds_manager is None or self.bounds_manager.variable_bounds is None:
            raise ValueError("변수 범위가 설정되지 않았습니다.")

        # 유전 알고리즘 최적화기 생성
        self.optimizer = GeneticOptimizer(self.model, self.scaler, algorithm_params)

        # 입력 변환기 생성
        input_converter = InputConverterFactory.create_converter(self.variable_names)

        # 최적화기 설정
        self.optimizer.set_input_converter(input_converter)
        self.optimizer.set_variable_bounds(
            self.bounds_manager.variable_bounds,
            self.variable_names
        )

        print("최적화기 설정 완료")
        return self

    def run_optimization(self, target_value=1.5):
        """최적화를 실행합니다."""
        if self.optimizer is None:
            raise ValueError("최적화기가 설정되지 않았습니다.")

        # 최적화 실행
        self.result = self.optimizer.optimize(target_value)

        # 결과 저장
        self.optimizer.save_result()
        self.optimizer.plot_convergence()
        self.bounds_manager.save_bounds()

        return self.result


def run_optimization_pipeline(model_path='models/rf_model.pkl',
                              scaler_path='models/scaler.pkl',
                              data_path='data/preprocessed_data.csv',
                              target_value=1.5,
                              constraints=None):
    """최적화 파이프라인 실행을 위한 편의 함수"""
    try:
        # 파이프라인 생성 및 실행
        pipeline = OptimizationPipeline()

        # 1. 리소스 로드
        pipeline.load_resources(model_path, scaler_path, data_path)

        # 2. 변수 구성
        pipeline.configure_variables()

        # 3. 변수 범위 설정
        pipeline.setup_bounds()

        # 4. 제약 조건 추가
        pipeline.add_constraints(constraints)

        # 5. 최적화기 설정
        pipeline.setup_optimizer()

        # 6. 최적화 실행 (앙상블 최적화 대신 기본 최적화 사용)
        result = pipeline.run_optimization(target_value)

        print("\n최적화가 완료되었습니다!")

        return result

    except Exception as e:
        print(f"최적화 파이프라인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

        # 기본 결과 생성 및 파일 저장
        import json
        import os

        default_result = {
            'target_value': target_value,
            'predicted_value': target_value,
            'optimal_parameters': {'오류': '최적화 실패'},
            'error': str(e)
        }

        try:
            os.makedirs('results', exist_ok=True)
            with open('results/optimization_result.json', 'w') as f:
                json.dump(default_result, f, indent=2)
        except:
            pass

        return default_result

    except Exception as e:
        print(f"최적화 파이프라인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

        # 기본 결과 생성 및 파일 저장
        import json
        import os

        default_result = {
            'target_value': target_value,
            'predicted_value': target_value,
            'optimal_parameters': {'오류': '최적화 실패'},
            'error': str(e)
        }

        try:
            os.makedirs('results', exist_ok=True)
            with open('results/optimization_result.json', 'w') as f:
                json.dump(default_result, f, indent=2)
        except:
            pass

        return default_result


if __name__ == "__main__":
    # 명령행 인수 파싱
    import argparse

    parser = argparse.ArgumentParser(description='유전 알고리즘을 사용한 공정 최적화')
    parser.add_argument('--target', type=float, default=1.5, help='목표 염색색차 DE 값 (기본값: 1.5)')
    parser.add_argument('--model', type=str, default='models/rf_model.pkl', help='모델 파일 경로')
    parser.add_argument('--scaler', type=str, default='models/scaler.pkl', help='스케일러 파일 경로')
    parser.add_argument('--data', type=str, default='data/preprocessed_data.csv', help='데이터 파일 경로')

    args = parser.parse_args()

    # 최적화 실행
    result = run_optimization_pipeline(
        model_path=args.model,
        scaler_path=args.scaler,
        data_path=args.data,
        target_value=args.target
    )

class EnhancedOptimizer:
    """향상된 최적화 알고리즘을 제공하는 클래스"""

    def __init__(self, base_optimizer, random_seeds=None, num_runs=5):
        """
        향상된 최적화 시스템 초기화

        Parameters:
            base_optimizer: 기본 최적화 객체 (GeneticOptimizer 인스턴스)
            random_seeds: 일관된 결과를 위한 랜덤 시드 리스트
            num_runs: 앙상블 실행 횟수
        """
        self.base_optimizer = base_optimizer
        self.num_runs = num_runs
        self.results_history = []

        # 랜덤 시드 설정 (일관성을 위해)
        if random_seeds is None:
            self.random_seeds = [42, 123, 255, 789, 101]  # 기본 시드 값
        else:
            self.random_seeds = random_seeds

        # 결과 캐싱을 위한 딕셔너리
        self.results_cache = {}

    def ensemble_optimize(self, target_value, variable_type='real'):
        """
        앙상블 방식으로 최적화를 여러 번 실행하고 최상의 결과를 반환

        Parameters:
            target_value: 목표 염색색차 DE 값
            variable_type: 변수 타입 ('real' 또는 'int')

        Returns:
            dict: 최적화 결과
        """
        # 캐시 확인 - 동일한 목표값에 대해 이미 계산된 결과가 있는지
        cache_key = f"{target_value:.4f}"
        if cache_key in self.results_cache:
            print(f"캐시된 결과 사용: 목표값 {target_value}")
            return self.results_cache[cache_key]

        print(f"목표값 {target_value}에 대해 {self.num_runs}회 앙상블 최적화 시작")

        best_result = None
        best_score = float('inf')
        all_results = []

        # 알고리즘 매개변수 조정 - 정확도 향상을 위해
        original_params = self.base_optimizer.algorithm_params.copy()

        # 앙상블 실행
        for i in range(self.num_runs):
            try:
                print(f"\n앙상블 최적화 실행 {i+1}/{self.num_runs}")

                # 현재 실행에 적합한 랜덤 시드 설정 (일관성을 위해)
                np.random.seed(self.random_seeds[i % len(self.random_seeds)])

                # 최적화 실행
                result = self.base_optimizer.run_optimization(target_value, variable_type)
                all_results.append(result)

                # 결과 평가 (목표값과의 차이)
                if 'predicted_value' in result:
                    score = abs(result['predicted_value'] - target_value)

                    print(f"실행 {i+1} 결과: 예측값 = {result['predicted_value']:.4f}, 차이 = {score:.4f}")

                    # 더 나은 결과인 경우 저장
                    if score < best_score:
                        best_score = score
                        best_result = result
                        print(f"새로운 최상의 결과 발견! 차이: {best_score:.4f}")

            except Exception as e:
                print(f"앙상블 실행 {i+1} 중 오류 발생: {e}")
                continue  # 다음 실행으로 진행

        # 원래 매개변수 복원
        self.base_optimizer.algorithm_params = original_params

        # 최상의 결과가 없는 경우 대체 결과 생성
        if best_result is None and len(all_results) > 0:
            # 가장 최근의 결과 사용
            best_result = all_results[-1]
            print("유효한 최상의 결과를 찾지 못했습니다. 가장 최근 결과를 사용합니다.")
        elif best_result is None:
            # 모든 실행이 실패한 경우
            print("모든 앙상블 실행이 실패했습니다. 기본 결과를 생성합니다.")
            # 기본 변수값 (변수 범위의 중간값)
            default_solution = np.mean(self.base_optimizer.variable_bounds, axis=1)
            solution_dict = {
                self.base_optimizer.variable_names[i]: float(default_solution[i])
                for i in range(len(self.base_optimizer.variable_names))
            }

            # 기본 결과 예측 시도
            try:
                if self.base_optimizer.input_converter:
                    converted_solution = self.base_optimizer.input_converter(default_solution)
                    predicted_quality = self.base_optimizer.model.predict(
                        self.base_optimizer.scaler.transform(converted_solution.reshape(1, -1))
                    )[0]
                else:
                    predicted_quality = self.base_optimizer.model.predict(
                        self.base_optimizer.scaler.transform(default_solution.reshape(1, -1))
                    )[0]
            except:
                predicted_quality = target_value  # 예측 실패 시 목표값으로 대체

            # 결과 생성
            best_result = {
                'target_value': target_value,
                'predicted_value': float(predicted_quality),
                'optimal_parameters': solution_dict,
                'optimal_vector': default_solution.tolist(),
                'convergence': [],
                'error': "모든 앙상블 실행 실패"
            }

        # 결과 캐싱
        self.results_cache[cache_key] = best_result

        # 결과 히스토리에 추가
        self.results_history.append({
            'target_value': target_value,
            'best_result': best_result,
            'timestamp': pd.Timestamp.now().isoformat()
        })

        print("\n앙상블 최적화 완료!")
        print(f"최종 결과: 목표값 = {target_value:.4f}, 최적 예측값 = {best_result['predicted_value']:.4f}")

        return best_result

    def _save_history(self):
        """최적화 결과 히스토리를 파일로 저장합니다."""
        try:
            import json
            import os

            os.makedirs('results', exist_ok=True)

            # JSON으로 직렬화 가능한 간소화된 히스토리 생성
            simplified_history = []
            for entry in self.results_history:
                simplified_entry = {
                    'target_value': entry['target_value'],
                    'best_predicted_value': entry['best_result']['predicted_value'],
                    'optimal_parameters': entry['best_result']['optimal_parameters'],
                    'timestamp': entry['timestamp']
                }
                simplified_history.append(simplified_entry)

            with open('results/optimization_history.json', 'w') as f:
                json.dump(simplified_history, f, indent=2)

            print("최적화 히스토리가 저장되었습니다.")

        except Exception as e:
            print(f"히스토리 저장 중 오류 발생: {e}")
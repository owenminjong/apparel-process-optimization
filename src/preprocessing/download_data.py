import os
import gdown
import requests
from pathlib import Path
import re

def download_data(force_download=False):
    """
    Google Drive에서 데이터 파일을 다운로드합니다.

    Args:
        force_download (bool): True이면 이미 파일이 존재해도 강제로 다시 다운로드합니다.
    """
    # 데이터 디렉토리 생성
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Google Drive 공유 링크와 저장할 파일명 매핑
    files = {
        "https://docs.google.com/spreadsheets/d/1sbLfJ7BHnNqElwu2og2Q7MIy5yYX9ARl/edit?usp=drive_link&ouid=106112173059127185977&rtpof=true&sd=true": "LOT 물량.xlsx",
        "https://docs.google.com/spreadsheets/d/1mc3ta6B_-6Nce8jVusPCFusI0T0LI0vL/edit?usp=drive_link&ouid=106112173059127185977&rtpof=true&sd=true": "CCM 측정값.xlsx",
        "https://drive.google.com/file/d/1hayIWxFsoraECby4BUI4CLUO0-NEu1-2/view?usp=drive_link": "PRODUCTION_TREND.csv",
        "https://docs.google.com/spreadsheets/d/1KlN0czIPDLAgCPOFfeE5LjbPDIhUrGSq/edit?usp=drive_link&ouid=106112173059127185977&rtpof=true&sd=true": "CCM 측정값_품질전처리 데이터셋.xlsx",
        "https://docs.google.com/spreadsheets/d/1V26EbWlCniA-IDzEU2NTrDnxwbZXBu-c/edit?usp=drive_link&ouid=106112173059127185977&rtpof=true&sd=true": "LOT 물량_품질전처리 데이터셋.xlsx",
        "https://drive.google.com/file/d/1D2SkTkFDAvvmDk-eLjAicBp3r2ujamfH/view?usp=drive_link": "PRODUCTION_TREND_품질전처리 데이터셋.csv",
        "https://drive.google.com/file/d/1iFOjoAjRqveB3x0kymeBtcuZI8WWPrXK/view?usp=drive_link": "최종데이터셋_전처리 완료.csv"
    }

    # 파일 다운로드
    for file_url, filename in files.items():
        output_path = data_dir / filename

        # 파일이 존재하지 않거나 강제 다운로드 옵션이 True인 경우 다운로드
        if not output_path.exists() or force_download:
            print(f"다운로드 중: {filename}")
            try:
                # 파일 유형에 따라 다른 처리
                if "file/d/" in file_url:  # 일반 파일
                    # file_id 추출
                    file_id = re.search(r'file/d/([^/]+)', file_url).group(1)
                    gdown.download(f"https://drive.google.com/uc?id={file_id}",
                                   str(output_path), quiet=False)
                elif "spreadsheets/d/" in file_url:  # 스프레드시트
                    # file_id 추출
                    file_id = re.search(r'spreadsheets/d/([^/]+)', file_url).group(1)
                    gdown.download(f"https://drive.google.com/uc?id={file_id}&export=download",
                                   str(output_path), quiet=False)

                print(f"다운로드 완료: {filename}")
            except Exception as e:
                print(f"다운로드 실패: {filename}, 오류: {e}")
        else:
            print(f"파일이 이미 존재합니다: {filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='데이터 파일 다운로드')
    parser.add_argument('--force', action='store_true', help='이미 존재하는 파일도 강제로 다시 다운로드')
    args = parser.parse_args()

    download_data(force_download=args.force)
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================================================================================
# 설정값
# ================================================================================
INPUT_ROOT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\segment_10m"
OUTPUT_FILE = 'Terminal_Max_Temp_Deviation_Detailed.csv'

MOD_TEMP_COL = 'mod_temp_list'
EXCLUDE_MODEL = "Porter2EV"

# ================================================================================
def analyze_file_deviation(file_path):
    """ 개별 파일에서 0이 포함되지 않은 행의 모듈 온도 편차 최댓값 및 파일 정보 추출 """
    try:
        p = Path(file_path)
        
        # 경로 분석
        rel_path = os.path.relpath(file_path, INPUT_ROOT_DIR)
        path_parts = rel_path.split(os.sep)
        
        vehicle_model = path_parts[0] if len(path_parts) > 0 else "Unknown"
        terminal_id = path_parts[1] if len(path_parts) > 1 else "Unknown"

        # Porter2EV 제외
        if EXCLUDE_MODEL.lower() in vehicle_model.lower():
            return None

        # CSV 읽기
        try: df = pd.read_csv(file_path, encoding='utf-8')
        except:
            try: df = pd.read_csv(file_path, encoding='cp949')
            except: return None

        if df.empty or MOD_TEMP_COL not in df.columns:
            return None

        max_dev_in_file = 0.0
        found_valid_row = False
        
        for temp_str in df[MOD_TEMP_COL]:
            if pd.isna(temp_str): continue
            
            try:
                # 쉼표 구분자 처리 및 숫자 변환
                temps = [float(x.strip()) for x in str(temp_str).split(',') if x.strip()]
            except: continue
                
            if not temps: continue
            
            # 0 또는 0.0 포함 시 제외 
            if 0.0 in temps or 0 in temps: continue
            
            # 해당 행의 편차 계산
            deviation = max(temps) - min(temps)
            if deviation > max_dev_in_file:
                max_dev_in_file = deviation
            found_valid_row = True

        if not found_valid_row:
            return None

        return {
            '차종': vehicle_model,
            '단말기번호': terminal_id,
            '최대_편차': max_dev_in_file,
            '파일_이름': p.name,
            '파일_경로': str(p)
        }

    except Exception:
        return None

# ================================================================================
def main():
    print(f"파일 목록 수집 중 ({INPUT_ROOT_DIR})")
    
    files_to_process = []
    for root, dirs, files in os.walk(INPUT_ROOT_DIR):
        for f in files:
            if f.lower().endswith('.csv'):
                files_to_process.append(os.path.join(root, f))

    total_files = len(files_to_process)
    if total_files == 0:
        print("파일을 찾을 수 없음")
        return

    print(f" {total_files:,}개 파일 ")

    results = []
    
    for f in tqdm(files_to_process, desc="분석 진행 중"):
        res = analyze_file_deviation(f)
        if res:
            results.append(res)

    if results:
        print(f"데이터 집계 및 저장 중")
        df_res = pd.DataFrame(results)
        
        # 단말기별로 '최대_편차'가 가장 큰 행만 남기기
        df_res = df_res.sort_values(by='최대_편차', ascending=False)
        terminal_summary = df_res.drop_duplicates(subset=['차종', '단말기번호'], keep='first')
        
        # 결과 저장
        terminal_summary.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n결과 파일: {OUTPUT_FILE}")
    else:
        print("\n분석할 데이터가 없음")

if __name__ == "__main__":
    main()
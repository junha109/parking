import os
import pandas as pd
import shutil
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 경고 메시지 숨김
warnings.filterwarnings('ignore')

# ================================================================================
# 설정값
# ================================================================================
INPUT_ROOT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\segment_10m"
OUTPUT_ROOT_DIR = r'Z:\SamsungSTF\Processed_Data\Parking\classified_10m'
LOG_OUTPUT_FILE = 'Zero_Temperature_Segments_Log.csv'

SOC_COL_NAME = 'soc'  
MOD_TEMP_COL = 'mod_temp_list'
TIME_COL = 'time'
CABLE_CONN_COL = 'chrg_cable_conn' # 충전 케이블 연결 여부 컬럼 추가
CHARGING_THRESHOLD = 5

# 제외할 차종
EXCLUDE_MODEL = "Porter2EV"

# ================================================================================
def get_season(month):
    """월을 입력받아 계절 반환"""
    if month in [12, 1, 2]: return "겨울"
    elif month in [3, 4, 5]: return "봄"
    elif month in [6, 7, 8]: return "여름"
    elif month in [9, 10, 11]: return "가을"
    return "알수없음"

def process_single_file(file_path):
    """
    조건 1: SOC 차이가 CHARGING_THRESHOLD 이상
    조건 2: chrg_cable_conn 컬럼에 1이 포함
    """
    try:
        p = Path(file_path)
        rel_path = os.path.relpath(file_path, INPUT_ROOT_DIR)
        path_parts = rel_path.split(os.sep)
        
        vehicle_model = path_parts[0] if len(path_parts) > 0 else "Unknown"
        terminal_id = path_parts[1] if len(path_parts) > 1 else "Unknown"

        if EXCLUDE_MODEL.lower() in vehicle_model.lower():
            return 0, None

        # CSV 읽기
        try: 
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            try: df = pd.read_csv(file_path, encoding='cp949')
            except: return 0, None

        if df.empty:
            return 0, None

        # 계절 정보 파악
        season = "알수없음"
        if TIME_COL in df.columns:
            try:
                start_time = pd.to_datetime(df[TIME_COL].iloc[0])
                season = get_season(start_time.month)
            except: pass

        # 모듈 온도 0도 포함 체크
        zero_count = 0
        error_info = None
        if MOD_TEMP_COL in df.columns:
            def count_zeros(temp_str):
                if pd.isna(temp_str): return False
                temps = [x.strip() for x in str(temp_str).split(',') if x.strip()]
                return '0' in temps or '0.0' in temps
            
            zero_rows_mask = df[MOD_TEMP_COL].apply(count_zeros)
            zero_count = zero_rows_mask.sum()
            
            if zero_count > 0:
                error_info = {
                    '차종': vehicle_model, '단말기번호': terminal_id, '계절': season,
                    '세그먼트 내에 모듈온도가 0이 포함된 행 갯수': zero_count,
                    '파일 이름': p.name, '파일 경로': str(p)
                }

        # ========================================================================
        # 충전 분류 로직 수정
        # ========================================================================
        # SOC 차이 조건
        soc_cond = False
        if SOC_COL_NAME in df.columns:
            start_soc = df[SOC_COL_NAME].iloc[0]
            end_soc = df[SOC_COL_NAME].iloc[-1]
            soc_cond = (end_soc - start_soc) >= CHARGING_THRESHOLD

        # 충전 케이블 연결 조건 (1이 하나라도 있으면 True)
        cable_cond = False
        if CABLE_CONN_COL in df.columns:
            cable_cond = (df[CABLE_CONN_COL] == 1).any()

        # 최종 분류 (두 조건 중 하나라도 만족하면 Charging)
        category = 'Charging' if (soc_cond or cable_cond) else 'Rest'
        new_tag = category.lower()

        # 파일 복사
        target_dir = os.path.join(OUTPUT_ROOT_DIR, os.path.dirname(rel_path), category)
        os.makedirs(target_dir, exist_ok=True)

        original_filename = p.name
        new_filename = original_filename.replace('parking', new_tag) if 'parking' in original_filename else f"{new_tag}_{original_filename}"
        
        shutil.copy2(file_path, os.path.join(target_dir, new_filename))
        return 1, error_info
            
    except Exception:
        return 0, None

def main():
    print(f"파일 목록 검색 중({INPUT_ROOT_DIR})")
    
    files_to_process = []
    for root, dirs, files in os.walk(INPUT_ROOT_DIR):
        if os.path.abspath(OUTPUT_ROOT_DIR) in os.path.abspath(root):
            continue
        for f in files:
            if f.lower().endswith('.csv'):
                files_to_process.append(os.path.join(root, f))

    total_files = len(files_to_process)
    if total_files == 0:
        print("처리할 파일 없음")
        return

    print(f"병렬 처리 시작 총 {total_files:,}개")
    print(f"분류 조건: SOC 상승 {CHARGING_THRESHOLD}% 이상 OR 케이블 연결 신호(1) 존재")

    success_count = 0
    zero_temp_logs = []
    
    max_workers = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, f): f for f in files_to_process}
        
        for future in tqdm(as_completed(futures), total=total_files, desc="분류 및 분석 중", unit="file"):
            try:
                success, error_log = future.result()
                success_count += success
                if error_log:
                    zero_temp_logs.append(error_log)
            except: pass

    if zero_temp_logs:
        log_df = pd.DataFrame(zero_temp_logs)
        cols = ['차종', '단말기번호', '계절', '세그먼트 내에 모듈온도가 0이 포함된 행 갯수', '파일 이름', '파일 경로']
        log_df = log_df[cols]
        log_df.to_csv(LOG_OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n0도 포함 로그 저장: {LOG_OUTPUT_FILE} ({len(zero_temp_logs)}건)")

    print("-" * 50)
    print(f"작업 완료")
    print(f"총 파일: {total_files:,}개, 성공: {success_count:,}개")
if __name__ == "__main__":
    main()
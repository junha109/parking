import pandas as pd
import numpy as np
import os
import warnings
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ================================================================================
# 경로 및 설정 지정
# ================================================================================
RAW_DATA_FOLDER = r"Z:\SamsungSTF\Processed_Data\Merged_period_final"
PARKING_OUTPUT_FOLDER = r"Z:\SamsungSTF\Processed_Data\Parking\segment3"
MIN_PARKING_SEC = 600 # 최소 주차 시간 (초)

# ================================================================================
# 보조 함수
# ================================================================================
def calculate_avg_temp(temp_str):
    """문자열 형태의 온도 리스트를 평균값으로 계산"""
    try:
        if pd.isna(temp_str) or temp_str == '': # ,로 구분
            return np.nan
        temps = [float(x) for x in str(temp_str).split(',') if x.strip()]
        return sum(temps) / len(temps) if temps else np.nan
    except: 
        return np.nan

def get_start_time(file_path):
    """파일 첫 행에서 시간 정보를 추출"""
    try:
        df = pd.read_csv(file_path, nrows=1, usecols=['time'])
        return pd.to_datetime(df['time'].iloc[0])
    except:
        return pd.Timestamp.max


# ================================================================================
def process_file_logic(file_info):
    file_path, vehicle_type, device_id, output_root = file_info
    
    try:
        # 파일 읽기 
        try: 
            df = pd.read_csv(file_path, encoding='utf-8')
        except: 
            try: 
                df = pd.read_csv(file_path, encoding='cp949')
            except: 
                df = pd.read_csv(file_path, encoding='iso-8859-1')

        if df.empty: 
            return 0

        # 필수 컬럼 확인 및 전처리
        if 'speed' not in df.columns or 'time' not in df.columns: 
            return 0
            
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['speed', 'time']).sort_values('time')

        if df.empty: 
            return 0

        # 전력 소모 컬럼(pack_current) 확인
        current_col = 'pack_current' if 'pack_current' in df.columns else ('current' if 'current' in df.columns else None)
        if current_col:
            df[current_col] = pd.to_numeric(df[current_col], errors='coerce')

        # 년월 추출 (첫 데이터 기준)
        year_month = df['time'].iloc[0].strftime('%Y%m')

        # 모듈 온도 평균 계산
        if 'mod_temp_list' in df.columns:
            df['mod_avg_temp'] = df['mod_temp_list'].apply(calculate_avg_temp)

        # 주차 판단 로직 (속도 0 & 전류 3 이하)
        speed_cond = (df['speed'] == 0)
        current_cond = (df[current_col] <= 3) if current_col else True
        
        df['is_parking'] = speed_cond & current_cond
        # 상태가 변할 때마다 그룹 ID 부여
        df['group_id'] = (df['is_parking'] != df['is_parking'].shift()).cumsum()

        # 저장 경로 설정
        save_dir = os.path.join(output_root, vehicle_type, device_id, year_month)
        os.makedirs(save_dir, exist_ok=True)

        segment_count = 0
        original_name_marker = os.path.basename(file_path).replace('.csv', '').replace('.CSV', '')[-5:] 

        for _, group in df.groupby('group_id'):
            if not group['is_parking'].iloc[0]: 
                continue
            
            # 주차 지속 시간 확인
            duration_sec = (group['time'].iloc[-1] - group['time'].iloc[0]).total_seconds()
            if duration_sec < MIN_PARKING_SEC: 
                continue

            segment_count += 1
            out_name = f"TEMP_{device_id}_{year_month}_{original_name_marker}_{segment_count:03d}.csv"
            out_path = os.path.join(save_dir, out_name)
            
            cols_to_save = [c for c in group.columns if c not in ['is_parking', 'group_id']]
            group[cols_to_save].to_csv(out_path, index=False)
        
        return segment_count

    except Exception:
        return 0

# ================================================================================
# 후처리 (파일 정렬 및 이름 변경)
# ================================================================================
def sort_and_rename_files(target_root):
    print(f"\n 생성된 파일 날짜순 정렬 및 넘버링 ")
    
    target_dirs = []
    for root, _, files in os.walk(target_root):
        if any(f.startswith('TEMP_') for f in files):
            target_dirs.append(root)
    
    if not target_dirs:
        print(" 정리할 파일이 없습니다.")
        return

    for folder in tqdm(target_dirs, desc="Finalizing"):
        csv_files = glob.glob(os.path.join(folder, "TEMP_*.csv"))
        if not csv_files: 
            continue
        
        # 파일 시작 시간 기준으로 리스트 생성 및 정렬
        file_time_list = [(f_path, get_start_time(f_path)) for f_path in csv_files]
        file_time_list.sort(key=lambda x: x[1])
        
        for idx, (old_path, _) in enumerate(file_time_list):
            parts = os.path.basename(old_path).split('_')
            d_id = parts[1] if len(parts) >= 3 else "unknown"
            ym = parts[2] if len(parts) >= 3 else "000000"
            
            new_name = f"{d_id}_{ym}_parking_{idx+1:03d}.csv"
            new_path = os.path.join(folder, new_name)
            
            try: 
                os.rename(old_path, new_path)
            except: 
                pass

# ================================================================================
# 메인 실행부
# ================================================================================
def main():
    print(f"\n 주차 세그먼트 추출")
    
    target_files = []
    print("\n 파일 탐색 중...")
    
    for root, _, files in os.walk(RAW_DATA_FOLDER):
        vehicle_type = os.path.basename(root)
        for f in files:
            if not f.lower().endswith('.csv'): 
                continue
            
            name_body = f.replace('.csv', '').replace('.CSV', '')
            parts = name_body.split('_')
            
            # 11자리 숫자로 된 Device ID 찾기
            device_id = next((part for part in parts if part.isdigit() and len(part) == 11), None)
            
            if device_id:
                target_files.append((os.path.join(root, f), vehicle_type, device_id, PARKING_OUTPUT_FOLDER))

    if not target_files:
        print(" 처리할 파일이 없음")
        return

    max_workers = max(1, os.cpu_count() - 2)
    print(f"\n 데이터 처리 시작 (Workers: {max_workers})")

    total_segments = 0
    with tqdm(total=len(target_files), desc="Processing") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_file_logic, f): f for f in target_files}
            
            for future in as_completed(future_to_file):
                try: 
                    total_segments += future.result()
                except: 
                    pass
                finally: 
                    pbar.update(1)

    # 임시 파일들을 시간순으로 재정렬하여 최종 이름 부여
    sort_and_rename_files(PARKING_OUTPUT_FOLDER)
    print(f"\n 작업 완료. 총 {total_segments:,}개")

if __name__ == '__main__':

    main()

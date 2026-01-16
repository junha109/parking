import pandas as pd
import numpy as np
import os
import warnings
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 경고 메시지 숨김
warnings.filterwarnings('ignore')

# ================================================================================
# 경로 지정
# ================================================================================
RAW_DATA_FOLDER = r"Z:\SamsungSTF\Processed_Data\Merged_period_final"
PARKING_OUTPUT_FOLDER = r"Z:\SamsungSTF\Processed_Data\Parking\segment"
MIN_PARKING_SEC = 3600  

# ================================================================================
# 보조 함수
# ================================================================================
def calculate_avg_temp(temp_str):
    try:
        if pd.isna(temp_str) or temp_str == '': return np.nan
        temps = [float(x) for x in str(temp_str).split(',') if x.strip()]
        return sum(temps) / len(temps) if temps else np.nan
    except: return np.nan

def get_start_time(file_path):
    try:
        df = pd.read_csv(file_path, nrows=1, usecols=['time'])
        return pd.to_datetime(df['time'].iloc[0])
    except:
        return pd.Timestamp.max

# ================================================================================
# 파일 처리기
# ================================================================================
def process_file_logic(file_info):
    file_path, vehicle_type, device_id, output_root = file_info
    
    try:
        # 파일 읽기
        try: df = pd.read_csv(file_path, encoding='utf-8')
        except: 
            try: df = pd.read_csv(file_path, encoding='cp949')
            except: df = pd.read_csv(file_path, encoding='iso-8859-1')

        if df.empty: return 0


        if 'speed' not in df.columns: return 0
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
        
        # pack_current 찾기
        current_col = None
        if 'pack_current' in df.columns: current_col = 'pack_current'
        elif 'current' in df.columns: current_col = 'current' 

        if current_col:
            df[current_col] = pd.to_numeric(df[current_col], errors='coerce')

        df = df.dropna(subset=['speed'])
        
        # 시간 변환
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time'])
            df = df.sort_values('time')
            if df.empty: return 0
        else: return 0 

        # 년월 추출
        try:
            year_month = df['time'].iloc[len(df)//2].strftime('%Y%m')
        except:
            year_month = "Unknown_Date"

        # 모듈 온도
        if 'mod_temp_list' in df.columns:
            df['mod_avg_temp'] = df['mod_temp_list'].apply(calculate_avg_temp)

        # 주차 판단 로직 
        speed_cond = (df['speed'] == 0)
        
        if current_col:
            current_cond = (df[current_col] <= 3)
        else:
            current_cond = True 

        df['is_parking'] = speed_cond & current_cond
        df['group_id'] = (df['is_parking'] != df['is_parking'].shift()).cumsum()

        # 저장
        save_dir = os.path.join(output_root, vehicle_type, device_id, year_month)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        segment_count = 0
        
        original_name_marker = os.path.basename(file_path).replace('.csv', '').replace('.CSV', '')[-5:] 

        for g_id, group in df.groupby('group_id'):
            if not group['is_parking'].iloc[0]: continue
            
            duration_sec = (group['time'].iloc[-1] - group['time'].iloc[0]).total_seconds()
            if duration_sec < MIN_PARKING_SEC: continue

            segment_count += 1
            

            out_name = f"TEMP_{device_id}_{year_month}_{original_name_marker}_{segment_count:03d}.csv"
            out_path = os.path.join(save_dir, out_name)
            
            cols_to_save = [c for c in group.columns if c not in ['is_parking', 'group_id']]
            group[cols_to_save].to_csv(out_path, index=False)
        
        return segment_count

    except Exception:
        return 0

# ================================================================================
# 후처리 
# ================================================================================
def sort_and_rename_files(target_root):
    print(f"\n 생성된 파일 날짜순 정렬 및 최종 넘버링")
    
    target_dirs = []
    for root, dirs, files in os.walk(target_root):
        if any(f.startswith('TEMP_') for f in files):
            target_dirs.append(root)
    
    if not target_dirs:
        print(" -> 정리할 파일이 없음음")
        return

    for folder in tqdm(target_dirs, desc="Finalizing"):
        csv_files = glob.glob(os.path.join(folder, "TEMP_*.csv"))
        if not csv_files: continue
        
        # 모든 파일의 시작 시간을 읽음
        file_time_list = []
        for f_path in csv_files:
            file_time_list.append((f_path, get_start_time(f_path)))
        
        # 시간순 정렬
        file_time_list.sort(key=lambda x: x[1])
        
        # 001번부터 다시 이름 붙이기
        for idx, (old_path, _) in enumerate(file_time_list):
            filename = os.path.basename(old_path)
            parts = filename.split('_')
            # 구조: TEMP_{device}_{ym}_{marker}_{count}.csv
            if len(parts) >= 3:
                d_id = parts[1]
                ym = parts[2]
            else:
                d_id = "unknown"
                ym = "000000"
            
            # 최종 이름: device_ym_parking_001.csv
            new_name = f"{d_id}_{ym}_parking_{idx+1:03d}.csv"
            new_path = os.path.join(folder, new_name)
            
            try: os.rename(old_path, new_path)
            except: pass

# ================================================================================
# 실행 함수
# ================================================================================
def main():
    print(f"\n주차 세그먼트 추출")
    
    target_files = []
    print("\n파일 탐색 중...")
    
    for root, dirs, files in os.walk(RAW_DATA_FOLDER):
        vehicle_type = os.path.basename(root)
        for f in files:
            if not f.lower().endswith('.csv'): continue
            
            name_body = f.replace('.csv', '').replace('.CSV', '')
            parts = name_body.split('_')
            device_id = None
            for part in parts:
                if part.isdigit() and len(part) == 11:
                    device_id = part
                    break
            
            if device_id:
                target_files.append((os.path.join(root, f), vehicle_type, device_id, PARKING_OUTPUT_FOLDER))

    if not target_files:
        print(" 처리할 파일이 없음")
        return

    max_workers = max(1, os.cpu_count() - 2)
    print(f"\n데이터 처리 시작 (Workers: {max_workers})")

    total_segments = 0
    with tqdm(total=len(target_files), desc="Processing") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_file_logic, f): f for f in target_files}
            
            for future in as_completed(future_to_file):
                try: total_segments += future.result()
                except: pass
                finally: pbar.update(1)

    sort_and_rename_files(PARKING_OUTPUT_FOLDER)
    print(f"\n 총 {total_segments:,}개의 파일 생성")

if __name__ == '__main__':

    main()

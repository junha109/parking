import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================================================================================
# 설정값
# ================================================================================
BASE_DIR = r'Z:\SamsungSTF\Processed_Data\Parking\Classified_final'
TIME_COL = 'time'
CABLE_CONN_COL = 'chrg_cable_conn'
MIN_REST_DURATION = 10 
TARGET_MODELS = ["EV6", "Ioniq5"]
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# ================================================================================
def process_single_charging_file(file_path):
    try:
        try: df = pd.read_csv(file_path, encoding='utf-8')
        except: df = pd.read_csv(file_path, encoding='cp949')

        if df.empty or CABLE_CONN_COL not in df.columns: return False

        conn_indices = df.index[df[CABLE_CONN_COL] == 1].tolist()
        if not conn_indices: return False

        first_1, last_1 = conn_indices[0], conn_indices[-1]
        rest_folder = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'Rest')
        os.makedirs(rest_folder, exist_ok=True)

        segments = [('pre', df.loc[:first_1 - 1].copy()), ('post', df.loc[last_1 + 1:].copy())]
        temp_count = 0

        for tag, segment_df in segments:
            if not segment_df.empty and TIME_COL in segment_df.columns:
                times = pd.to_datetime(segment_df[TIME_COL], format=TIME_FORMAT, errors='coerce').dropna()
                if len(times) < 2: continue
                
                duration = (times.iloc[-1] - times.iloc[0]).total_seconds() / 60
                if duration >= MIN_REST_DURATION:
                    orig_name = Path(file_path).stem
                    temp_name = f"TEMP_{orig_name}_{tag}.csv"
                    segment_df.to_csv(os.path.join(rest_folder, temp_name), index=False, encoding='utf-8-sig')
                    temp_count += 1

        # Charging 업데이트
        df.loc[first_1:last_1].to_csv(file_path, index=False, encoding='utf-8-sig')
        return True
    except:
        return False

def rename_single_temp_file(temp_path):
    """임시 파일의 이름 변경"""
    try:
        try: df_temp = pd.read_csv(temp_path, encoding='utf-8')
        except: df_temp = pd.read_csv(temp_path, encoding='cp949')

        start_time = pd.to_datetime(df_temp[TIME_COL].iloc[0], format=TIME_FORMAT, errors='coerce')
        if pd.isna(start_time): return False
        
        time_str = start_time.strftime('%Y%m%d_%H%M%S')
        terminal_id = Path(temp_path).parts[-4]
        
        new_name = f"{terminal_id}_{time_str}_rest.csv"
        new_path = os.path.join(os.path.dirname(temp_path), new_name)

        counter = 1
        name_base, ext = os.path.splitext(new_name)
        while os.path.exists(new_path):
            new_path = os.path.join(os.path.dirname(temp_path), f"{name_base}_{counter}{ext}")
            counter += 1

        os.rename(temp_path, new_path)
        return True
    except:
        return False

# ================================================================================
def main():
    #  파일 목록 수집
    charging_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        rel_path = Path(root).relative_to(BASE_DIR)
        if len(rel_path.parts) == 0:
            dirs[:] = [d for d in dirs if d in TARGET_MODELS]
            continue
        if root.endswith('Charging'):
            for f in files:
                if f.lower().endswith('.csv'):
                    charging_files.append(os.path.join(root, f))

    # 1단계 병렬 실행
    print(f" {len(charging_files):,}개 파일")
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_charging_file, f) for f in charging_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="분리 중"):
            pass

    # 임시 파일 목록 수집
    temp_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        rel_path = Path(root).relative_to(BASE_DIR)
        if len(rel_path.parts) == 0:
            dirs[:] = [d for d in dirs if d in TARGET_MODELS]
            continue
        if root.endswith('Rest'):
            for f in files:
                if f.startswith('TEMP_') and f.endswith('.csv'):
                    temp_files.append(os.path.join(root, f))

    # 2단계 병렬 실행
    print(f" 이름 변경 : {len(temp_files):,}개 파일")
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(rename_single_temp_file, f) for f in temp_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="재명명 중"):
            pass

if __name__ == "__main__":
    main()
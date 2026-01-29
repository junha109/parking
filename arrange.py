import os
import pandas as pd
import glob
import shutil
from tqdm import tqdm

# ================================================================================
# 경로 설정
# ================================================================================
PARKING_ROOT = r"Z:\SamsungSTF\Processed_Data\Parking\segment_10m"

def fix_and_rearrange_files(root_path):
    print(f"\n 파일 실제 날짜 확인 및 이동 시작")
    
    # 모든 CSV 파일 찾기
    all_files = []
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.lower().endswith('.csv'):
                all_files.append(os.path.join(root, f))

    for file_path in tqdm(all_files, desc="Moving files"):
        try:
            # 파일의 첫 줄만 읽어서 실제 시간 확인
            df_sample = pd.read_csv(file_path, nrows=1, usecols=['time'])
            if df_sample.empty: continue
            
            actual_time = pd.to_datetime(df_sample['time'].iloc[0])
            actual_ym = actual_time.strftime('%Y%m')
            
            # 현재 경로 분석
            # 구조: .../segment3/{vehicle_type}/{device_id}/{current_ym}/filename.csv
            path_parts = os.path.normpath(file_path).split(os.sep)
            
            # 뒤에서부터 폴더명 추출
            current_ym = path_parts[-2]
            device_id = path_parts[-3]
            vehicle_type = path_parts[-4]
            base_root = os.sep.join(path_parts[:-4])
            
            # 실제 날짜와 폴더명이 다르면 이동
            if actual_ym != current_ym:
                target_dir = os.path.join(base_root, vehicle_type, device_id, actual_ym)
                os.makedirs(target_dir, exist_ok=True)
                
                new_path = os.path.join(target_dir, os.path.basename(file_path))
                
                # 파일 이동 (shutil.move 사용)
                shutil.move(file_path, new_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("\n폴더별 파일명 재넘버링 시작")
    
    # 다시 한번 전체 구조를 돌며 넘버링 정리
    for root, dirs, files in os.walk(root_path):
        # 최하단 폴더(년월 폴더)인지 확인
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        if not csv_files:
            continue
            
        # 폴더 내 모든 파일의 시작 시간을 읽어 정렬
        file_time_list = []
        for f in csv_files:
            f_path = os.path.join(root, f)
            try:
                t = pd.to_datetime(pd.read_csv(f_path, nrows=1, usecols=['time'])['time'].iloc[0])
                file_time_list.append((f_path, t))
            except:
                file_time_list.append((f_path, pd.Timestamp.max))
        
        # 시간 순 정렬
        file_time_list.sort(key=lambda x: x[1])
        
        # 경로 정보 추출
        path_parts = os.path.normpath(root).split(os.sep)
        d_id = path_parts[-2]
        ym = path_parts[-1]
        
        # 임시 이름으로 먼저 변경 (이름 중복 방지)
        temp_renamed = []
        for idx, (old_path, _) in enumerate(file_time_list):
            temp_path = old_path + ".tmp"
            os.rename(old_path, temp_path)
            temp_renamed.append(temp_path)
            
        # 최종 이름으로 변경: {device}_{ym}_parking_{idx}.csv
        for idx, temp_path in enumerate(temp_renamed):
            final_name = f"{d_id}_{ym}_parking_{idx+1:03d}.csv"
            final_path = os.path.join(root, final_name)
            os.rename(temp_path, final_path)

    print("\n모든 작업 완료")

if __name__ == '__main__':
    fix_and_rearrange_files(PARKING_ROOT)
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ================================================================================
# 설정값 
# ================================================================================
BASE_DIR = r'Z:\SamsungSTF\Processed_Data\Parking\classified_10m2'
CABLE_CONN_COL = 'chrg_cable_conn'
TARGET_MODELS = ["EV6", "Ioniq5"]

def verify_charging_segments():
    print(f"{TARGET_MODELS} Charging 세그먼트 내 0 존재 여부 확인")
    
    #  대상 파일 수집
    charging_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        rel_path = Path(root).relative_to(BASE_DIR)
        parts = rel_path.parts
        
        # 대상 차종 필터링
        if len(parts) == 0:
            dirs[:] = [d for d in dirs if d in TARGET_MODELS]
            continue
            
        if root.endswith('Charging'):
            for f in files:
                if f.lower().endswith('.csv'):
                    charging_files.append(os.path.join(root, f))

    if not charging_files:
        print(" 검사할 Charging 파일이 없음")
        return

    print(f"총 {len(charging_files):,}개 파일")

    #  파일 내용 검사
    error_files = []
    
    for file_path in tqdm(charging_files, desc="검사 중"):
        try:
            try:
                df = pd.read_csv(file_path, usecols=[CABLE_CONN_COL], encoding='utf-8')
            except:
                df = pd.read_csv(file_path, usecols=[CABLE_CONN_COL], encoding='cp949')

            # 0이 하나라도 포함되어 있는지 확인
            if (df[CABLE_CONN_COL] == 0).any():
                error_files.append({
                    '파일명': os.path.basename(file_path),
                    '경로': file_path,
                    '0의 개수': (df[CABLE_CONN_COL] == 0).sum()
                })
        except Exception as e:
            continue

    #  결과 
    print("\n" + "="*50)
    if not error_files:
        print("모든 Charging 세그먼트에 0 없음")
    else:
        print(f"0이 포함된 파일 {len(error_files)}개 발견")
        print("="*50)
        # 상위 10개만 출력
        for err in error_files[:10]:
            print(f"파일명: {err['파일명']}, 0의 개수: {err['0의 개수']}, 경로: {err['경로']}")
            
        # 결과 저장
        err_df = pd.DataFrame(error_files)
        err_df.to_csv('Charging_Verify_Errors.csv', index=False, encoding='utf-8-sig')
    print("="*50)

if __name__ == "__main__":
    verify_charging_segments()
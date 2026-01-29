import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ================================================================================
# 원본 데이터 루트 경로
INPUT_ROOT_DIR = r'Z:\SamsungSTF\Processed_Data\Parking\classified'

# 결과 파일을 저장할 경로 및 이름
OUTPUT_EXCEL_PATH = r"Z:\SamsungSTF\Processed_Data\Parking\faulty_files_report.xlsx"
OUTPUT_LIST_CSV = r"Z:\SamsungSTF\Processed_Data\Parking\faulty_files_list.csv"

BMS_TEMP_COL = 'ext_temp' 


# ================================================================================
def check_zero_temp_file(file_path):
    try:
        
        try: 
            df = pd.read_csv(file_path, usecols=[BMS_TEMP_COL], encoding='utf-8')
        except: 
            try: df = pd.read_csv(file_path, usecols=[BMS_TEMP_COL], encoding='cp949')
            except: return None 

        if df.empty: return None

       
        df[BMS_TEMP_COL] = pd.to_numeric(df[BMS_TEMP_COL], errors='coerce').fillna(0)
        
        # -----------------------------------------------------------
        # ext_temp 값이 모두 0인지 확인
        # -----------------------------------------------------------
        if (df[BMS_TEMP_COL] == 0).all():
            
            # -------------------------------------------------------
            # 경로 및 파일명에서 차종과 단말기 번호 추출
            # -------------------------------------------------------
            try:
                rel_path = os.path.relpath(file_path, INPUT_ROOT_DIR)
                path_parts = rel_path.split(os.sep)
                
                # 폴더 구조상 첫 번째가 차종
                vehicle_type = path_parts[0] if len(path_parts) > 0 else "Unknown"
                
                # 단말기 번호 추출 로직
                filename = os.path.basename(file_path)
                name_parts = filename.split('_')
                
                device_id = "Unknown"
                #  파일명에서 10자리 이상 숫자 찾기
                for part in name_parts:
                    if part.isdigit() and len(part) >= 10:
                        device_id = part
                        break
                
                #  파일명에서 못 찾으면 폴더명(path_parts[1]) 사용
                if device_id == "Unknown" and len(path_parts) > 1:
                    device_id = path_parts[1]

            except:
                vehicle_type = "Unknown"
                device_id = "Unknown"

            return {
                '차종': vehicle_type,
                '단말기 번호': device_id,
                '파일 경로': file_path
            }
            
        return None

    except Exception:
        return None

# ================================================================================
# 전체 파일 수 카운트
# ================================================================================
def count_total_files_in_folder(vehicle, terminal_id):


    str_vehicle = str(vehicle).strip()
    
    # 엑셀/데이터에서 앞자리 0이 빠져있으므로 11자리로 맞춤 
    str_terminal = str(terminal_id).strip().zfill(11)

    # 경로 생성
    target_path = os.path.join(INPUT_ROOT_DIR, str_vehicle, str_terminal)

    csv_count = 0
    if os.path.exists(target_path):
        for root, dirs, files in os.walk(target_path):
            for file in files:
                if file.lower().endswith('.csv'):
                    csv_count += 1
    else:
        # 경로를 못 찾은 경우 재시도 (단말기 번호 앞뒤 공백 제거 후)
        target_path_raw = os.path.join(INPUT_ROOT_DIR, str_vehicle, str(terminal_id).strip())
        if os.path.exists(target_path_raw):
             for root, dirs, files in os.walk(target_path_raw):
                for file in files:
                    if file.lower().endswith('.csv'):
                        csv_count += 1
        else:
            return 0 

    return csv_count


# ================================================================================
def main():
    print("="*80)
    print(f"ext_temp가 모두 0인 파일 스캔 ")
    print(f"대상 경로: {INPUT_ROOT_DIR}")
    print("="*80)
    
    # 전체 파일 
    files_to_check = []
    for root, dirs, files in os.walk(INPUT_ROOT_DIR):
        for f in files:
            if f.lower().endswith('.csv'):
                files_to_check.append(os.path.join(root, f))
    
    if not files_to_check:
        print("검사할 파일이 없음.")
        return

    print(f"총 검사 대상 파일 : {len(files_to_check):,}개")

    # 병렬 처리로 이상치 찾기
    faulty_list = []
    max_workers = max(1, os.cpu_count() - 2)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(check_zero_temp_file, f) for f in files_to_check]
        
        for future in tqdm(as_completed(futures), total=len(files_to_check), desc="이상치 탐색 중"):
            res = future.result()
            if res:
                faulty_list.append(res)

    if not faulty_list:
        print("\n이상치 파일이 발견되지 않음")
        return

    # 이상치 데이터프레임 생성
    df_faulty = pd.DataFrame(faulty_list)
    
    # 파일 경로는 CSV로 저장 
    df_faulty.to_csv(OUTPUT_LIST_CSV, index=False, encoding='utf-8-sig')
    print(f"\n 이상치 상세 리스트 저장 : {OUTPUT_LIST_CSV}")

    print("\n" + "="*80)
    print(f"단말기별 전체 파일 수 집계 및 불량률 계산")
    print("="*80)

    # 차종/단말기 번호 기준으로 그룹화하여 '이상치 파일 수' 계산
    # as_index=False로 하여 컬럼을 살려둠
    summary_df = df_faulty.groupby(['차종', '단말기 번호']).size().reset_index(name='이상치 파일 수')

    
    total_counts = []
    for idx, row in tqdm(summary_df.iterrows(), total=len(summary_df), desc="전체 파일 카운트"):
        count = count_total_files_in_folder(row['차종'], row['단말기 번호'])
        total_counts.append(count)
    
    summary_df['실제 파일 수'] = total_counts

    # 불량률 계산
    # 실제 파일 수가 0인 경우 나눗셈 에러 방지
    summary_df['불량률(%)'] = summary_df.apply(
        lambda x: round((x['이상치 파일 수'] / x['실제 파일 수'] * 100), 2) if x['실제 파일 수'] > 0 else 0.0, 
        axis=1
    )

    # 컬럼 순서 정리 및 정렬
    summary_df = summary_df[['차종', '단말기 번호', '실제 파일 수', '이상치 파일 수', '불량률(%)']]
    summary_df = summary_df.sort_values(by='불량률(%)', ascending=False)

    # 최종 결과 엑셀 저장
    summary_df.to_excel(OUTPUT_EXCEL_PATH, index=False)

    print("\n" + "="*80)
    print(f" 생성된 파일 경로: {OUTPUT_EXCEL_PATH}")
    print("="*80)
    

if __name__ == "__main__":
    main()
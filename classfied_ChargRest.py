import os
import pandas as pd
import shutil
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 경고 메시지 숨김
warnings.filterwarnings('ignore')

# ================================================================================
# [설정 영역]
# ================================================================================
INPUT_ROOT_DIR = r'Z:\SamsungSTF\Processed_Data\Parking\segment'
OUTPUT_ROOT_DIR = r'Z:\SamsungSTF\Processed_Data\Parking\classified'
SOC_COL_NAME = 'soc'  
CHARGING_THRESHOLD = 10 


# ================================================================================
def process_single_file(file_path):
    try:
        # CSV 읽기
        try: df = pd.read_csv(file_path, encoding='utf-8')
        except:
            try: df = pd.read_csv(file_path, encoding='cp949')
            except: return 0 

        if df.empty or SOC_COL_NAME not in df.columns:
            return 0

        # SOC 차이 계산
        try:
            start_soc = df[SOC_COL_NAME].iloc[0]
            end_soc = df[SOC_COL_NAME].iloc[-1]
            soc_diff = end_soc - start_soc
        except:
            return 0

        # 분류 (Charging vs Rest)
        if soc_diff >= CHARGING_THRESHOLD:
            category = 'Charging'
            new_tag = 'charging'  # 파일명에 넣을 태그
        else:
            category = 'Rest'
            new_tag = 'rest'      # 파일명에 넣을 태그

        # 경로 계산
        relative_path = os.path.relpath(os.path.dirname(file_path), INPUT_ROOT_DIR)
        
        
        target_dir = os.path.join(OUTPUT_ROOT_DIR, relative_path, category)
        os.makedirs(target_dir, exist_ok=True)


        original_filename = os.path.basename(file_path)
        
        if 'parking' in original_filename:
            # 'parking'을 'charging' 또는 'rest'로 교체
            new_filename = original_filename.replace('parking', new_tag)
        else:
            # 만약 파일명에 'parking'이 없다면 맨 앞에 태그 붙이기
            new_filename = f"{new_tag}_{original_filename}"

        target_file_path = os.path.join(target_dir, new_filename)

        # 5. 파일 저장
        shutil.copy2(file_path, target_file_path)

        return 1 

    except Exception:
        return 0

# ================================================================================
# 메인 실행 함수
# ================================================================================
def main():
    print(f" 입력: {INPUT_ROOT_DIR}")
    print(f" 출력: {OUTPUT_ROOT_DIR}")
    
    files_to_process = []
    
    # 파일 탐색
    for root, dirs, files in os.walk(INPUT_ROOT_DIR):
        if os.path.abspath(OUTPUT_ROOT_DIR) in os.path.abspath(root):
            continue
        for f in files:
            if f.lower().endswith('.csv'):
                files_to_process.append(os.path.join(root, f))

    total_files = len(files_to_process)
    if total_files == 0:
        print(" 처리할 파일 없음")
        return

    print(f"{total_files:,}개 파일 병렬 처리 시작")

    max_workers = max(1, os.cpu_count() - 2)
    success_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, f) for f in files_to_process]
        
        for future in tqdm(as_completed(futures), total=total_files, desc="Processing"):
            success_count += future.result()

    print("-" * 50)
    print(f" 작업 완료")
    print(f"   - 성공: {success_count:,}개")  
    print(f"   - 실패: {total_files - success_count:,}개")

if __name__ == "__main__":
    main()
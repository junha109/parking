import os
import pandas as pd
import shutil
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 경고 메시지 숨김
warnings.filterwarnings('ignore')

# ================================================================================
# 설정 영역
# ================================================================================
INPUT_ROOT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\segment_10m"
OUTPUT_ROOT_DIR = r'Z:\SamsungSTF\Processed_Data\Parking\Classified_final'
SOC_COL_NAME = 'soc'  
CHARGING_THRESHOLD = 5 # SOC 차이 임계값 (충전으로 분류할 최소 SOC 증가량)

# 대상 차종 리스트 
TARGET_MODELS = ['EV6', 'Ioniq5'] 

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
        start_soc = df[SOC_COL_NAME].iloc[0]
        end_soc = df[SOC_COL_NAME].iloc[-1]
        soc_diff = end_soc - start_soc

        # 분류 
        category = 'Charging' if soc_diff >= CHARGING_THRESHOLD else 'Rest'
        new_tag = category.lower()

        # 경로 및 파일명 계산
        relative_path = os.path.relpath(os.path.dirname(file_path), INPUT_ROOT_DIR)
        target_dir = os.path.join(OUTPUT_ROOT_DIR, relative_path, category)
        os.makedirs(target_dir, exist_ok=True)

        original_filename = os.path.basename(file_path)
        new_filename = original_filename.replace('parking', new_tag) if 'parking' in original_filename else f"{new_tag}_{original_filename}"

        # 파일 복사
        shutil.copy2(file_path, os.path.join(target_dir, new_filename))
        return 1 

    except Exception:
        return 0

# ================================================================================
# 메인 실행 함수
# ================================================================================
def main():
    print(f" 입력 경로: {INPUT_ROOT_DIR}")
    print(f" 대상 차종: {', '.join(TARGET_MODELS)} ")
    
    files_to_process = []
    

    for model in TARGET_MODELS:
        model_path = os.path.join(INPUT_ROOT_DIR, model)
        
        if not os.path.exists(model_path):
            print(f" {model} 폴더를 찾을 수 없음")
            continue
            
        print(f"  - {model} 데이터 탐색 중")
        for root, dirs, files in os.walk(model_path):
            for f in files:
                if f.lower().endswith('.csv'):
                    files_to_process.append(os.path.join(root, f))

    total_files = len(files_to_process)
    if total_files == 0:
        print(" 처리할 파일이 없음")
        return

    print(f"총 {total_files:,}개 파일. 병렬 처리 시작")

    # 병렬 처리 실행
    max_workers = max(1, os.cpu_count() - 2)
    success_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, f) for f in files_to_process]
        for future in tqdm(as_completed(futures), total=total_files, desc="Processing"):
            success_count += future.result()

    print("-" * 50)
    print(f" 작업 완료: {success_count:,}개 성공 / {total_files - success_count:,}개 실패")

if __name__ == "__main__":
    main()
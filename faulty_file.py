import pandas as pd
import os
from tqdm import tqdm

# ================================================================================
# 삭제 목록이 있는 CSV 파일 경로
# ================================================================================
TARGET_LIST_CSV = r"Z:\SamsungSTF\Processed_Data\Parking\Faulty_ZeroTemp_Files_With_Month.csv"

# CSV 내에서 파일 경로가 적힌 컬럼명 
PATH_COL_NAME = '파일 경로' 

def main():
    # -----------------------------------------------------------
    # CSV 파일 로드 및 확인
    # -----------------------------------------------------------
    if not os.path.exists(TARGET_LIST_CSV):
        print(f"파일이 없음\n{TARGET_LIST_CSV}")
        return

    try:
        df = pd.read_csv(TARGET_LIST_CSV)
    except Exception as e:
        print(f"CSV 파일을 읽을 수 없음 {e}")
        return

    # 컬럼 존재 여부 확인
    if PATH_COL_NAME not in df.columns:
        print(f" CSV 파일 안에 '{PATH_COL_NAME}' 컬럼이 없음")
        print(f"현재 컬럼 목록: {df.columns.tolist()}")
        return

    # 삭제 대상 경로 리스트 추출
    files_to_delete = df[PATH_COL_NAME].dropna().tolist()
    total_files = len(files_to_delete)

    if total_files == 0:
        print("삭제할 파일 목록이 비어 있음")
        return

    # -----------------------------------------------------------
    # 파일 수 확인 
    # -----------------------------------------------------------
 
    print("=" * 60)
    print(f" 대상 리스트 파일: {os.path.basename(TARGET_LIST_CSV)}")
    print(f" 총 파일 수: {total_files:,} 개")
    print("-" * 60)
    
    confirm = input(" 맞으면 'delete' 입력 ")

    if confirm.lower() != 'delete':
        print("\n 삭제 취소")
        return

    # -----------------------------------------------------------
    # 파일 삭제
    # -----------------------------------------------------------
    print(f"\n삭제 시작")
    
    deleted_count = 0
    error_count = 0
    not_found_count = 0

    for file_path in tqdm(files_to_delete, desc="Deleting"):
        try:
            if os.path.exists(file_path):
                os.remove(file_path) # 파일 삭제
                deleted_count += 1
            else:
                not_found_count += 1
        except Exception as e:
            print(f"\n삭제 실패 {file_path} : {e}")
            error_count += 1

    # -----------------------------------------------------------
    # 결과 리포트
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f" 삭제 : {deleted_count:,} 개")
    print(f" 파일이 이미 없음  : {not_found_count:,} 개 ")
    print(f" 삭제 실패   : {error_count:,} 개")
    print("=" * 60)

if __name__ == "__main__":
    main()
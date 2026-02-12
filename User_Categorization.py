import pandas as pd

# ================================================================================
# 설정 및 데이터 로드
# ================================================================================
INPUT_CSV = r"Z:\SamsungSTF\Processed_Data\Parking\code\Clustering_8th\BMS_Weather_Clustering_Final_Labels4.csv"
OUTPUT_USER_CSV = "User_Categorization2.csv"

def main():
    print(f"데이터 로드 중 {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"{INPUT_CSV} 파일을 찾을 수 없음")
        return

    #  file_name 컬럼의 앞 11자리를 단말기 번호로 추출
    if 'file_name' in df.columns:
        df['device_id'] = df['file_name'].astype(str).str[:11]
    else:
        print("'file_name' 컬럼이 존재하지 않음")
        return

    # ================================================================================
    # 클러스터 그룹 매핑 (K=5)
    # ================================================================================
    # Cluster 2, 3 -> 실내
    # Cluster 0, 1 -> 실외
    # Cluster 4    -> 애매
    group_mapping = {
        4: '실내', 3: '실내',
        0: '실외', 2: '실외',
        1: '애매'
    }
    
    # cluster_k5 컬럼 확인 후 매핑
    if 'cluster_k5' in df.columns:
        df['parking_type'] = df['cluster_k5'].map(group_mapping)
    else:
        print("'cluster_k5' 컬럼을 찾을 수 없음")
        return

    # ================================================================================
    # 유저별(차종 + 단말기 번호) 그룹
    # ================================================================================
    print("유저 분류 및 비율 계산 중")
    
    # 유저별(차종, 단말기) 각 parking_type의 개수 계산
    pivot_table = df.pivot_table(index=['vehicle', 'device_id'], 
                                 columns='parking_type', 
                                 aggfunc='size', 
                                 fill_value=0)
    
    # 전체 세그먼트 합계
    pivot_table['total_segments'] = pivot_table.sum(axis=1)

    # 유저별로 가장 많이 나타난 그룹 찾기
    available_groups = [g for g in ['실내', '실외', '애매'] if g in pivot_table.columns]
    pivot_table['final_class'] = pivot_table[available_groups].idxmax(axis=1)

    # 최종 분류된 그룹의 비율(%) 계산
    pivot_table['percentage'] = pivot_table.apply(
        lambda x: (x[x['final_class']] / x['total_segments']) * 100, axis=1
    )

    # ================================================================================
    # 결과 정리 및 저장
    # ================================================================================
    final_df = pivot_table[['final_class', 'percentage']].reset_index()
    final_df.columns = ['차종', '단말기 번호', '분류 결과', '비율(%)']

    # CSV 저장
    final_df.to_csv(OUTPUT_USER_CSV, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print(f"완료 : {OUTPUT_USER_CSV}")
    print(f"분석 단말기 수: {len(final_df):,}개")
    print("="*60)

if __name__ == "__main__":
    main()
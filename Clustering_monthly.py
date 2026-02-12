import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ================================================================================
# 설정 및 데이터 로드
# ================================================================================
INPUT_CSV = r"Z:\SamsungSTF\Processed_Data\Parking\code\BMS_Weather_Temp_Average_lin4.csv" 
OUTPUT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\code\Clus_monthly2"
# 결과 파일명 수정
DEVICE_MONTH_STATS_CSV = "Device_Monthly_Cluster_Counts_K5_2.csv"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"폴더 생성 완료: {OUTPUT_DIR}")

    print(f"데이터 로드 중: {INPUT_CSV}")
    try:
        # 데이터 로드 시 'time' 컬럼을 datetime으로 변환
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"{INPUT_CSV} 파일을 찾을 수 없음")
        return

    # 1. 정보 추출 (단말기 번호 및 월 정보)
    # 단말기 번호 (앞 11자리)
    df['device_id'] = df['file_name'].astype(str).str[:11]
    
    # 월(Month) 정보 추출 (YYYY-MM 형식)
    if 'time' in df.columns:
        df['month'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m')
    else:
        # 파일명 구조(01241225206_202301...)에서 날짜 부분 추출 시도
        df['month'] = df['file_name'].astype(str).str.split('_').str[1].str[:4] + "-" + \
                      df['file_name'].astype(str).str.split('_').str[1].str[4:6]

    # 2. 클러스터링 전처리 및 실행 (K=5)
    features = ['ext_temp_avg', 'weather_temp_avg', 'temp_diff'] #
    df_clean = df.dropna(subset=features).copy()
    
    if df_clean.empty:
        print("분석할 데이터가 없음")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])

    print("K=5 클러스터링 수행 중")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_clean['cluster_k5'] = kmeans.fit_predict(X_scaled)

    # --------------------------------------------------------------------------------
    # 단말기별 월별 클러스터 세그먼트 수 집계
    # --------------------------------------------------------------------------------
    # image_584965.png 스타일의 VID_month 생성
    df_clean['VID_month'] = df_clean['device_id'] + "_" + df_clean['month']

    # 피벗 테이블 생성: 인덱스는 VID_month, 컬럼은 클러스터 번호
    monthly_stats = df_clean.pivot_table(index=['vehicle', 'VID_month'], 
                                        columns='cluster_k5', 
                                        aggfunc='size', 
                                        fill_value=0)
    
    # 컬럼명 정리 (Cluster_0, Cluster_1, ...)
    monthly_stats.columns = [f'Cluster_{col}' for col in monthly_stats.columns]
    
    # 해당 월에 가장 많이 나타난 클러스터 결과 추가
    monthly_stats['Main_Cluster'] = monthly_stats.idxmax(axis=1)
    # 해당 월의 총 세그먼트 수 추가
    monthly_stats['Total_Segments'] = monthly_stats.sum(axis=1, numeric_only=True)
    
    final_stats = monthly_stats.reset_index()
    
    # 결과 저장
    save_path = os.path.join(OUTPUT_DIR, DEVICE_MONTH_STATS_CSV)
    final_stats.to_csv(save_path, index=False, encoding='utf-8-sig')

# --------------------------------------------------------------------------------
    # 시각화 (기존 스타일 유지)
    # --------------------------------------------------------------------------------
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

# Scatter Plot (Cluster 분포 확인)
    plt.figure(figsize=(12, 10))
    
    # 그래프를 변수 'ax'에 할당합니다.
    ax = sns.scatterplot(data=df_clean, x='weather_temp_avg', y='ext_temp_avg', 
                        hue='cluster_k5', palette='viridis', alpha=0.4, s=10)
    
    # 생성된 범례(legend)를 직접 제거합니다.
    if ax.legend_ is not None:
        ax.legend_.remove()
    
    # === 글씨 크기 조절 코드 ===
    plt.xlabel('Weather Average Temperature (℃)', fontsize=25)
    plt.ylabel('BMS External Temperature Average (℃)', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # 1:1 Reference Line
    min_val, max_val = df_clean['weather_temp_avg'].min(), df_clean['weather_temp_avg'].max()
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    # 중심점 표시 
    centroids = df_clean.groupby('cluster_k5')[['weather_temp_avg', 'ext_temp_avg']].mean()
    for cluster_id, pos in centroids.iterrows():
        plt.text(pos['weather_temp_avg'], pos['ext_temp_avg'], str(int(cluster_id)), 
                 fontsize=22, fontweight='bold', color='black', ha='center', va='center') 

    plt.title('Indoor/Outdoor Parking USER Clustering', fontsize=30, pad=20)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Monthly_Analysis_Scatter_K5.png'), dpi=300)
    plt.close()

    print("\n" + "="*60)
    print(f"집계 완료: {save_path}")
    print("="*60)

if __name__ == "__main__":

    main()

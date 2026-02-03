import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ================================================================================
# 설정 및 데이터 로드
# ================================================================================
INPUT_CSV = "BMS_Weather_Temp_Average_lin.csv" 
OUTPUT_LABELED_CSV = "BMS_Weather_Clustering_Final_Labels.csv"
K_LIST = [3, 4, 5, 6]  
# 출력 경로 지정 
OUTPUT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\code\Clustering_5th"

def main():
    #  출력 디렉토리가 없으면 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"폴더 생성 완료: {OUTPUT_DIR}")

    print(f"데이터 로드 중: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"{INPUT_CSV} 파일을 찾을 수 없음")
        return

    # 클러스터링에 사용할 특성 선택
    features = ['ext_temp_avg', 'weather_temp_avg', 'temp_diff']
    
    # 분석용 데이터 정제 (결측치 제거)
    df_clean = df.dropna(subset=features).copy()
    
    if df_clean.empty:
        print("분석할 유효한 데이터가 없음")
        return

    # 데이터 스케일링 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])

    # 시각화 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # K-Means 루프 수행 및 라벨 기록
    print(f"클러스터링 및 결과 기록 (K={K_LIST})")
    
    for k in K_LIST:
        print(f"\nK={k} 분석 중")
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        df_clean[f'cluster_k{k}'] = cluster_labels

        # Scatter Plot 저장 경로 적용
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=df_clean, x='weather_temp_avg', y='ext_temp_avg', 
                        hue=f'cluster_k{k}', style='status', palette='viridis', alpha=0.4, s=10)
        
        min_val = min(df_clean['weather_temp_avg'].min(), df_clean['ext_temp_avg'].min())
        max_val = max(df_clean['weather_temp_avg'].max(), df_clean['ext_temp_avg'].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        
        plt.title(f'온도 기반 클러스터링 분석 (K={k})', fontsize=20)
        save_path_scatter = os.path.join(OUTPUT_DIR, f'Clustering_Result_K{k}.png')
        plt.savefig(save_path_scatter, dpi=300, bbox_inches='tight')
        plt.close()

        # Boxplot 저장 경로 적용
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_clean, x=f'cluster_k{k}', y='temp_diff', palette='viridis')
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'K={k} 클러스터별 온도 차이(BMS-기온) 분포', fontsize=20)
        save_path_box = os.path.join(OUTPUT_DIR, f'Cluster_TempDiff_Boxplot_K{k}.png')
        plt.savefig(save_path_box, dpi=300, bbox_inches='tight')
        plt.close()

        # 요약 리포트 저장 경로 적용
        summary = df_clean.groupby(f'cluster_k{k}').agg({
            'ext_temp_avg': ['mean', 'std'],
            'weather_temp_avg': ['mean', 'std'],
            'temp_diff': ['mean', 'min', 'max'],
            'file_name': 'count'
        })
        summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
        save_path_summary = os.path.join(OUTPUT_DIR, f'Cluster_Summary_Report_K{k}.csv')
        summary.to_csv(save_path_summary, encoding='utf-8-sig')

    # 최종 라벨 결과 저장 경로 적용
    final_save_path = os.path.join(OUTPUT_DIR, OUTPUT_LABELED_CSV)
    df_clean.to_csv(final_save_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print(f"클러스터링 결과 저장 완료 -> {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
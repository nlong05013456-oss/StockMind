import pandas as pd
import numpy as np
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering(clean_data_path="data/processed/clean_retail_data.csv", train_cutoff="2023-06-30"):
    print("[*] Đang thực hiện phân cụm nhu cầu sản phẩm (Fit CHỈ trên dữ liệu Train quá khứ)...")
    
    if not os.path.exists(clean_data_path):
        raise FileNotFoundError(f"Không tìm thấy file: {clean_data_path}")
        
    df = pd.read_csv(clean_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Chỉ dùng khoảng thời gian Train để tính toán đặc trưng phân cụm
    train_df = df[df['Date'] <= train_cutoff]
    
    sku_stats = train_df.groupby("SKU_ID").agg(
        Avg_Daily_Sales=("Units Sold", "mean"),
        Sales_Std=("Units Sold", "std"),
        Active_Days_Pct=("Units Sold", lambda x: (x > 0).mean())
    ).reset_index()
    
    sku_stats["Sales_Std"] = sku_stats["Sales_Std"].fillna(0)
    sku_stats["Sales_CV"] = sku_stats["Sales_Std"] / (sku_stats["Avg_Daily_Sales"] + 1e-5)
    
    features = ["Avg_Daily_Sales", "Sales_Std", "Sales_CV", "Active_Days_Pct"]
    
    # 2. Fit Scaler và KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sku_stats[features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    raw_clusters = kmeans.fit_predict(X_scaled)
    sku_stats["Raw_Cluster"] = raw_clusters
    
    # 3. Sắp xếp lại Cụm theo doanh số trung bình tăng dần (Deterministic Sorting)
    cluster_order = (
        sku_stats.groupby("Raw_Cluster")["Avg_Daily_Sales"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    
    # Map raw_cluster -> 0 (Slow Moving), 1 (Medium), 2 (Fast Moving)
    label_map = {old_id: new_id for new_id, old_id in enumerate(cluster_order)}
    sku_stats["Cluster"] = sku_stats["Raw_Cluster"].map(label_map)
    
    segment_names = {
        0: "Slow Moving",
        1: "Medium",
        2: "Fast Moving"
    }
    sku_stats["Segment"] = sku_stats["Cluster"].map(segment_names)
    
    # Đổi tên cột hiển thị cho sạch
    sku_stats = sku_stats.rename(columns={"Sales_Std": "Sales_Volatility"})
    
    # 4. Lưu Artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    joblib.dump(kmeans, "models/clustering_model.pkl")
    joblib.dump(scaler, "models/cluster_scaler.pkl")
    
    output_cols = ["SKU_ID", "Cluster", "Segment", "Avg_Daily_Sales", "Sales_Volatility", "Sales_CV", "Active_Days_Pct"]
    sku_stats[output_cols].to_csv("data/processed/product_segments.csv", index=False)
    
    print("✅ Phân cụm hoàn tất!")
    print(sku_stats.groupby("Segment")["Avg_Daily_Sales"].agg(["count", "mean", "std"]))
    
    return sku_stats

if __name__ == "__main__":
    perform_clustering()
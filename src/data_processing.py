import pandas as pd
import numpy as np
import os

def load_and_clean_raw(file_path):
    print("[*] Nạp và làm sạch dữ liệu thô...")
    df = pd.read_csv(file_path)
    
    # 1. Loại bỏ trùng lặp và bản ghi âm/bất hợp lý
    df = df.drop_duplicates()
    df = df[(df['Units Sold'] >= 0) & (df['Inventory Level'] >= 0) & (df['Price'] > 0)]
    
    # 2. Tạo mã định danh Multi-Series SKU_ID (Store ID + Product ID)
    df['SKU_ID'] = df['Store ID'].astype(str) + "_" + df['Product ID'].astype(str)
    
    # 3. Chuyển đổi thời gian và lọc dữ liệu (2022 đến 2023)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'] >= '2022-01-01') & (df['Date'] <= '2023-12-31')]
    
    # Sắp xếp theo chuỗi thời gian cho từng SKU
    df = df.sort_values(by=['SKU_ID', 'Date']).reset_index(drop=True)
    print(f"📊 Dữ liệu gốc sau khi chọn: {df.shape[0]} bản ghi trên {df['SKU_ID'].nunique()} SKU_ID.")
    return df

def fit_transform_outliers(df, train_cutoff='2023-06-30'):
    print("[*] Xử lý Outlier (Fit ngưỡng Quantile 0.99 CHỈ trên tập Train)...")
    
    # Chỉ dùng tập Train để tính ngưỡng quantile per category (chống rò rỉ dữ liệu)
    train_mask = df['Date'] <= train_cutoff
    train_df = df[train_mask]
    
    cat_upper_limits = train_df.groupby('Category')['Units Sold'].quantile(0.99).to_dict()
    
    # Global fallback quantile nếu category không có trong train
    global_upper_limit = train_df['Units Sold'].quantile(0.99)
    
    def clip_units(row):
        upper = cat_upper_limits.get(row['Category'], global_upper_limit)
        return min(row['Units Sold'], upper)
    
    df['Units Sold'] = df.apply(clip_units, axis=1)
    return df

def feature_engineering(df):
    print("[*] Thực hiện Feature Engineering (Leakage-Free)...")
    
    # 1. Các biến Lags (Doanh số quá khứ)
    df['Lag_1'] = df.groupby('SKU_ID')['Units Sold'].shift(1)
    df['Lag_7'] = df.groupby('SKU_ID')['Units Sold'].shift(7)
    df['Lag_14'] = df.groupby('SKU_ID')['Units Sold'].shift(14)
    
    # 2. Thống kê Rolling (chỉ nhìn về quá khứ nhờ shift(1))
    df['Rolling_Mean_7'] = df.groupby('SKU_ID')['Units Sold'].transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )
    df['Rolling_STD_7'] = df.groupby('SKU_ID')['Units Sold'].transform(
        lambda x: x.shift(1).rolling(window=7).std()
    )
    df['Rolling_Mean_30'] = df.groupby('SKU_ID')['Units Sold'].transform(
        lambda x: x.shift(1).rolling(window=30).mean()
    )
    df['Rolling_STD_30'] = df.groupby('SKU_ID')['Units Sold'].transform(
        lambda x: x.shift(1).rolling(window=30).std()
    )
    
    # 3. Tương tác giá & Khuyến mãi
    df['Promo_Effect'] = df['Discount'] / df['Price']
    
    # 4. Biến thời gian / Lịch
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # 5. Log Transformation biến mục tiêu
    df['Target_Log'] = np.log1p(df['Units Sold'])
    
    # Loại bỏ các dòng đầu tiên bị NaN do Lag/Rolling 30 ngày
    clean_df = df.dropna(subset=['Lag_1', 'Lag_7', 'Lag_14', 'Rolling_Mean_7', 'Rolling_Mean_30']).copy()
    print(f"✅ Dữ liệu sau Feature Engineering: {clean_df.shape[0]} bản ghi.")
    return clean_df

def save_processed_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Đã lưu dữ liệu làm sạch tại: {output_path}")

if __name__ == "__main__":
    RAW_DATA = 'data/unprocessed/retail_store_inventory.csv'
    CLEAN_DATA = 'data/processed/clean_retail_data.csv'
    
    if os.path.exists(RAW_DATA):
        df = load_and_clean_raw(RAW_DATA)
        df = fit_transform_outliers(df, train_cutoff='2023-06-30')
        df = feature_engineering(df)
        save_processed_data(df, CLEAN_DATA)
    else:
        print(f"[!] Lỗi: Không tìm thấy file {RAW_DATA}")
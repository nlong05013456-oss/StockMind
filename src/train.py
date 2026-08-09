import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from clustering import perform_clustering

class NaiveModelWrapper:
    """Wrapper đơn giản cho mô hình Naive (ŷ_t = Lag_1) để tương thích interface predict"""
    def predict(self, X):
        return np.log1p(X['Lag_1'].values)

def train_and_benchmark():
    print("[*] Đang khởi động tiến trình Huấn luyện & Benchmark Mô hình...")
    
    clean_path = 'data/processed/clean_retail_data.csv'
    segment_path = 'data/processed/product_segments.csv'
    
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Chưa có file dữ liệu {clean_path}. Hãy chạy data_processing.py trước.")
        
    if not os.path.exists(segment_path):
        print("[!] Chưa thấy product_segments.csv, tự động chạy phân cụm...")
        perform_clustering(clean_path)
        
    df = pd.read_csv(clean_path)
    segments = pd.read_csv(segment_path)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.merge(segments[['SKU_ID', 'Cluster', 'Segment']], on='SKU_ID', how='left')
    df = df.sort_values(['SKU_ID', 'Date']).reset_index(drop=True)
    
    # 1. Định nghĩa Features Matrix
    features_num = [
        'Price', 'Discount', 'Lag_1', 'Lag_7', 'Lag_14',
        'Rolling_Mean_7', 'Rolling_STD_7', 'Rolling_Mean_30', 'Rolling_STD_30',
        'Promo_Effect', 'Month', 'DayOfWeek', 'Is_Weekend', 'Cluster'
    ]
    features_cat = ['Region', 'Weather Condition']
    
    X = pd.get_dummies(df[features_num + features_cat], columns=features_cat, drop_first=True)
    y_log = np.log1p(df['Units Sold'])
    y_true_all = df['Units Sold'].values
    
    # 2. Phân chia Temporal Split (Train: <= 2023-06-30 | Test: > 2023-06-30)
    train_mask = df['Date'] <= '2023-06-30'
    test_mask = df['Date'] > '2023-06-30'
    
    X_train, y_train_log = X[train_mask], y_log[train_mask]
    X_test, y_test_log = X[test_mask], y_log[test_mask]
    y_test_true = y_true_all[test_mask]
    
    print(f"📊 Tập Train: {X_train.shape[0]} dòng | Tập Test: {X_test.shape[0]} dòng.")
    
    # 3. Benchmark các mô hình
    benchmark_results = {}
    models_dict = {}
    
    # --- Model 1: Naive Baseline (ŷ = Lag_1) ---
    y_pred_naive = X_test['Lag_1'].values
    mae_naive = mean_absolute_error(y_test_true, y_pred_naive)
    rmse_naive = np.sqrt(mean_squared_error(y_test_true, y_pred_naive))
    benchmark_results['Naive'] = {'mae': mae_naive, 'rmse': rmse_naive}
    models_dict['Naive'] = NaiveModelWrapper()
    print(f"   [1] Naive Baseline  -> MAE: {mae_naive:.2f} | RMSE: {rmse_naive:.2f}")
    
    # --- Model 2: Random Forest ---
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train_log)
    y_pred_rf_log = rf_model.predict(X_test)
    y_pred_rf = np.expm1(y_pred_rf_log)
    mae_rf = mean_absolute_error(y_test_true, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test_true, y_pred_rf))
    benchmark_results['RandomForest'] = {'mae': mae_rf, 'rmse': rmse_rf}
    models_dict['RandomForest'] = rf_model
    print(f"   [2] Random Forest   -> MAE: {mae_rf:.2f} | RMSE: {rmse_rf:.2f}")
    
    # --- Model 3: XGBoost ---
    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train_log)
    y_pred_xgb_log = xgb_model.predict(X_test)
    y_pred_xgb = np.expm1(y_pred_xgb_log)
    mae_xgb = mean_absolute_error(y_test_true, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test_true, y_pred_xgb))
    benchmark_results['XGBoost'] = {'mae': mae_xgb, 'rmse': rmse_xgb}
    models_dict['XGBoost'] = xgb_model
    print(f"   [3] XGBoost Regressor -> MAE: {mae_xgb:.2f} | RMSE: {rmse_xgb:.2f}")
    
    # 4. Tự động chọn mô hình có MAE thấp nhất
    best_model_name = min(benchmark_results, key=lambda k: benchmark_results[k]['mae'])
    best_model = models_dict[best_model_name]
    print(f"🏆 MÔ HÌNH XUẤT SẮC NHẤT: {best_model_name} (MAE: {benchmark_results[best_model_name]['mae']:.2f})")
    
    # 5. Tính Residual Error (Test Set) của mô hình xuất sắc nhất
    if best_model_name == 'Naive':
        y_best_pred = y_pred_naive
    elif best_model_name == 'RandomForest':
        y_best_pred = y_pred_rf
    else:
        y_best_pred = y_pred_xgb
        
    test_df = df[test_mask].copy()
    test_df['y_true'] = y_test_true
    test_df['y_pred'] = y_best_pred
    test_df['residual'] = test_df['y_true'] - test_df['y_pred']
    
    # Global Error Metrics
    mae_global = mean_absolute_error(test_df['y_true'], test_df['y_pred'])
    rmse_global = np.sqrt(mean_squared_error(test_df['y_true'], test_df['y_pred']))
    error_std_global = np.std(test_df['residual'])
    
    # Cluster-level Error Metrics
    cluster_metrics = {}
    for seg, group in test_df.groupby('Segment'):
        mae_seg = mean_absolute_error(group['y_true'], group['y_pred'])
        rmse_seg = np.sqrt(mean_squared_error(group['y_true'], group['y_pred']))
        std_seg = np.std(group['residual'])
        cluster_metrics[seg] = {
            'mae': float(mae_seg),
            'rmse': float(rmse_seg),
            'error_std': float(std_seg),
            'count': int(len(group))
        }
        
    # SKU-level Error Metrics
    sku_metrics = {}
    for sku, group in test_df.groupby('SKU_ID'):
        if len(group) >= 5: # Ít nhất 5 mẫu quan sát trong tập test
            mae_sku = mean_absolute_error(group['y_true'], group['y_pred'])
            rmse_sku = np.sqrt(mean_squared_error(group['y_true'], group['y_pred']))
            std_sku = np.std(group['residual'])
            sku_metrics[sku] = {
                'mae': float(mae_sku),
                'rmse': float(rmse_sku),
                'error_std': float(std_sku),
                'count': int(len(group))
            }
            
    # 6. Lưu Artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/final_model.pkl')
    
    business_logic = {
        'model_benchmark': benchmark_results,
        'best_model_name': best_model_name,
        'global_metrics': {
            'mae': float(mae_global),
            'rmse': float(rmse_global),
            'error_std': float(error_std_global)
        },
        'cluster_metrics': cluster_metrics,
        'sku_metrics': sku_metrics,
        'service_levels': {
            0.90: 1.282,
            0.95: 1.645,
            0.99: 2.326
        },
        'fallback_error_hierarchy': ['sku', 'cluster', 'global'],
        'target_columns': X.columns.tolist()
    }
    
    joblib.dump(business_logic, 'models/business_logic.pkl')
    print("✅ Đã hoàn tất huấn luyện và lưu business_logic.pkl thành công!")

if __name__ == "__main__":
    train_and_benchmark()
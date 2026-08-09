import numpy as np
import pandas as pd

def get_error_std_with_fallback(sku_id, cluster_name, business_logic):
    """
    Cơ chế Fallback Sai số: SKU Level -> Cluster Level -> Global Level
    """
    sku_metrics = business_logic.get('sku_metrics', {})
    cluster_metrics = business_logic.get('cluster_metrics', {})
    global_metrics = business_logic.get('global_metrics', {})
    
    if sku_id in sku_metrics and sku_metrics[sku_id].get('count', 0) >= 5:
        return sku_metrics[sku_id]['error_std'], "SKU Level"
    
    if cluster_name in cluster_metrics:
        return cluster_metrics[cluster_name]['error_std'], f"Cluster Level ({cluster_name})"
        
    return global_metrics.get('error_std', 10.0), "Global Level"

def calculate_inventory_recommendation(
    daily_forecast: float,
    planning_horizon: int,
    lead_time: int,
    service_level: float,
    current_inventory: float,
    sku_id: str,
    cluster_name: str,
    business_logic: dict
) -> dict:
    """
    Tầng 2 & 3: Tính toán Hoạch định Tồn kho & Quyết định Nhập hàng
    """
    # 1. Tra cứu Z-score từ Service Level
    service_levels_map = business_logic.get('service_levels', {0.90: 1.282, 0.95: 1.645, 0.99: 2.326})
    z_score = service_levels_map.get(service_level, 1.645)
    
    # 2. Tra cứu Error Uncertainty theo Fallback Hierarchy
    error_std, error_source = get_error_std_with_fallback(sku_id, cluster_name, business_logic)
    
    # 3. Expected Demand trong kỳ hoạch định
    expected_demand = daily_forecast * planning_horizon
    
    # 4. Safety Stock chuẩn: SS = Z * error_std * sqrt(LeadTime)
    safety_stock = z_score * error_std * np.sqrt(lead_time)
    safety_days = safety_stock / daily_forecast if daily_forecast > 0 else 0.0
    
    # 5. Target Inventory & Recommended Order
    target_inventory = expected_demand + safety_stock
    recommended_order = max(0, int(round(target_inventory - current_inventory)))
    
    # 6. Days of Inventory (DOI) & Unified Risk Evaluation
    doi = current_inventory / daily_forecast if daily_forecast > 0 else 0.0
    
    if doi < lead_time:
        risk_status = "Stock-Out Risk"
        risk_icon = "🔴"
        risk_msg = f"CẢNH BÁO HẾT HÀNG! Tồn kho hiện tại ({doi:.1f} ngày) ít hơn Lead Time chờ hàng ({lead_time} ngày)."
    elif doi < (lead_time + safety_days):
        risk_status = "Low Stock"
        risk_icon = "🟡"
        risk_msg = f"CẢNH BÁO TỒN KHO THẤP! Tồn kho ({doi:.1f} ngày) đã chạm ngưỡng Safety Coverage ({lead_time + safety_days:.1f} ngày)."
    elif doi <= (planning_horizon * 1.5):
        risk_status = "Optimal"
        risk_icon = "🟢"
        risk_msg = f"TỒN KHO TỐI ƯU! Tồn kho hiện tại ({doi:.1f} ngày) nằm trong khoảng an toàn của chu kỳ hoạch định."
    else:
        risk_status = "Overstocked"
        risk_icon = "🔵"
        risk_msg = f"CẢNH BÁO DƯ THỪA TỒN KHO! Tồn kho ({doi:.1f} ngày) cao hơn 1.5 lần chu kỳ hoạch định ({planning_horizon} ngày)."
        
    return {
        "daily_forecast": daily_forecast,
        "planning_horizon": planning_horizon,
        "expected_demand": expected_demand,
        "lead_time": lead_time,
        "service_level": service_level,
        "z_score": z_score,
        "error_std": error_std,
        "error_source": error_source,
        "safety_stock": safety_stock,
        "safety_days": safety_days,
        "target_inventory": target_inventory,
        "current_inventory": current_inventory,
        "recommended_order": recommended_order,
        "doi": doi,
        "risk_status": risk_status,
        "risk_icon": risk_icon,
        "risk_msg": risk_msg
    }

def run_what_if_simulation(model, product_row: pd.DataFrame, target_columns: list, promo_discount: float) -> float:
    """
    Giả lập Kịch bản (What-If Simulation): Thay đổi Discount -> Tính lại Forecast
    """
    row_copy = product_row.copy()
    row_copy['Discount'] = promo_discount
    if 'Price' in row_copy.columns and row_copy['Price'].values[0] > 0:
        row_copy['Promo_Effect'] = promo_discount / row_copy['Price'].values[0]
        
    X_sim = pd.get_dummies(row_copy)
    X_sim = X_sim.reindex(columns=target_columns, fill_value=0)
    
    pred_log = model.predict(X_sim)[0]
    return float(np.expm1(pred_log))

def generate_explanation_md(res: dict) -> str:
    """
    Tạo văn bản giải thích toán học cho quyết định nhập hàng (Explainable Decision Support)
    """
    md = f"""
### 💡 Tại sao Hệ thống Khuyên Nhập **{res['recommended_order']} sản phẩm**?

1. **Nhu cầu Dự báo ({res['planning_horizon']} ngày)**: 
   $$\\text{{Expected Demand}} = {res['daily_forecast']:.1f} \\text{{ sp/ngày}} \\times {res['planning_horizon']} \\text{{ ngày}} = {res['expected_demand']:.1f} \\text{{ sp}}$$

2. **Tồn kho An toàn (Safety Stock)**:
   * **Service Level ({res['service_level']:.0%})**: $Z = {res['z_score']:.3f}$
   * **Độ lệch chuẩn sai số ({res['error_source']})**: $\\sigma = {res['error_std']:.2f}$ sp
   * **Lead Time**: ${res['lead_time']}$ ngày
   $$\\text{{Safety Stock}} = {res['z_score']:.3f} \\times {res['error_std']:.2f} \\times \\sqrt{{{res['lead_time']}}} = {res['safety_stock']:.1f} \\text{{ sp}}$$

3. **Mục tiêu Tồn kho (Target Inventory)**:
   $$\\text{{Target Inventory}} = {res['expected_demand']:.1f} + {res['safety_stock']:.1f} = {res['target_inventory']:.1f} \\text{{ sp}}$$

4. **Khuyến nghị Đặt hàng**:
   $$\\text{{Recommended Order}} = \\max(0, {res['target_inventory']:.1f} - {res['current_inventory']}) = \\mathbf{{{res['recommended_order']}}} \\text{{ sản phẩm}}$$
    """
    return md

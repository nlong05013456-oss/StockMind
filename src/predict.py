import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from inventory_engine import (
    calculate_inventory_recommendation,
    run_what_if_simulation,
    generate_explanation_md
)

# =========================================================
# 1. Cấu hình Trang & Branding
# =========================================================
st.set_page_config(
    page_title="StockMind AI — Inventory Decision Support System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 StockMind AI")
st.markdown("### **AI-Powered Multi-Series Demand Forecasting & Inventory Decision Support System**")
st.caption("✨ *Predict Demand. Optimize Inventory. Make Smarter Decisions.*")
st.markdown("---")

# =========================================================
# 2. Nạp Dữ liệu & Mô hình
# =========================================================
@st.cache_resource
def load_system_artifacts():
    model = joblib.load("models/final_model.pkl")
    logic = joblib.load("models/business_logic.pkl")
    df = pd.read_csv("data/processed/clean_retail_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    clusters = pd.read_csv("data/processed/product_segments.csv")
    return model, logic, df, clusters

try:
    model, logic, df, clusters = load_system_artifacts()
except Exception as e:
    st.error(f"❌ Lỗi nạp mô hình hoặc dữ liệu: {e}. Vui lòng đảm bảo đã chạy `data_processing.py`, `clustering.py`, `train.py`.")
    st.stop()

# =========================================================
# 3. Sidebar Configuration
# =========================================================
st.sidebar.title("🧠 StockMind AI")
st.sidebar.header("⚙️ Decision Parameters")

# Chọn SKU_ID
sku_list = sorted(df["SKU_ID"].unique())
selected_sku = st.sidebar.selectbox("1. Select SKU_ID (Store_Product)", sku_list)

# Lấy dòng dữ liệu mới nhất của SKU được chọn
sku_history = df[df["SKU_ID"] == selected_sku].sort_values("Date")
latest_row = sku_history.tail(1)

category = latest_row["Category"].values[0] if not latest_row.empty else "Groceries"
default_lead_times = {
    "Groceries": 3,
    "Toys": 5,
    "Electronics": 7,
    "Furniture": 14,
    "Clothing": 5
}
suggested_lt = default_lead_times.get(category, 5)

# Tùy chỉnh tham số nghiệp vụ
planning_horizon = st.sidebar.selectbox("2. Planning Horizon (Days)", [7, 14, 30], index=1)
lead_time = st.sidebar.slider("3. Supplier Lead Time (Days)", min_value=1, max_value=30, value=suggested_lt)

service_level_label = st.sidebar.selectbox("4. Desired Service Level", ["90%", "95%", "99%"], index=1)
service_level_map = {"90%": 0.90, "95%": 0.95, "99%": 0.99}
service_level = service_level_map[service_level_label]

latest_inv = int(latest_row["Inventory Level"].values[0]) if not latest_row.empty else 100
current_inv = st.sidebar.slider("5. Current Inventory Level (Units)", min_value=0, max_value=2000, value=latest_inv)

# What-if promo discount slider
base_discount = float(latest_row["Discount"].values[0]) if not latest_row.empty else 0.0
st.sidebar.subheader("🧪 What-If Promotion Simulation")
sim_discount = st.sidebar.slider("Simulated Discount (%)", min_value=0, max_value=50, value=int(base_discount))

# =========================================================
# 4. Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "📦 Inventory Decision & What-If", 
    "🧠 Product Demand Segmentation", 
    "📊 Model Performance & Benchmarks"
])

# Merge thông tin Phân cụm
sku_segment_row = clusters[clusters["SKU_ID"] == selected_sku]
cluster_name = sku_segment_row["Segment"].values[0] if not sku_segment_row.empty else "Medium"

# =========================================================
# TAB 1 — INVENTORY DECISION & WHAT-IF
# =========================================================
with tab1:
    st.subheader(f"StockMind Decision Matrix for `{selected_sku}` (Category: {category} | Segment: {cluster_name})")
    
    if not latest_row.empty:
        # 1. Dự báo nhu cầu 1 ngày bằng Model
        X_input = pd.get_dummies(latest_row)
        X_input = X_input.reindex(columns=logic["target_columns"], fill_value=0)
        
        pred_log = model.predict(X_input)[0]
        daily_forecast = float(np.expm1(pred_log))
        
        # 2. Gửi sang Inventory Engine tính toán
        res = calculate_inventory_recommendation(
            daily_forecast=daily_forecast,
            planning_horizon=planning_horizon,
            lead_time=lead_time,
            service_level=service_level,
            current_inventory=current_inv,
            sku_id=selected_sku,
            cluster_name=cluster_name,
            business_logic=logic
        )
        
        # 3. KPI Metrics Top Bar
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Daily Forecast", f"{res['daily_forecast']:.1f} units")
        col2.metric(f"Demand ({res['planning_horizon']}d)", f"{res['expected_demand']:.0f} units")
        col3.metric("Safety Stock", f"{res['safety_stock']:.1f} units")
        col4.metric("Current Inventory", f"{res['current_inventory']} units")
        col5.metric("Days of Inventory", f"{res['doi']:.1f} days")
        
        st.markdown("---")
        
        # 4. Status Badge & Recommendation Box
        st_col1, st_col2 = st.columns([1, 2])
        
        with st_col1:
            st.markdown(f"### Status: {res['risk_icon']} **{res['risk_status']}**")
            if res['risk_status'] == "Stock-Out Risk":
                st.error(res['risk_msg'])
            elif res['risk_status'] == "Low Stock":
                st.warning(res['risk_msg'])
            elif res['risk_status'] == "Optimal":
                st.success(res['risk_msg'])
            else:
                st.info(res['risk_msg'])
                
        with st_col2:
            st.success(
                f"## 💡 StockMind Recommended Order: **{res['recommended_order']} units**\n"
                f"*Target Inventory: {res['target_inventory']:.0f} units | Safety Coverage: {res['safety_days']:.1f} days*"
            )
            
        # 5. Explainable Decision Breakdown Markdown
        st.markdown(generate_explanation_md(res))
        
        # 6. Plotly Calculation Waterfall / Bar Chart
        st.markdown("### 📊 Decision Waterfall Breakdown")
        fig_waterfall = go.Figure(go.Bar(
            x=["Expected Demand", "Safety Stock", "Target Inventory", "Current Inventory", "Recommended Order"],
            y=[res['expected_demand'], res['safety_stock'], res['target_inventory'], -res['current_inventory'], res['recommended_order']],
            text=[f"{res['expected_demand']:.0f}", f"+{res['safety_stock']:.1f}", f"={res['target_inventory']:.0f}", f"-{res['current_inventory']}", f"={res['recommended_order']}"],
            textposition="auto",
            marker_color=["#3366CC", "#FF9900", "#109618", "#DC3912", "#0099C6"]
        ))
        fig_waterfall.update_layout(
            title=f"StockMind Calculation Flow for {selected_sku}",
            height=380,
            yaxis_title="Units"
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # 7. What-If Simulation Comparison
        st.markdown("---")
        st.subheader("🧪 What-If Promotion Simulation Analysis")
        
        sim_daily_forecast = run_what_if_simulation(
            model=model,
            product_row=latest_row,
            target_columns=logic["target_columns"],
            promo_discount=sim_discount
        )
        
        sim_res = calculate_inventory_recommendation(
            daily_forecast=sim_daily_forecast,
            planning_horizon=planning_horizon,
            lead_time=lead_time,
            service_level=service_level,
            current_inventory=current_inv,
            sku_id=selected_sku,
            cluster_name=cluster_name,
            business_logic=logic
        )
        
        pct_change = ((sim_daily_forecast - daily_forecast) / daily_forecast) * 100 if daily_forecast > 0 else 0.0
        
        w_col1, w_col2, w_col3 = st.columns(3)
        w_col1.metric("Base Forecast (Discount " + f"{base_discount:.0f}%)", f"{daily_forecast:.1f} units/day")
        w_col2.metric("Simulated Forecast (Discount " + f"{sim_discount:.0f}%)", f"{sim_daily_forecast:.1f} units/day", delta=f"{pct_change:+.1f}%")
        w_col3.metric("Simulated Recommended Order", f"{sim_res['recommended_order']} units", delta=f"{sim_res['recommended_order'] - res['recommended_order']:+d} units")

# =========================================================
# TAB 2 — PRODUCT DEMAND SEGMENTATION
# =========================================================
with tab2:
    st.subheader("Product Demand Segmentation (K-Means Clustering)")
    st.markdown("Phân cụm sản phẩm dựa trên dữ liệu bán hàng quá khứ (Train set) thành 3 nhóm: **Slow Moving**, **Medium**, **Fast Moving**.")
    
    seg_col1, seg_col2, seg_col3 = st.columns(3)
    counts = clusters["Segment"].value_counts()
    
    seg_col1.metric("🐢 Slow Moving SKUs", f"{counts.get('Slow Moving', 0)}")
    seg_col2.metric("⚖️ Medium SKUs", f"{counts.get('Medium', 0)}")
    seg_col3.metric("🚀 Fast Moving SKUs", f"{counts.get('Fast Moving', 0)}")
    
    st.markdown("---")
    
    fig_seg = px.bar(
        clusters.groupby("Segment")["Avg_Daily_Sales"].mean().reset_index(),
        x="Segment",
        y="Avg_Daily_Sales",
        color="Segment",
        title="Average Daily Sales by Segment",
        color_discrete_map={"Slow Moving": "#9E9E9E", "Medium": "#FFB74D", "Fast Moving": "#66BB6A"}
    )
    st.plotly_chart(fig_seg, use_container_width=True)
    
    st.write("### 📋 SKU Demand Segmentation Table")
    search_query = st.text_input("🔍 Search SKU_ID:", "")
    if search_query:
        filtered_clusters = clusters[clusters["SKU_ID"].str.contains(search_query, case=False)]
    else:
        filtered_clusters = clusters
        
    st.dataframe(filtered_clusters, use_container_width=True)

# =========================================================
# TAB 3 — MODEL PERFORMANCE & BENCHMARKS
# =========================================================
with tab3:
    st.subheader("StockMind Model Performance & Temporal Test Benchmark")
    st.markdown("So sánh hiệu năng giữa các mô hình trên tập **Test theo thời gian** (`Date > 2023-06-30`).")
    
    best_name = logic.get("best_model_name", "RandomForest")
    benchmarks = logic.get("model_benchmark", {})
    
    bench_data = []
    for m_name, metrics in benchmarks.items():
        is_best = "🏆 YES" if m_name == best_name else "NO"
        bench_data.append({
            "Model": m_name,
            "MAE (Units)": f"{metrics['mae']:.2f}",
            "RMSE (Units)": f"{metrics['rmse']:.2f}",
            "Selected Production Model": is_best
        })
        
    st.write("### 🏆 Model Comparison Matrix")
    st.table(pd.DataFrame(bench_data))
    
    st.markdown("---")
    st.write("### 📊 Segment-level Error Breakdown")
    c_metrics = logic.get("cluster_metrics", {})
    c_data = []
    for seg_name, metrics in c_metrics.items():
        c_data.append({
            "Segment": seg_name,
            "MAE (Units)": f"{metrics['mae']:.2f}",
            "RMSE (Units)": f"{metrics['rmse']:.2f}",
            "Error Std (σ)": f"{metrics['error_std']:.2f}",
            "Test Observations": metrics['count']
        })
    st.table(pd.DataFrame(c_data))
    
    st.markdown("---")
    st.write(f"### 📈 Actual vs. Forecast Time Series for `{selected_sku}`")
    
    # Plot Actual vs Forecast for selected SKU on Test set
    test_sku_history = sku_history[sku_history["Date"] > "2023-06-30"].copy()
    if not test_sku_history.empty:
        X_sku_test = pd.get_dummies(test_sku_history)
        X_sku_test = X_sku_test.reindex(columns=logic["target_columns"], fill_value=0)
        
        preds_log = model.predict(X_sku_test)
        test_sku_history["Forecast"] = np.expm1(preds_log)
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=test_sku_history["Date"], y=test_sku_history["Units Sold"], mode='lines+markers', name='Actual Sales'))
        fig_ts.add_trace(go.Scatter(x=test_sku_history["Date"], y=test_sku_history["Forecast"], mode='lines', name='StockMind Forecast', line=dict(dash='dash', color='orange')))
        
        fig_ts.update_layout(title=f"Actual vs. StockMind Forecast Sales for {selected_sku} (Test Period)", xaxis_title="Date", yaxis_title="Units Sold")
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Không có dữ liệu test sau 2023-06-30 cho SKU này.")
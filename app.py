import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Cipla Smart Inventory Forecaster",
    layout="wide"
)

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------


@st.cache_resource
def load_models():
    # Wrap in try-except to catch missing file errors specifically
    try:
        encoder = joblib.load("encoder.pkl")
        model = joblib.load("lr_model.pkl")
        return encoder, model
    except FileNotFoundError:
        st.error("Model files not found! Please upload encoder.pkl and lr_model.pkl to GitHub.")
        st.stop()


encoder, model = load_models()

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown(
    """
    <h1 style='color:#1f4ed8'>💊 Cipla Smart Inventory Forecaster</h1>
    <p>AI-powered stock requirement prediction & inventory planning</p>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------
st.sidebar.header("⚙️ Controls")

strength_mg = st.sidebar.selectbox("Strength (mg)", [250, 500, 650])
unit_price = st.sidebar.number_input("Unit Price (₹)", value=12.5)

historical_sales_qty = st.sidebar.number_input("Historical Sales Qty", value=1800)
rolling_mean_3m_sales = st.sidebar.number_input("Rolling Mean (3M)", value=1750)
rolling_mean_6m_sales = st.sidebar.number_input("Rolling Mean (6M)", value=1650)

sales_growth_yoy = st.sidebar.slider("YoY Sales Growth", 0.0, 1.0, 0.12)
demand_volatility = st.sidebar.slider("Demand Volatility", 0.0, 1.0, 0.18)

lead_time_days = st.sidebar.number_input("Lead Time (Days)", value=14)
supplier_reliability_score = st.sidebar.slider("Supplier Reliability", 0.0, 1.0, 0.92)

current_inventory = st.sidebar.number_input("Current Inventory", value=1200)
safety_stock = st.sidebar.number_input("Safety Stock", value=400)

year = st.sidebar.number_input("Year", value=2026)
month = st.sidebar.selectbox("Month", list(range(1, 13)))
quarter = (month - 1) // 3 + 1

therapeutic_category = st.sidebar.selectbox(
    "Therapeutic Category", ["Analgesic", "Antibiotic", "Cardiac"]
)
dosage_form = st.sidebar.selectbox(
    "Dosage Form", ["Tablet", "Capsule", "Syrup"]
)
location = st.sidebar.selectbox(
    "Location", ["Mumbai", "Hyderabad", "Delhi"]
)

chronic_use_flag = st.sidebar.checkbox("Chronic Use")
flu_season_flag = st.sidebar.checkbox("Flu Season")
festival_season_flag = st.sidebar.checkbox("Festival Season")
monsoon_flag = st.sidebar.checkbox("Monsoon Season")

predict_btn = st.sidebar.button("🚀 Predict Stock")

# ---------------------------------------------------
# Prediction Logic
# ---------------------------------------------------
# if predict_btn:

#     input_df = pd.DataFrame({
#         "strength_mg": [strength_mg],
#         "unit_price": [unit_price],
#         "historical_sales_qty": [historical_sales_qty],
#         "rolling_mean_3m_sales": [rolling_mean_3m_sales],
#         "rolling_mean_6m_sales": [rolling_mean_6m_sales],
#         "sales_growth_yoy": [sales_growth_yoy],
#         "demand_volatility": [demand_volatility],
#         "lead_time_days": [lead_time_days],
#         "supplier_reliability_score": [supplier_reliability_score],
#         "current_inventory": [current_inventory],
#         "safety_stock": [safety_stock],
#         "year": [year],
#         "month": [month],
#         "quarter": [quarter],
#         "therapeutic_category": [therapeutic_category],
#         "dosage_form": [dosage_form],
#         "location": [location],
#         "chronic_use_flag": [int(chronic_use_flag)],
#         "flu_season_flag": [int(flu_season_flag)],
#         "festival_season_flag": [int(festival_season_flag)],
#         "monsoon_flag": [int(monsoon_flag)],
#     })

    # X_transformed = encoder.transform(input_df)
    # predicted_demand = int(model.predict(X_transformed)[0])
# if predict_btn:
#     # 1. Define the numeric features the model actually saw during fit()
#     # Ensure these are in the EXACT order they were in your training X
#     feature_cols = [
#         "strength_mg", "unit_price", "historical_sales_qty", 
#         "rolling_mean_3m_sales", "rolling_mean_6m_sales", 
#         "sales_growth_yoy", "demand_volatility", "lead_time_days", 
#         "supplier_reliability_score", "current_inventory", "safety_stock", 
#         "year", "month", "quarter", "chronic_use_flag", 
#         "flu_season_flag", "festival_season_flag", "monsoon_flag"
#     ]
    
    # # 2. Extract only these features from the input
    # # X_input = input_df[feature_cols]
    
    # # 3. Transform using the preprocessor (the ColumnTransformer from your previous message)
    # # Note: Ensure 'encoder' here is the ColumnTransformer/Scaler you saved
    # X_transformed = encoder.transform(input_df)
    
    # # 4. Predict
    # predicted_demand = int(model.predict(X_transformed)[0])
    # reorder_qty = max(
    #     predicted_demand + safety_stock - current_inventory,
    #     0
    # )

if predict_btn:
    # 1. Define the exact features your encoder expects
    feature_cols = [
        'strength_mg', 'unit_price', 'historical_sales_qty',
        'rolling_mean_3m_sales', 'rolling_mean_6m_sales', 'sales_growth_yoy',
        'demand_volatility', 'lead_time_days', 'supplier_reliability_score',
        'current_inventory', 'safety_stock', 
        'expected_demand_next_month',  # This is likely the column causing the error
        'year', 'month', 'therapeutic_category', 'dosage_form', 'location',
        'chronic_use_flag', 'flu_season_flag', 'festival_season_flag',
        'monsoon_flag', 'quarter'
    ]

    # 2. Create the input dataframe with a dummy value for the target column
    input_data = pd.DataFrame({
        "strength_mg": [strength_mg],
        "unit_price": [unit_price],
        "historical_sales_qty": [historical_sales_qty],
        "rolling_mean_3m_sales": [rolling_mean_3m_sales],
        "rolling_mean_6m_sales": [rolling_mean_6m_sales],
        "sales_growth_yoy": [sales_growth_yoy],
        "demand_volatility": [demand_volatility],
        "lead_time_days": [lead_time_days],
        "supplier_reliability_score": [supplier_reliability_score],
        "current_inventory": [current_inventory],
        "safety_stock": [safety_stock],
        "expected_demand_next_month": [0], # Dummy value to satisfy the encoder
        "year": [year],
        "month": [month],
        "therapeutic_category": [therapeutic_category],
        "dosage_form": [dosage_form],
        "location": [location],
        "chronic_use_flag": [int(chronic_use_flag)],
        "flu_season_flag": [int(flu_season_flag)],
        "festival_season_flag": [int(festival_season_flag)],
        "monsoon_flag": [int(monsoon_flag)],
        "quarter": [quarter],
    })

    # 3. Reorder columns to match the exact training order
    input_data = input_data[feature_cols]

    # 4. Transform and Predict
    X_transformed = encoder.transform(input_data)
    prediction = model.predict(X_transformed)
    predicted_demand = int(prediction[0])
    # ---------------------------------------------------
    # KPI Cards
    # ---------------------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("📦 Current Inventory", current_inventory)
    col2.metric("📈 Predicted Demand", predicted_demand)
    col3.metric("🔁 Reorder Quantity", reorder_qty)

    # ---------------------------------------------------
    # Inventory Alert
    # ---------------------------------------------------
    st.subheader("🚨 Inventory Status")

    if current_inventory >= predicted_demand:
        st.success("Inventory sufficient for next month")
    else:
        st.warning("Reorder required to avoid stock-out")

    # ---------------------------------------------------
    # Demand Trend Chart
    # ---------------------------------------------------
    st.subheader("📊 Sales vs Forecast")

    trend_df = pd.DataFrame({
        "Month": ["Last Month", "Next Month"],
        "Demand": [historical_sales_qty, predicted_demand]
    })

    st.line_chart(trend_df.set_index("Month"))

    # ---------------------------------------------------
    # Summary Table
    # ---------------------------------------------------
    st.subheader("📋 Inventory Projection")

    summary_df = pd.DataFrame({
        "Metric": [
            "Predicted Demand",
            "Current Inventory",
            "Safety Stock",
            "Reorder Quantity"
        ],
        "Value": [
            predicted_demand,
            current_inventory,
            safety_stock,
            reorder_qty
        ]
    })

    st.table(summary_df)

else:
    st.info("👈 Enter parameters and click **Predict Stock**")

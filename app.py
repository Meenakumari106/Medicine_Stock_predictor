import streamlit as st
import pandas as pd
import numpy as np
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
    try:
        encoder = joblib.load("encoder.pkl")  # encoder for 5 categorical flags
        model = joblib.load("lr_model.pkl")  # regression model
        return encoder, model
    except FileNotFoundError:
        st.error("Model files not found! Please upload encoder.pkl and lr_model.pkl.")
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

# Numeric inputs
strength_mg = st.sidebar.selectbox("Strength (mg)", [250, 500, 650])
unit_price = st.sidebar.number_input("Unit Price (₹)", value=12.5)
rolling_mean_3m_sales = st.sidebar.number_input("Rolling Mean (3M)", value=1750)
rolling_mean_6m_sales = st.sidebar.number_input("Rolling Mean (6M)", value=1650)
sales_growth_yoy = st.sidebar.slider("YoY Sales Growth", 0.0, 1.0, 0.12)
supplier_reliability_score = st.sidebar.slider("Supplier Reliability", 0.0, 1.0, 0.92)
year = st.sidebar.number_input("Year", value=2026)
month = st.sidebar.selectbox("Month", list(range(1, 13)))
quarter = (month - 1) // 3 + 1
current_inventory = st.sidebar.number_input("Current Inventory", value=1200)
safety_stock = st.sidebar.number_input("Safety Stock", value=400)

# Boolean flags (encoder columns)
chronic_use_flag = st.sidebar.checkbox("Chronic Use")
flu_season_flag = st.sidebar.checkbox("Flu Season")
festival_season_flag = st.sidebar.checkbox("Festival Season")
monsoon_flag = st.sidebar.checkbox("Monsoon Season")

predict_btn = st.sidebar.button("🚀 Predict Stock")

# ---------------------------------------------------
# Prediction Logic
# ---------------------------------------------------
if predict_btn:

    # -----------------------------
    # Numeric columns
    # -----------------------------
    numeric_cols = [
        "strength_mg", "unit_price", "rolling_mean_3m_sales", "rolling_mean_6m_sales",
        "sales_growth_yoy", "supplier_reliability_score", "year", "month"
    ]

    # -----------------------------
    # Encoder columns
    # -----------------------------
    encoder_cols = ["chronic_use_flag", "festival_season_flag", "flu_season_flag", "monsoon_flag", "quarter"]

    # -----------------------------
    # Build DataFrame
    # -----------------------------
    input_df = pd.DataFrame({
        **{col: [eval(col)] for col in numeric_cols + encoder_cols}
    })

    try:
        # -----------------------------
        # Encode categorical flags
        # -----------------------------
        X_encoded = encoder.transform(input_df[encoder_cols]).toarray()

        # Numeric columns as numpy
        X_numeric = input_df[numeric_cols].to_numpy()

        # Combine numeric + encoded categorical
        X_final = np.hstack([X_numeric, X_encoded])

        # Predict
        prediction = model.predict(X_final)
        predicted_demand = int(prediction[0])

        # Calculate reorder quantity
        reorder_qty = max(predicted_demand + safety_stock - current_inventory, 0)

        # -----------------------------
        # KPI Cards
        # -----------------------------
        col1, col2, col3 = st.columns(3)
        col1.metric("📦 Current Inventory", current_inventory)
        col2.metric("📈 Predicted Demand", predicted_demand)
        col3.metric("🔁 Reorder Quantity", reorder_qty)

        # -----------------------------
        # Inventory Alert
        # -----------------------------
        st.subheader("🚨 Inventory Status")
        if current_inventory >= predicted_demand:
            st.success("Inventory sufficient for next month")
        else:
            st.warning("Reorder required to avoid stock-out")

        # -----------------------------
        # Sales Trend Chart
        # -----------------------------
        st.subheader("📊 Sales vs Forecast")
        trend_df = pd.DataFrame({
            "Month": ["Last Avg", "Next Month"],
            "Demand": [rolling_mean_3m_sales, predicted_demand]
        })
        st.line_chart(trend_df.set_index("Month"))

        # -----------------------------
        # Summary Table
        # -----------------------------
        st.subheader("📋 Inventory Projection")
        summary_df = pd.DataFrame({
            "Metric": ["Predicted Demand", "Current Inventory", "Safety Stock", "Reorder Quantity"],
            "Value": [predicted_demand, current_inventory, safety_stock, reorder_qty]
        })
        st.table(summary_df)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Encoder expects:", encoder.feature_names_in_)
        st.write("Provided columns:", list(input_df.columns))

else:
    st.info("👈 Enter parameters and click **Predict Stock**")

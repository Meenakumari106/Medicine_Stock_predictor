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
# @st.cache_resource
# def load_models():
#     try:
#         encoder = joblib.load("encoder.pkl")  # encoder for 5 categorical flags
#         model = joblib.load("lr_model.pkl")  # regression model
#         return encoder, model
#     except FileNotFoundError:
#         st.error("Model files not found! Please upload encoder.pkl and lr_model.pkl.")
#         st.stop()

# encoder, model = load_models()
@st.cache_resource
def load_model():
    try:
        model = joblib.load("lr_model .pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please upload lr_model.pkl.")
        st.stop()

model = load_model()


# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown(
    """
    <h1 style='color:#1f4ed8'>💊 Cipla Smart Supply Planner</h1>
    <p>AI-powered stock requirement prediction & Location wise inventory planning</p>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------
st.sidebar.header("⚙️ Controls")
THERAPEUTIC_MAP = {
    "Antibiotic": 0,
    "CNS": 1,
    "Cardiac": 2,
    "Diabetes": 3,
    "Respiratory": 4
}

DOSAGE_FORM_MAP = {
    "Capsule": 0,
    "Inhaler": 1,
    "Injection": 2,
    "Syrup": 3,
    "Tablet": 4
}

LOCATION_MAP = {
    "Hyderabad - Ameerpet": 0,
    "Hyderabad - Dilsukhnagar": 1,
    "Hyderabad - Gachibowli": 2,
    "Hyderabad - Kukatpally": 3,
    "Hyderabad - Secunderabad": 4
}

QUARTER_MAP = {1: 0, 2: 1, 3: 2, 4: 3}

# Numeric inputs
medicine_name = st.sidebar.selectbox("Medicine_Id",['CiplaMed_1', 'CiplaMed_2', 'CiplaMed_3', 'CiplaMed_4',
       'CiplaMed_5', 'CiplaMed_6', 'CiplaMed_7', 'CiplaMed_8',
       'CiplaMed_9', 'CiplaMed_10', 'CiplaMed_11', 'CiplaMed_12',
       'CiplaMed_13', 'CiplaMed_14', 'CiplaMed_15', 'CiplaMed_16',
       'CiplaMed_17', 'CiplaMed_18', 'CiplaMed_19', 'CiplaMed_20',
       'CiplaMed_21', 'CiplaMed_22', 'CiplaMed_23', 'CiplaMed_24',
       'CiplaMed_25', 'CiplaMed_26', 'CiplaMed_27', 'CiplaMed_28',
       'CiplaMed_29', 'CiplaMed_30', 'CiplaMed_31', 'CiplaMed_32',
       'CiplaMed_33', 'CiplaMed_34', 'CiplaMed_35', 'CiplaMed_36',
       'CiplaMed_37', 'CiplaMed_38', 'CiplaMed_39', 'CiplaMed_40',
       'CiplaMed_41', 'CiplaMed_42', 'CiplaMed_43', 'CiplaMed_44',
       'CiplaMed_45', 'CiplaMed_46', 'CiplaMed_47', 'CiplaMed_48',
       'CiplaMed_49', 'CiplaMed_50', 'CiplaMed_51', 'CiplaMed_52',
       'CiplaMed_53', 'CiplaMed_54', 'CiplaMed_55', 'CiplaMed_56',
       'CiplaMed_57', 'CiplaMed_58', 'CiplaMed_59', 'CiplaMed_60',
       'CiplaMed_61', 'CiplaMed_62', 'CiplaMed_63', 'CiplaMed_64',
       'CiplaMed_65', 'CiplaMed_66', 'CiplaMed_67', 'CiplaMed_68',
       'CiplaMed_69', 'CiplaMed_70', 'CiplaMed_71', 'CiplaMed_72',
       'CiplaMed_73', 'CiplaMed_74', 'CiplaMed_75', 'CiplaMed_76',
       'CiplaMed_77', 'CiplaMed_78', 'CiplaMed_79', 'CiplaMed_80'])
location=st.sidebar.selectbox("Location", ['Hyderabad - Ameerpet', 'Hyderabad - Dilsukhnagar',
       'Hyderabad - Gachibowli', 'Hyderabad - Kukatpally',
       'Hyderabad - Secunderabad'])
current_inventory = st.sidebar.number_input("Current Inventory", value=1200)
year = st.sidebar.number_input("Year", value=2026)
month = st.sidebar.selectbox("Months", list(range(1, 13)))

strength_mg = st.sidebar.selectbox("Strength (mg)", [250, 500, 650])
unit_price = st.sidebar.number_input("Unit Price (₹)", value=12.5)
rolling_mean_3m_sales = st.sidebar.number_input("Rolling Mean (3M)", value=1750)
rolling_mean_6m_sales = st.sidebar.number_input("Rolling Mean (6M)", value=1650)
sales_growth_yoy = st.sidebar.slider("YoY Sales Growth", 0.0, 1.0, 0.12)
supplier_reliability_score = st.sidebar.slider("Supplier Reliability", 0.0, 1.0, 0.92)

quarter = (month - 1) // 3 + 1

safety_stock = st.sidebar.number_input("Safety Stock", value=400)

dosage_form = st.sidebar.selectbox("Dosage Form", ['Inhaler', 'Syrup', 'Injection', 'Tablet', 'Capsule'])
therapeutic_category = st.sidebar.selectbox("Therapeutic_category", ['Diabetes', 'CNS', 'Cardiac', 'Respiratory', 'Antibiotic'])
# location = LOCATION_MAPPING[location_label]
# dosage_form = DOSAGE_FORM_MAPPING[dosage_form_label]
therapeutic_category = THERAPEUTIC_MAP[therapeutic_category]
dosage_form = DOSAGE_FORM_MAP[dosage_form]
location = LOCATION_MAP[location]
quarter = QUARTER_MAP[quarter]



# Boolean flags (encoder columns)
# chronic_use_flag = st.sidebar.checkbox("Chronic Use")
# flu_season_flag = st.sidebar.checkbox("Flu Season")
# festival_season_flag = st.sidebar.checkbox("Festival Season")
# monsoon_flag = st.sidebar.checkbox("Monsoon Season")
chronic_use_flag = st.sidebar.selectbox("Chronic use", list(range(0, 2)))
flu_season_flag = st.sidebar.selectbox("Flu Season", list(range(0, 2)))
festival_season_flag = st.sidebar.selectbox("Festival Season", list(range(0, 2)))
monsoon_flag = st.sidebar.selectbox("Monsoon ", list(range(0, 2)))

predict_btn = st.sidebar.button("🚀 Predict Stock")

# ---------------------------------------------------
# Prediction Logic
# ---------------------------------------------------
if predict_btn:
    numerical_features = [
    "strength_mg",
    "unit_price",
    "historical_sales_qty",
    "rolling_mean_3m_sales",
    "rolling_mean_6m_sales",
    "sales_growth_yoy",
    "demand_volatility",
    "lead_time_days",
    "supplier_reliability_score",
    "current_inventory",
    "safety_stock",
    "expected_demand_next_month",
    "year",
    "month"
 ]

    categorical_features = [
        "therapeutic_category",
        "dosage_form",
        "location",
        "chronic_use_flag",
        "flu_season_flag",
        "festival_season_flag",
        "monsoon_flag",
        "quarter"
    ]


    # -----------------------------
    # Numeric columns
    # -----------------------------
    # numeric_cols = [
    #     "strength_mg", "unit_price", "rolling_mean_3m_sales", "rolling_mean_6m_sales",
    #     "sales_growth_yoy", "supplier_reliability_score", "year", "month"
    # ]

    # -----------------------------
    # Encoder columns
    # -----------------------------
    # encoder_cols = ["therapeutic_category",
    # "dosage_form",
    # "location",
    # "chronic_use_flag",
    # "flu_season_flag",
    # "festival_season_flag",
    # "monsoon_flag",
    # "quarter"
    #             ]

    historical_sales_qty = rolling_mean_3m_sales
    demand_volatility = 0.15
    lead_time_days = 30
    expected_demand_next_month = rolling_mean_3m_sales


    X_final = np.array([[
    strength_mg,
    unit_price,
    historical_sales_qty,
    rolling_mean_3m_sales,
    rolling_mean_6m_sales,
    sales_growth_yoy,
    demand_volatility,
    lead_time_days,
    supplier_reliability_score,
    current_inventory,
    safety_stock,
    expected_demand_next_month,
    year,
    month,
    therapeutic_category,
    dosage_form,
    location,
    chronic_use_flag,
    flu_season_flag,
    festival_season_flag,
    monsoon_flag,
    quarter
]])





    

    # # -----------------------------
    # Build DataFrame
    # -----------------------------
    # input_df = pd.DataFrame({
    #     **{col: [eval(col)] for col in numeric_cols + encoder_cols}
    # })

    try:
        # -----------------------------
        # Encode categorical flags
        # -----------------------------
        # X_encoded = encoder.transform(input_df[encoder_cols]).toarray()

        # Numeric columns as numpy
        # X_numeric = input_df[numeric_cols].to_numpy()

        # Combine numeric + encoded categorical
        # X_final = np.hstack([X_numeric, X_encoded])

        # Predict
        # prediction = model.predict(X_final)
        # predicted_demand = int(prediction[0])
        prediction_log = model.predict(X_final)
        predicted_demand = int(np.expm1(prediction_log[0]))


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
        st.subheader("📋 Inventory Projection")

        summary_df = pd.DataFrame({
            "Metric": [
                "Predicted Demand",
                "Current Inventory",
                "Safety Stock",
                "Reorder Quantity",
                "Location",
                "Dosage Form",
                "Therapeutic Category"
            ],
            "Value": [
                predicted_demand,
                current_inventory,
                safety_stock,
                reorder_qty,
                location,
                dosage_form,
                therapeutic_category
            ]
        })
        
        st.table(summary_df)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Model input shape:", X_final.shape)
        st.write("Model input values:", X_final)



        # -----------------------------
        # Summary Table
        # -----------------------------

    #     st.subheader("📋 Inventory Projection")
    #     summary_df = pd.DataFrame({
    #         "Metric": ["Predicted Demand", "Current Inventory", "Safety Stock", "Reorder Quantity","location",
    # "dosage_form",
    # "therapeutic_category",],
    #         "Value": [predicted_demand, current_inventory, safety_stock, reorder_qty]
    #     })
    #     st.table(summary_df)

    # except Exception as e:
    #     st.error(f"Prediction Error: {e}")
    #     # st.write("Encoder expects:", encoder.feature_names_in_)
    #     st.write("Provided columns:", list(input_df.columns))


else:
    st.info("👈 Enter parameters and click **Predict Stock**")

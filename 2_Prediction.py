import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap

# --- Page Config ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚀", layout="wide")

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    /* Main app background and font */
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Roboto', sans-serif;
        color: #333;
    }

    /* Titles and Headers */
    h1, h2, h3 {
        color: #0d3b66 !important; /* Dark Slate Blue */
        font-weight: 700 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d3b66 !important; /* Dark Slate Blue */
    }
    section[data-testid="stSidebar"] * {
        color: #fafafa !important; /* Off-white text */
    }

    /* Form Container Styling */
    .form-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Input Widgets (Dropdowns, number inputs) */
    div[data-baseweb="select"] > div, 
    input[type="number"], 
    input[type="text"] {
        background-color: #fafafa;
        border-radius: 8px !important;
        border: 1px solid #ddd;
        color: #333 !important;
    }
    
    /* Improved focus style for inputs */
    div[data-baseweb="select"] > div:focus-within,
    input[type="number"]:focus,
    input[type="text"]:focus {
        border-color: #17a2b8 !important;
        box-shadow: 0 0 0 3px rgba(23, 162, 184, 0.2) !important;
    }

    label {
        font-weight: 700 !important;
        color: #0d3b66 !important;
    }

    /* Predict Button Styling (works for form-submit too) */
    div.stButton > button,
    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #28a745, #218838) !important; /* Green gradient */
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        height: 3em !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: all 0.3s ease-in-out;
    }
    
    /* Hover effect */
    div.stButton > button:hover,
    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(90deg, #218838, #1e7e34) !important;
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
    }

    /* Prediction Result Metric */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d3b66, #17a2b8);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        text-align: center;
    }
    div[data-testid="stMetric"] * {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        font-size: 1.2rem;
    }
     div[data-testid="stMetric"] p {
        font-size: 2.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# --- Caching and Resource Loading ---
@st.cache_resource
def load_model_and_explainer(model_path):
    """Loads the model pipeline and creates a SHAP explainer."""
    try:
        with open(model_path, 'rb') as file:
            pipeline = pickle.load(file)
        
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['regressor']
        explainer = shap.TreeExplainer(model)
        
        return pipeline, preprocessor, explainer
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_data(data_path):
    """Loads the cleaned dataset."""
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        return None

# Load resources
pipeline, preprocessor, explainer = load_model_and_explainer('src/car_price_predictor.pkl')
df = load_data(r'src/cars24_cleaned.csv')

# --- App UI ---
st.title('🚀 Car Price Prediction Tool')
st.markdown("<h4 style='color: #5a7d9a;'>Enter the car's details below to get an accurate price estimate.</h4>", unsafe_allow_html=True)

if pipeline is None or df is None:
    st.error("⚠️ **Error:** A required file was not found. Please ensure 'src/car_price_predictor.pkl' and 'src/cars24_cleaned.csv' exist.")
    st.stop()

# --- Input Form ---
st.markdown('<div class="form-container">', unsafe_allow_html=True)
with st.form("prediction_form"):
    st.header("📝 Enter Car Details")
    
    # Dropdown data
    brand_to_model_map = df.groupby("Brand")["Model_Only"].unique().apply(list).to_dict()
    fuel_types = ['Diesel', 'Petrol', 'Electric', 'CNG', 'Hybrid']
    transmission_types = ['Manual', 'Auto']
    ownership_options = list(range(1, 11))

    col1, col2 = st.columns(2)
    with col1:
        selected_brand = st.selectbox('Brand', list(brand_to_model_map.keys()), index=0)
        available_models = brand_to_model_map.get(selected_brand, [])
        selected_model = st.selectbox('Model', available_models)
        transmission = st.selectbox('Transmission Type', transmission_types)
        
    with col2:
        fuel = st.selectbox('Fuel Type', fuel_types)
        year = st.number_input('Manufacturing Year', 2000, 2025, 2018)
        km_driven = st.number_input('Kilometers Driven', 0, 500000, 50000, 1000)
        ownership = st.selectbox('Ownership (e.g., 1 for First Owner)', ownership_options)

    submitted = st.form_submit_button('Predict Price')
st.markdown('</div>', unsafe_allow_html=True)


# --- Prediction & SHAP Explanation ---
if submitted:
    # --- Calculation ---
    current_year = 2025 
    car_age = current_year - year
    
    input_data = pd.DataFrame({
        'KM Driven': [km_driven],
        'Fuel Type': [fuel],
        'Transmission Type': [transmission],
        'Ownership': [ownership],
        'Brand': [selected_brand],
        'Model_Only': [selected_model],
        'Car Age': [car_age]
    })
    
    predicted_price = pipeline.predict(input_data)[0]
    
    # --- Display Results ---
    st.markdown("---")
    st.header("📊 Prediction Result")
    
    # Custom HTML for price display with fade-in animation
    st.markdown(
        f"""
        <style>
        @keyframes fadeIn {{
            0% {{ opacity: 0; transform: scale(0.95); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}
        .price-box {{
            animation: fadeIn 0.8s ease-in-out;
        }}
        </style>

        <div class="price-box"
             style="background: linear-gradient(135deg, #0d3b66, #17a2b8);
                    padding:25px; border-radius:12px; 
                    text-align:center; box-shadow:0 4px 15px rgba(0,0,0,0.15);">
            <label style="font-size:1.2rem; color:white; text-shadow:1px 1px 2px rgba(0,0,0,0.3);">
                Predicted Car Price
            </label>
            <p style="font-size:2.5rem; font-weight:700; color:white; 
                      text-shadow:1px 1px 2px rgba(0,0,0,0.3); margin:0;">
                ₹ {predicted_price:.2f} Lakhs
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("🔍 How the Model Made This Prediction")

    # Custom HTML for SHAP explanation box
    st.markdown(
        """
        <div style="background-color:#fff8e1;  /* light yellow bg */
                    border-left:6px solid #ff9800; /* orange accent */
                    padding:1.25rem; border-radius:10px; 
                    box-shadow:0 2px 4px rgba(0,0,0,0.05); margin-top: 1rem;">
            <p style="color:#333; font-weight:400; font-size:1rem; margin:0;">
                📌 The SHAP plot below shows which features pushed the prediction 
                <span style="color:red; font-weight:bold;">higher (in red)</span> 
                or <span style="color:blue; font-weight:bold;">lower (in blue)</span> 
                from the base value. The size of the bar indicates the magnitude of the feature's impact.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- SHAP Calculation and Plotting ---
    input_transformed = preprocessor.transform(input_data)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback for older scikit-learn versions
        num_features = list(preprocessor.named_transformers_['num'].feature_names_in_)
        ohe_transformer = preprocessor.named_transformers_['cat']
        cat_features_original = list(ohe_transformer.feature_names_in_)
        cat_features_generated = []
        for i, categories in enumerate(ohe_transformer.categories_):
            original_feature_name = cat_features_original[i]
            for category in categories:
                cat_features_generated.append(f"{original_feature_name}_{category}")
        feature_names = num_features + cat_features_generated
    
    shap_values = explainer.shap_values(input_transformed)
    
    if hasattr(input_transformed, "toarray"):
        input_transformed = input_transformed.toarray()
            
    input_transformed_df = pd.DataFrame(input_transformed, columns=feature_names)
    
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        input_transformed_df.iloc[0],
        matplotlib=False
    )
    
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    st.components.v1.html(shap_html, height=250, scrolling=True)


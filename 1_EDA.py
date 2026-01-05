# File: pages/1_📊_EDA_Dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Car Price EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    /* Main App */
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Roboto', sans-serif;
        color: #333;
    }

    h1, h2, h3 {
        color: #0d3b66 !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d3b66 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #fafafa !important;
    }

    /* Card-like container */
    .form-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Cache data loading ---
@st.cache_data
def load_data(data_path):
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        return None

# --- Plot helpers (cached to prevent re-render flickering) ---
@st.cache_data
def cached_univariate_plot(df, feature):
    buf = io.BytesIO()
    plt.figure(figsize=(8, 6))
    if pd.api.types.is_numeric_dtype(df[feature]):
        sns.histplot(df[feature], kde=True, bins=30)
    else:
        df[feature].value_counts().plot(kind="bar")
    plt.title(f"Distribution of {feature}", fontsize=16)
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@st.cache_data
def cached_bivariate_plot(df, x_feature, y_feature):
    buf = io.BytesIO()
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_feature, y=y_feature, alpha=0.6)
    plt.title(f"{x_feature} vs {y_feature}", fontsize=16)
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@st.cache_data
def cached_heatmap(df):
    buf = io.BytesIO()
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --- Load dataset ---
df = load_data("src/cars24_cleaned.csv")

st.title("📊 Car Price EDA Dashboard")
st.markdown("<h4 style='color: #5a7d9a;'>Explore patterns and relationships in the car dataset.</h4>", unsafe_allow_html=True)

if df is None:
    st.error("⚠️ **Error:** 'src/cars24_cleaned.csv' not found.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("EDA Options")
analysis_type = st.sidebar.radio(
    "Choose Analysis Type",
    ["Single Feature Analysis", "Two-Feature Relationship", "Correlation Insights"]
)

# --- Main Section ---
st.markdown('<div class="form-container">', unsafe_allow_html=True)

if analysis_type == "Single Feature Analysis":
    st.subheader("Single Feature Analysis")
    feature = st.selectbox("Select Feature", df.columns)

    img_base64 = cached_univariate_plot(df, feature)
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{img_base64}" 
                 alt="Distribution of {feature}" height="400">
        </div>
        """,
        unsafe_allow_html=True
    )

elif analysis_type == "Two-Feature Relationship":
    st.subheader("Two-Feature Relationship")
    x_feature = st.selectbox("X-axis Feature", df.columns, index=0)
    y_feature = st.selectbox("Y-axis Feature", df.columns, index=1)

    img_base64 = cached_bivariate_plot(df, x_feature, y_feature)
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{img_base64}" 
                 alt="{x_feature} vs {y_feature}" height="400">
        </div>
        """,
        unsafe_allow_html=True
    )

elif analysis_type == "Correlation Insights":
    st.subheader("Correlation Insights: Heatmap")

    img_base64 = cached_heatmap(df)
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{img_base64}" 
                 alt="Correlation Heatmap" height="450">
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

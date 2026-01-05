import streamlit as st
import pandas as pd
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Second-Hand Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700;900&display=swap');

    .stApp {
        background-color: #f8fafc;
        font-family: 'Roboto', sans-serif;
        color: #333;
    }

        /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d3b66 !important;
        position: fixed;               /* 🔑 makes sidebar fixed */
        top: 0;
        left: 0;
        height: 100vh;                 /* full viewport height */
        width: 21rem;                  /* stable width */
        overflow-y: auto;              /* scroll inside sidebar only */
        z-index: 100;
    }
    
    section[data-testid="stSidebar"] * {
        color: #fafafa !important;
    }
    
    /* Push main content so it doesn't go under sidebar */
    section[data-testid="stAppViewContainer"] {
        margin-left: 21rem;
        padding-left: 1.5rem;
    }
    
    /* Mobile fallback */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            position: relative;
            width: 100%;
            height: auto;
        }
    
        section[data-testid="stAppViewContainer"] {
            margin-left: 0;
            padding-left: 0;
        }
    }


    /* New Header Style */
    .main-header-container {
        background: linear-gradient(90deg, #0d3b66, #17a2b8);
        padding: 2rem 1rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    .main-header-container h1 {
        color: #ffffff !important;
        font-weight: 900;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .main-header-container h4 {
        text-align: center;
        color: #e0f2f6 !important; /* Lighter color for subtitle */
        font-weight: 500;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Card Containers */
    .content-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem; /* Reduced space between cards */
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d3b66, #17a2b8);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    div[data-testid="stMetric"] * {
        color: white !important;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        color: #0d3b66 !important;
        font-weight: 700;
    }
    .streamlit-expander {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
    }
    
    /* Navigation Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #17a2b8, #007bff) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        height: 3.5em !important;
        transition: all 0.3s ease-in-out;
    }
    
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(23, 162, 184, 0.4);
    }

    /* Technology Stack Styling */
    .tech-stack-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        justify-content: center;
        margin-top: 1rem;
    }
    .tech-tag {
        background-color: #eef2ff; /* Light indigo background */
        color: #4338ca; /* Indigo text */
        padding: 0.5rem 1rem;
        border-radius: 9999px; /* Pill shape */
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid #c7d2fe; /* Lighter indigo border */
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Title ---
st.sidebar.title("Navigation")

# --- Load Data ---
@st.cache_data
def load_data(data_path):
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        return None

df = load_data("src/cars24_cleaned.csv")

# --- Header ---
st.markdown("""
<div class='main-header-container'>
    <h1>🚗 Second-Hand Car Price Prediction</h1>
    <h4>Bringing Transparency to the Used Car Market with Machine Learning</h4>
</div>
""", unsafe_allow_html=True)

# --- Objective & Business Context ---
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.subheader("🎯 Project Objective")
st.write("""
The primary objective of this project is to solve the problem of **price ambiguity** in the used car market.  
We aim to develop a **machine learning model** that can accurately predict the value of a second-hand car
based on key features like **age, brand, mileage, and fuel type**.
""")

st.subheader("💼 Business Context")
st.write("""
The second-hand car market is **large and rapidly growing**, but it often lacks transparency compared to the new car market.  
This uncertainty creates challenges:
- 🚘 **Buyers** risk **overpaying** for vehicles.  
- 🏷️ **Sellers** risk **undervaluing** their assets.  

A reliable prediction tool empowers stakeholders with an **unbiased, data-driven price estimate**, 
helping build trust, enabling fairer negotiations, and streamlining transactions in the automotive industry.
""")
st.markdown('</div>', unsafe_allow_html=True)

# --- New Section: Technology Stack ---
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.subheader("🛠️ Technology Stack")
st.write("This project leverages a modern stack of data science and web development tools:")
st.markdown("""
<div class="tech-stack-container">
    <span class="tech-tag">🐍 Python</span>
    <span class="tech-tag">🐼 Pandas</span>
    <span class="tech-tag">🤖 Scikit-learn</span>
    <span class="tech-tag">📊 Matplotlib & Seaborn</span>
    <span class="tech-tag">🚀 Streamlit</span>
    <span class="tech-tag">🧠 SHAP</span>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- New Section: Project Workflow ---
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.subheader("⚙️ Project Workflow")

# Use columns to control the image size and center it
col1, col2, col3 = st.columns([1, 4, 1]) 
with col2:
    st.image("src/Data Collection,Data Exploration (EDA),Model Selection & Training,Model Evaluation,Model Deployment - visual selection (1).png", caption="A visual representation of the project workflow.", use_container_width=True)

st.write("The project followed a structured machine learning workflow to ensure robust and reliable results:")
st.markdown("""
1.  **Data Collection & Cleaning:** Sourced raw data and performed extensive cleaning to handle missing values, duplicates, and inconsistencies.
2.  **Exploratory Data Analysis (EDA):** Visualized data to uncover patterns, correlations, and key features influencing car prices.
3.  **Feature Engineering:** Created new features like 'Car Age' from existing data to improve model performance.
4.  **Model Training & Evaluation:** Trained multiple regression models and selected the best-performing one based on metrics like R-squared and RMSE.
5.  **Interactive Application:** Developed this Streamlit dashboard to provide an intuitive interface for users to interact with the model.
""")
st.markdown('</div>', unsafe_allow_html=True)


# --- Dataset Section ---
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.subheader("💾 About the Dataset")
if df is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Car Listings Analyzed", f"{len(df):,}")
    with col2:
        st.metric("Unique Car Brands Covered", df["Brand"].nunique())

    st.write("Our predictive model is trained on a rich dataset of used car listings with diverse features.")

    with st.expander("📂 View Dataset Features"):
        st.markdown("""
        - **Year**: Manufacturing year  
        - **Car Model**: Model name (e.g., Maruti Swift Dzire)  
        - **KM Driven**: Total kilometers driven  
        - **Fuel Type**: Petrol, Diesel, CNG, etc.  
        - **Transmission Type**: Manual / Automatic  
        - **Ownership**: Number of previous owners  
        - **Price (in Lakhs)**: Target variable for prediction  
        """)
else:
    st.warning("⚠️ Dataset not found. Please ensure 'src/cars24_cleaned.csv' is in the correct path.")
st.markdown('</div>', unsafe_allow_html=True)

# --- About Me ---
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.subheader("👨‍💻 About Me")
col1, col2 = st.columns([3, 1])
with col1:
    st.write("""
Hello! I am a passionate **Data Scientist & ML Engineer** specializing in building **end-to-end data products**.  
This project demonstrates my expertise in:
- 🔍 Data cleaning & preprocessing  
- 📊 Exploratory Data Analysis (EDA)  
- 🤖 Machine Learning model building  
- 🚀 Interactive apps using **Streamlit** """)
with col2:
    st.subheader("🔗 Connect with me:")
    st.markdown("""
    - [💼 LinkedIn](https://www.linkedin.com/in/k-yadagiri/)  
    - [🐙 GitHub](https://github.com/k-yadagiri)  
    - [🌐 Portfolio](https://yadagiri-k-portfolio.netlify.app/)  
    """)
st.markdown('</div>', unsafe_allow_html=True)


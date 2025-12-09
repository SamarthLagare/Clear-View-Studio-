import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import pickle

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, confusion_matrix

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Data Nexus Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navigation State
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'df' not in st.session_state:
    st.session_state.df = None

def navigate_to(page):
    st.session_state.page = page

# --- 2. PROFESSIONAL DARK CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117;
        color: #ffffff;
    }

    /* Layout */
    .block-container {
        max-width: 95%;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* Headings */
    h1, h2, h3 { text-align: left; font-weight: 600; }
    p { text-align: left; color: #a0a0a0; }

    /* Buttons (Teal Accent) */
    div.stButton > button {
        background-color: #1E2130;
        color: white;
        border: 1px solid #333;
        border-radius: 6px;
        height: 3em;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        border-color: #00d4ff;
        color: #00d4ff;
        background-color: #161924;
    }

    /* Primary Action Button */
    .primary-btn > button {
        background-color: #00d4ff !important;
        color: black !important;
        border: none !important;
        font-weight: 700 !important;
    }
    .primary-btn > button:hover {
        background-color: #00a3cc !important;
        transform: scale(1.01);
    }

    /* Cards */
    .data-card {
        background-color: #161924;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] { color: #00d4ff; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #333; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff; }

    /* Hide Default */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---

def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    return None

def download_model(model):
    output = BytesIO()
    pickle.dump(model, output)
    return output.getvalue()

def download_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 4. COMPONENT: DATA STUDIO ---
def render_data_studio():
    c1, c2 = st.columns([1, 3])
    
    with c1:
        st.markdown("### Import")
        file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'], label_visibility="collapsed")
        
        if file:
            st.session_state.df = load_data(file)
            st.success("File Loaded Successfully")
        
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("### Cleaning Tools")
            
            # Cleaning Actions
            if st.button("Remove Duplicates"):
                st.session_state.df.drop_duplicates(inplace=True)
                st.rerun()
            
            if st.button("Fill Missing (Mean)"):
                num_cols = st.session_state.df.select_dtypes(include=np.number).columns
                st.session_state.df[num_cols] = st.session_state.df[num_cols].fillna(st.session_state.df[num_cols].mean())
                st.rerun()
            
            drop_col = st.selectbox("Drop Column", st.session_state.df.columns)
            if st.button("Drop Selected Column"):
                st.session_state.df.drop(columns=[drop_col], inplace=True)
                st.rerun()

    with c2:
        if st.session_state.df is not None:
            st.markdown("### Data Preview")
            st.dataframe(st.session_state.df, use_container_width=True, height=400)
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", st.session_state.df.shape[0])
            m2.metric("Columns", st.session_state.df.shape[1])
            m3.metric("Missing Values", st.session_state.df.isna().sum().sum())
            m4.metric("Duplicates", st.session_state.df.duplicated().sum())
        else:
            st.info("Upload a dataset to activate the studio.")

# --- 5. COMPONENT: EDA LAB ---
def render_eda_lab():
    if st.session_state.df is None:
        st.warning("Please upload data in Data Studio first.")
        return

    df = st.session_state.df
    num_df = df.select_dtypes(include=np.number)
    
    t1, t2, t3 = st.tabs(["📊 Distribution", "🔥 Correlation", "📈 Relationships"])
    
    with t1:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.caption("Settings")
            col_dist = st.selectbox("Select Column", num_df.columns, key="dist_col")
            color_opt = st.selectbox("Color By", [None] + list(df.columns), key="dist_color")
        with c2:
            fig = px.histogram(df, x=col_dist, color=color_opt, marginal="box", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown("#### Correlation Matrix")
        if not num_df.empty:
            corr = num_df.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough numeric data for correlation.")

    with t3:
        c1, c2 = st.columns([1, 3])
        with c1:
            x_axis = st.selectbox("X Axis", num_df.columns, key="sc_x")
            y_axis = st.selectbox("Y Axis", num_df.columns, key="sc_y")
            z_axis = st.selectbox("Z Axis (3D)", [None] + list(num_df.columns), key="sc_z")
            color_sc = st.selectbox("Color", [None] + list(df.columns), key="sc_c")
        with c2:
            if z_axis:
                fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=color_sc, template="plotly_dark")
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_sc, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# --- 6. COMPONENT: MODEL FORGE ---
def render_model_forge():
    if st.session_state.df is None:
        st.warning("Please upload data first.")
        return

    df = st.session_state.df.dropna() # Drop NA for training safety
    
    # Encoder for categorical
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    c_conf, c_res = st.columns([1, 3])
    
    with c_conf:
        st.markdown("### Configuration")
        target = st.selectbox("Target Variable", df.columns)
        
        task_type = st.radio("Task Type", ["Regression", "Classification"])
        
        st.markdown("**Model Selection**")
        if task_type == "Regression":
            model_name = st.selectbox("Algorithm", ["Linear Regression", "Random Forest", "Neural Network (MLP)"])
        else:
            model_name = st.selectbox("Algorithm", ["Logistic Regression", "Random Forest", "Neural Network (MLP)"])
            
        split_size = st.slider("Train/Test Split", 0.1, 0.9, 0.8)
        
        # Hyperparams for NN
        if "Neural Network" in model_name:
            st.caption("Deep Learning Params")
            hidden_layers = st.text_input("Hidden Layers (e.g. 100,50)", "100,50")
            max_iter = st.number_input("Max Iterations", 100, 5000, 500)

        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        train_btn = st.button("TRAIN MODEL")
        st.markdown('</div>', unsafe_allow_html=True)

    with c_res:
        if train_btn:
            X = df.drop(columns=[target])
            y = df[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_size, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            model = None
            
            # REGRESSION
            if task_type == "Regression":
                if model_name == "Linear Regression": model = LinearRegression()
                elif model_name == "Random Forest": model = RandomForestRegressor()
                elif model_name == "Neural Network (MLP)":
                    layers = tuple(map(int, hidden_layers.split(',')))
                    model = MLPRegressor(hidden_layer_sizes=layers, max_iter=max_iter, random_state=42)
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                score = r2_score(y_test, preds)
                mse = mean_squared_error(y_test, preds)
                
                st.success(f"Training Complete! R² Score: {score:.4f}")
                
                m1, m2 = st.columns(2)
                m1.metric("R² Score", f"{score:.4f}")
                m2.metric("MSE", f"{mse:.4f}")
                
                # Plot
                fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted", template="plotly_dark")
                fig.add_shape(type="line", line=dict(dash='dash'), x0=y.min(), y0=y.max(), x1=y.min(), y1=y.max())
                st.plotly_chart(fig, use_container_width=True)

            # CLASSIFICATION
            else:
                if model_name == "Logistic Regression": model = LogisticRegression()
                elif model_name == "Random Forest": model = RandomForestClassifier()
                elif model_name == "Neural Network (MLP)":
                    layers = tuple(map(int, hidden_layers.split(',')))
                    model = MLPClassifier(hidden_layer_sizes=layers, max_iter=max_iter, random_state=42)
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                acc = accuracy_score(y_test, preds)
                st.success(f"Training Complete! Accuracy: {acc:.2%}")
                st.metric("Accuracy", f"{acc:.2%}")
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, preds)
                fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            # Export
            st.divider()
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("⬇️ Download Trained Model", download_model(model), "model.pkl")
            with col_d2:
                if st.session_state.df is not None:
                    st.download_button("⬇️ Download Cleaned Data", download_csv(st.session_state.df), "cleaned_data.csv")

# ==========================================
# 7. NAVIGATION SIDEBAR
# ==========================================
with st.sidebar:
    st.title("Navigation")
    
    opts = ["Home", "Data Studio", "EDA Lab", "Model Forge"]
    # Logic to sync radio with session state
    idx = 0
    if st.session_state.page == "data": idx = 1
    elif st.session_state.page == "eda": idx = 2
    elif st.session_state.page == "model": idx = 3
    
    nav = st.radio("Go to", opts, index=idx)
    
    if nav == "Home" and st.session_state.page != "home":
        st.session_state.page = "home"
        st.rerun()
    elif nav == "Data Studio" and st.session_state.page != "data":
        st.session_state.page = "data"
        st.rerun()
    elif nav == "EDA Lab" and st.session_state.page != "eda":
        st.session_state.page = "eda"
        st.rerun()
    elif nav == "Model Forge" and st.session_state.page != "model":
        st.session_state.page = "model"
        st.rerun()

# ==========================================
# 8. PAGES
# ==========================================

if st.session_state.page == 'home':
    st.title("DATA NEXUS PRO")
    st.markdown("### The Ultimate Analytics Platform")
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3, gap="medium")
    
    with c1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("## 💾")
        st.markdown("### Data Studio")
        st.caption("Upload, clean, and prepare your datasets.")
        if st.button("Launch Studio", key="h_data"):
            navigate_to("data")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("## 📊")
        st.markdown("### EDA Lab")
        st.caption("Visualize trends with automated plotting.")
        if st.button("Launch Lab", key="h_eda"):
            navigate_to("eda")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("## 🧠")
        st.markdown("### Model Forge")
        st.caption("Train ML & Neural Networks instantly.")
        if st.button("Launch Forge", key="h_model"):
            navigate_to("model")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'data':
    st.title("💾 Data Studio")
    render_data_studio()

elif st.session_state.page == 'eda':
    st.title("📊 EDA Lab")
    render_eda_lab()

elif st.session_state.page == 'model':
    st.title("🧠 Model Forge")
    render_model_forge()

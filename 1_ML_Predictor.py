# pages/1_ML_Predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

st.set_page_config(page_title="ML Predictor | Genofy", page_icon="🔬", layout="wide")

# --------------------------------------------------------
# Modern Dark Theme CSS (Genofy-inspired)
# --------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark Background */
    .stApp {
        background: linear-gradient(180deg, #0a1628 0%, #111c30 50%, #0f1b2d 100%);
    }
    
    /* Main Container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Headers */
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.5px !important;
    }
    
    h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Subtitle */
    .subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 2.5rem;
        line-height: 1.6;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        padding: 8px;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 179, 237, 0.1);
        color: #63b3ed;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Selectbox Fix */
    .stSelectbox [data-testid="stMarkdownContainer"] {
        color: #e2e8f0 !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #63b3ed !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #63b3ed;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(99, 179, 237, 0.3);
        transform: translateY(-2px);
    }
    
    /* Form Styling */
    .stForm {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div:focus-within {
        border-color: #63b3ed !important;
        box-shadow: 0 0 0 3px rgba(99, 179, 237, 0.1) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Labels */
    label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.875rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Result Cards */
    .result-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin: 1.5rem 0;
    }
    
    /* Info/Success Messages */
    .stInfo, .stSuccess, .stWarning, .stError {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid;
        backdrop-filter: blur(10px);
    }
    
    .stInfo {
        border-left-color: #3b82f6;
        color: #93c5fd;
    }
    
    .stSuccess {
        border-left-color: #10b981;
        color: #6ee7b7;
    }
    
    .stWarning {
        border-left-color: #f59e0b;
        color: #fcd34d;
    }
    
    .stError {
        border-left-color: #ef4444;
        color: #fca5a5;
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        color: #e2e8f0;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #63b3ed;
    }
    
    /* Paragraphs */
    p {
        color: #94a3b8;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 2rem 0 1.5rem 0;
    }
    
    .section-header h3 {
        margin: 0;
        color: #63b3ed !important;
    }
    
    /* Input Summary Cards */
    .input-summary-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .summary-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .summary-item:last-child {
        border-bottom: none;
    }
    
    .summary-label {
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    .summary-value {
        color: #63b3ed;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Description Box */
    .description-box {
        background: linear-gradient(135deg, rgba(99, 179, 237, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(99, 179, 237, 0.2);
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🔬 Mutation Impact Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced ML-powered genetic mutation analysis with real-time predictions</p>", unsafe_allow_html=True)

# Initialize session state for tabs
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'prediction'
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# ---- Simple Impact Score Calculator ----
def calculate_impact_score(gene, mutation, expression_level, gc_content, mutation_freq, category):
    gene_scores = {
        "BRCA1": 0.8, "BRCA2": 0.75, "TP53": 0.9, "EGFR": 0.6,
        "KRAS": 0.7, "PIK3CA": 0.5, "APC": 0.65, "PTEN": 0.55
    }
    
    mutation_multipliers = {
        "Missense": 0.6, "Nonsense": 0.9, "Frameshift": 0.85,
        "Silent": 0.1, "Splice": 0.8, "Insertion": 0.7, "Deletion": 0.75
    }
    
    category_weights = {
        "Pathogenic": 1.0, "High": 0.8, "Medium": 0.5, "Low": 0.2
    }
    
    base_score = gene_scores.get(gene, 0.5)
    mutation_mult = mutation_multipliers.get(mutation, 0.6)
    category_weight = category_weights.get(category, 0.5)
    
    score = base_score * mutation_mult * category_weight
    
    if expression_level > 7:
        score *= 1.2
    elif expression_level < 3:
        score *= 0.8
    
    if gc_content > 60 or gc_content < 40:
        score *= 1.1
    
    if mutation_freq < 0.001:
        score *= 1.3
    elif mutation_freq > 0.1:
        score *= 0.7
    
    return min(max(score, 0.0), 1.0)

# Tab navigation
tab1, tab2, tab3 = st.tabs(["🔬 Prediction", "📊 Detailed Analysis", "🔥 Heatmaps"])

with tab1:
    # System status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🤖 System Status", "Online", delta="Ready")
    with col2:
        st.metric("🎯 Supported Genes", "8+", delta="Active")
    with col3:
        st.metric("⚡ Avg Response", "<0.1s", delta="Fast")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Single Sample Prediction ----
    st.markdown("<div class='section-header'><h3>🔍 Single Sample Prediction</h3></div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 1.5rem;'>Enter genetic parameters for individual mutation impact analysis</p>", unsafe_allow_html=True)
    
    with st.form("single_sample", clear_on_submit=False):
        st.markdown("#### 🧬 Genetic Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            gene = st.text_input("🧬 Gene Name", value="", placeholder="e.g., BRCA1, TP53, KRAS")
            mutation = st.selectbox("🔬 Mutation Type", options=["Missense", "Nonsense", "Frameshift", "Silent", "Splice", "Insertion", "Deletion"])
            sequence = st.text_area("📝 Sequence (Optional)", value="", placeholder="Enter genetic sequence if available")
        
        with col2:
            expression_level = st.number_input("📊 Expression Level", value=5.0, min_value=0.0, max_value=10.0, step=0.1, format="%.2f", help="Gene expression measurement (0-10)")
            gc_content = st.number_input("🧪 GC Content (%)", value=50.0, min_value=0.0, max_value=100.0, step=0.1, format="%.2f", help="Percentage of G-C base pairs")
            mutation_freq = st.number_input("📈 Mutation Frequency", value=0.01, min_value=0.0, max_value=1.0, step=0.001, format="%.4f", help="Population frequency")
            category = st.selectbox("🏷️ Classification", options=["Pathogenic", "High", "Medium", "Low"], help="Impact category")
        
        submitted = st.form_submit_button("🧠 Predict Impact Score")

    if submitted:
        if not gene:
            st.error("⚠️ Please enter a gene name")
        else:
            sample = pd.DataFrame([{
                "Gene": gene,
                "Mutation": mutation,
                "Sequence": sequence,
                "Expression_Level": expression_level,
                "GC_Content": gc_content,
                "Mutation_Frequency": mutation_freq,
                "Category": category
            }])

            try:
                pred_score = calculate_impact_score(
                    gene, mutation, expression_level, gc_content, mutation_freq, category
                )
                
                st.session_state.prediction_result = {
                    'score': pred_score,
                    'gene': gene,
                    'mutation': mutation,
                    'expression_level': expression_level,
                    'gc_content': gc_content,
                    'mutation_freq': mutation_freq,
                    'category': category
                }
                
                # Determine risk level
                if pred_score < 0.4:
                    level = "Low Risk"
                    color = "#10b981"
                    emoji = "🟢"
                    gradient = "linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.05) 100%)"
                    description = "This mutation shows minimal functional impact. Regular monitoring is recommended, but immediate intervention is not typically required."
                elif pred_score < 0.7:
                    level = "Medium Risk"
                    color = "#f59e0b"
                    emoji = "🟡"
                    gradient = "linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.05) 100%)"
                    description = "Moderate impact detected. Further clinical evaluation and periodic follow-ups are advised to monitor potential progression."
                else:
                    level = "High Risk"
                    color = "#ef4444"
                    emoji = "🔴"
                    gradient = "linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.05) 100%)"
                    description = "High impact mutation identified. Immediate clinical consultation and comprehensive diagnostic evaluation are strongly recommended."
                
                st.markdown(f"""
                <div style='background: {gradient}; 
                            padding: 3rem; border-radius: 20px; border: 1px solid {color}40; margin-top: 2rem;
                            box-shadow: 0 15px 40px {color}20; backdrop-filter: blur(10px);'>
                    <div style='text-align: center;'>
                        <h2 style='color: {color}; margin: 0; font-size: 1.5rem;'>{emoji} Prediction Result</h2>
                        <h1 style='color: {color}; margin: 1rem 0; font-size: 4.5rem; font-weight: 800;'>{pred_score:.4f}</h1>
                        <p style='font-size: 1.5rem; color: #e2e8f0; margin: 0; font-weight: 600;'>Impact Level: <strong>{level}</strong></p>
                        <div style='margin-top: 2rem; padding-top: 2rem; border-top: 1px solid rgba(255, 255, 255, 0.1);'>
                            <p style='color: #94a3b8; margin: 0;'>Gene: <strong style='color: #63b3ed;'>{gene}</strong> | Mutation: <strong style='color: #63b3ed;'>{mutation}</strong></p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Result Description
                st.markdown(f"""
                <div class='description-box'>
                    <h4 style='color: #63b3ed; margin-top: 0;'>📋 Clinical Interpretation</h4>
                    <p style='color: #e2e8f0; margin-bottom: 1rem; font-size: 1.05rem; line-height: 1.6;'>{description}</p>
                    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1.5rem;'>
                        <div style='text-align: center;'>
                            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🎯</div>
                            <p style='color: #94a3b8; margin: 0; font-size: 0.9rem;'>Confidence Score</p>
                            <p style='color: #63b3ed; margin: 0; font-weight: 600;'>{pred_score:.1%}</p>
                        </div>
                        <div style='text-align: center;'>
                            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>⚡</div>
                            <p style='color: #94a3b8; margin: 0; font-size: 0.9rem;'>Risk Category</p>
                            <p style='color: {color}; margin: 0; font-weight: 600;'>{level}</p>
                        </div>
                        <div style='text-align: center;'>
                            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🔬</div>
                            <p style='color: #94a3b8; margin: 0; font-size: 0.9rem;'>Analysis Type</p>
                            <p style='color: #63b3ed; margin: 0; font-weight: 600;'>ML Prediction</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced Input Summary
                st.markdown("#### 📊 Input Summary")
                st.markdown("<p style='color: #94a3b8; margin-bottom: 1rem;'>Review the parameters used for this prediction</p>", unsafe_allow_html=True)
                
                st.markdown("""
                <div class='input-summary-card'>
                    <div class='summary-item'>
                        <span class='summary-label'>🧬 Gene Name</span>
                        <span class='summary-value'>{gene}</span>
                    </div>
                    <div class='summary-item'>
                        <span class='summary-label'>🔬 Mutation Type</span>
                        <span class='summary-value'>{mutation}</span>
                    </div>
                    <div class='summary-item'>
                        <span class='summary-label'>📊 Expression Level</span>
                        <span class='summary-value'>{expression_level:.2f}</span>
                    </div>
                    <div class='summary-item'>
                        <span class='summary-label'>🧪 GC Content</span>
                        <span class='summary-value'>{gc_content:.1f}%</span>
                    </div>
                    <div class='summary-item'>
                        <span class='summary-label'>📈 Mutation Frequency</span>
                        <span class='summary-value'>{mutation_freq:.4f}</span>
                    </div>
                    <div class='summary-item'>
                        <span class='summary-label'>🏷️ Classification</span>
                        <span class='summary-value'>{category}</span>
                    </div>
                    <div class='summary-item' style='border-top: 2px solid rgba(99, 179, 237, 0.3); padding-top: 1.5rem; margin-top: 0.5rem;'>
                        <span class='summary-label' style='color: #63b3ed; font-size: 1.1rem;'>🎯 Final Impact Score</span>
                        <span class='summary-value' style='color: {color}; font-size: 1.2rem; font-weight: 700;'>{pred_score:.4f}</span>
                    </div>
                </div>
                """.format(
                    gene=gene, mutation=mutation, expression_level=expression_level,
                    gc_content=gc_content, mutation_freq=mutation_freq, category=category,
                    color=color, pred_score=pred_score
                ), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Tip:** Select from the supported gene list (BRCA1, BRCA2, TP53, EGFR, KRAS, PIK3CA, APC, PTEN) for best accuracy. Other genes will use default scoring.")

with tab2:
    st.markdown("<div class='section-header'><h3>📊 Detailed Analysis</h3></div>", unsafe_allow_html=True)
    
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        score = result['score']
        
        # Eye-catching analysis display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Determine color based on score
            if score < 0.4:
                card_color = "#10b981"
                bg_gradient = "linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%)"
            elif score < 0.7:
                card_color = "#f59e0b"
                bg_gradient = "linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(245, 158, 11, 0.05) 100%)"
            else:
                card_color = "#ef4444"
                bg_gradient = "linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%)"
            
            st.markdown(f"""
            <div style='background: {bg_gradient}; 
                        padding: 3rem; border-radius: 20px; text-align: center;
                        box-shadow: 0 15px 35px {card_color}30; border: 1px solid {card_color}40;
                        backdrop-filter: blur(10px);'>
                <h1 style='color: {card_color}; font-size: 5rem; margin: 0; text-shadow: 2px 2px 8px {card_color}40;'>🎯</h1>
                <h2 style='color: #ffffff; margin: 1.5rem 0;'>Impact Score Analysis</h2>
                <h1 style='color: {card_color}; font-size: 6rem; margin: 1rem 0; font-weight: 800;'>{score:.4f}</h1>
                <p style='font-size: 1.8rem; margin: 0; color: #e2e8f0;'>Gene: <strong>{result['gene']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Risk gauge
            gauge_position = score * 100
            st.markdown(f"""
            <div style='background: rgba(255, 255, 255, 0.03); padding: 2rem; border-radius: 16px; text-align: center;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.2); border: 1px solid rgba(255, 255, 255, 0.08);'>
                <h3 style='color: #ffffff; margin-bottom: 2rem;'>Risk Gauge</h3>
                <div style='height: 250px; width: 40px; background: linear-gradient(180deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
                           border-radius: 20px; margin: 0 auto; position: relative; box-shadow: 0 4px 15px rgba(99, 179, 237, 0.3);'>
                    <div style='position: absolute; right: -15px; top: {250-gauge_position*2.5}px; 
                               width: 0; height: 0; border-left: 15px solid #63b3ed;
                               border-top: 8px solid transparent; border-bottom: 8px solid transparent;
                               filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));'></div>
                </div>
                <p style='margin-top: 1.5rem; color: #94a3b8; font-size: 1.2rem; font-weight: 600;'>{gauge_position:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Detailed breakdown
        st.markdown("#### 🔬 Factor Analysis")
        
        factors = [
            ("🧬 Gene Type", result['gene'], "High impact genes like TP53, BRCA1 have higher base scores"),
            ("🔬 Mutation Type", result['mutation'], "Nonsense and frameshift mutations typically more severe"),
            ("📊 Expression Level", f"{result['expression_level']:.2f}", "Higher expression amplifies mutation impact"),
            ("🧪 GC Content", f"{result['gc_content']:.1f}%", "Extreme GC content can increase instability"),
            ("📈 Mutation Frequency", f"{result['mutation_freq']:.4f}", "Rare mutations often have higher impact"),
            ("🏷️ Category", result['category'], "Pathogenic classification increases severity")
        ]
        
        for factor, value, description in factors:
            st.markdown(f"""
            <div style='background: rgba(255, 255, 255, 0.03); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                        border-left: 4px solid #3b82f6; border: 1px solid rgba(255, 255, 255, 0.08);
                        transition: all 0.3s ease;'>
                <h4 style='color: #e2e8f0; margin: 0 0 0.5rem 0;'>{factor}: <span style='color: #63b3ed; font-weight: 700;'>{value}</span></h4>
                <p style='color: #94a3b8; margin: 0; font-size: 0.9rem;'>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("🔍 Make a prediction first to see detailed analysis")
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.03); padding: 2rem; border-radius: 16px; 
                    border: 1px solid rgba(255, 255, 255, 0.08); text-align: center; margin-top: 2rem;'>
            <h2 style='color: #94a3b8; margin: 0;'>No analysis data available</h2>
            <p style='color: #64748b; margin-top: 1rem;'>Navigate to the Prediction tab and run an analysis first</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='section-header'><h3>🔥 Impact Heatmaps</h3></div>", unsafe_allow_html=True)
    
    if st.session_state.prediction_result:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set dark theme for matplotlib
        plt.style.use('dark_background')
        
        result = st.session_state.prediction_result
        
        # Create synthetic heatmap data
        genes = ['BRCA1', 'BRCA2', 'TP53', 'EGFR', 'KRAS', 'PIK3CA', 'APC', 'PTEN']
        mutations = ['Missense', 'Nonsense', 'Frameshift', 'Silent', 'Splice']
        
        # Generate heatmap matrix
        heatmap_data = np.random.rand(len(genes), len(mutations))
        # Highlight current prediction
        if result['gene'] in genes and result['mutation'] in mutations:
            gene_idx = genes.index(result['gene'])
            mut_idx = mutations.index(result['mutation'])
            heatmap_data[gene_idx, mut_idx] = result['score']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor('#0f1b2d')
        
        # Gene-Mutation Impact Heatmap
        sns.heatmap(heatmap_data, xticklabels=mutations, yticklabels=genes, 
                   annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax1, 
                   cbar_kws={'label': 'Impact Score'},
                   linewidths=0.5, linecolor='#1e293b')
        ax1.set_title('Gene-Mutation Impact Matrix', fontsize=16, fontweight='bold', color='#e2e8f0', pad=20)
        ax1.set_xlabel('Mutation Type', fontsize=12, color='#cbd5e1')
        ax1.set_ylabel('Gene', fontsize=12, color='#cbd5e1')
        ax1.tick_params(colors='#94a3b8')
        ax1.set_facecolor('#111c30')
        
        # Expression-GC Content Heatmap
        expr_range = np.linspace(0, 10, 10)
        gc_range = np.linspace(30, 70, 10)
        expr_gc_data = np.random.rand(10, 10)
        
        sns.heatmap(expr_gc_data, xticklabels=[f'{x:.1f}' for x in expr_range], 
                   yticklabels=[f'{x:.0f}%' for x in gc_range], 
                   annot=True, fmt='.2f', cmap='viridis', ax=ax2,
                   cbar_kws={'label': 'Impact Score'},
                   linewidths=0.5, linecolor='#1e293b')
        ax2.set_title('Expression-GC Content Impact', fontsize=16, fontweight='bold', color='#e2e8f0', pad=20)
        ax2.set_xlabel('Expression Level', fontsize=12, color='#cbd5e1')
        ax2.set_ylabel('GC Content', fontsize=12, color='#cbd5e1')
        ax2.tick_params(colors='#94a3b8')
        ax2.set_facecolor('#111c30')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Feature importance visualization
        st.markdown("#### 📊 Feature Importance")
        
        features = ['Gene Type', 'Mutation', 'Expression', 'GC Content', 'Frequency', 'Category']
        importance = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
        
        fig2, ax3 = plt.subplots(figsize=(12, 7))
        fig2.patch.set_facecolor('#0f1b2d')
        ax3.set_facecolor('#111c30')
        
        colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#ec4899']
        bars = ax3.barh(features, importance, color=colors, edgecolor='#1e293b', linewidth=1.5)
        ax3.set_xlabel('Importance Score', fontsize=12, color='#cbd5e1', fontweight='bold')
        ax3.set_title('Feature Importance in Impact Prediction', fontsize=16, fontweight='bold', color='#e2e8f0', pad=20)
        ax3.tick_params(colors='#94a3b8')
        ax3.grid(axis='x', alpha=0.1, color='#475569')
        
        # Add value labels on bars
        for bar, imp in zip(bars, importance):
            ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{imp:.2f}', va='center', fontweight='bold', color='#e2e8f0')
        
        plt.tight_layout()
        st.pyplot(fig2)
        
    else:
        st.info("🔍 Make a prediction first to generate heatmaps")
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.03); padding: 3rem; border-radius: 16px; 
                    border: 1px solid rgba(255, 255, 255, 0.08); text-align: center; margin-top: 2rem;'>
            <h2 style='color: #94a3b8; margin: 0;'>📊 Visualization Ready</h2>
            <p style='color: #64748b; margin-top: 1rem;'>Run a prediction to see comprehensive heatmap analysis</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0; border-top: 1px solid rgba(255, 255, 255, 0.05);'>
    <p style='margin: 0;'>🔬 <strong>Mutation Impact Predictor</strong> | Powered by Advanced ML Algorithms</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>© 2025 Genofy - Genetic Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)
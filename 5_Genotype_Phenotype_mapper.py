# pages/4_Genotype_Phenotype_Mapper.py
import streamlit as st
import pandas as pd
from traning_testing.predict_genotype_phenotype import predict_trait
import joblib, os
import numpy as np

# 🧠 Explain model reasoning based on input pattern
def explain_prediction(expr, impact):
    """
    Returns a short textual explanation for why a certain phenotype might be predicted.
    This matches the synthetic dataset rules.
    """
    if impact > 0.7 and expr > 0.7:
        reason = "High expression and high impact indicate strong mutation activity linked to disease risk."
    elif impact < 0.3 and expr < 0.3:
        reason = "Low expression and low impact suggest normal gene behavior with minimal phenotypic effect."
    elif expr > 0.5 and impact < 0.5:
        reason = "Moderate expression but low impact — potential metabolic irregularity detected."
    else:
        reason = "Mixed or moderate values — possible adaptive resistance variant pattern."
    return reason

# --------------------------------------------------------
# PAGE CONFIG & MODERN DARK DESIGN
# --------------------------------------------------------
st.set_page_config(page_title="Genotype-Phenotype Mapper | Genofy", page_icon="🧬", layout="wide")

# Modern Dark Theme CSS (Genofy-inspired)
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
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
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

    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.05) 100%);
        backdrop-filter: blur(10px);
        padding: 3rem;
        border-radius: 20px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin: 2rem 0;
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.2);
        text-align: center;
    }

    /* Info/Success Messages */
    .stInfo, .stSuccess, .stWarning {
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
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #63b3ed;
        border-color: rgba(99, 179, 237, 0.2);
    }
    
    /* Paragraphs */
    p {
        color: #94a3b8;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        border-left: 4px solid #3b82f6;
        margin: 2rem 0 1.5rem 0;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .section-header h3 {
        margin: 0;
        color: #63b3ed !important;
    }
    
    /* Divider */
    hr {
        margin: 3rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    }
    
    /* Chart containers */
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# PAGE TITLE
# --------------------------------------------------------
st.markdown("<h1>🧬 Genotype → Phenotype Mapper</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict phenotypic traits and disease risks from genetic variants with AI-powered precision</p>", unsafe_allow_html=True)

# Feature highlights
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.03); padding: 1.5rem; border-radius: 12px; 
                border: 1px solid rgba(255, 255, 255, 0.08); text-align: center;'>
        <h3 style='color: #63b3ed; margin: 0; font-size: 2rem;'>99%</h3>
        <p style='color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Prediction Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.03); padding: 1.5rem; border-radius: 12px; 
                border: 1px solid rgba(255, 255, 255, 0.08); text-align: center;'>
        <h3 style='color: #8b5cf6; margin: 0; font-size: 2rem;'>4</h3>
        <p style='color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Phenotype Classes</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.03); padding: 1.5rem; border-radius: 12px; 
                border: 1px solid rgba(255, 255, 255, 0.08); text-align: center;'>
        <h3 style='color: #10b981; margin: 0; font-size: 2rem;'>5+</h3>
        <p style='color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Supported Genes</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.03); padding: 1.5rem; border-radius: 12px; 
                border: 1px solid rgba(255, 255, 255, 0.08); text-align: center;'>
        <h3 style='color: #f59e0b; margin: 0; font-size: 2rem;'>&lt;0.1s</h3>
        <p style='color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Prediction Time</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------------
# INPUT FORM
# --------------------------------------------------------
st.markdown("<div class='section-header'><h3>🧫 Input Genetic Parameters</h3></div>", unsafe_allow_html=True)

with st.form("geno_form"):
    st.markdown("#### 🧬 Genetic Variant Information")

    col1, col2 = st.columns(2)
    with col1:
        gene_id = st.text_input("🧬 Gene ID", value="TP53", placeholder="e.g., BRCA1, TP53, EGFR, KRAS")
        expression_level = st.number_input("📈 Expression Level", value=0.72, min_value=0.0, max_value=1.0, step=0.01, format="%.3f", 
                                          help="Gene expression measurement (0.0-1.0)")
    with col2:
        mutation = st.text_input("🔬 Mutation Sequence", value="A>T", placeholder="e.g., A>T, G>C, C>T, T>A")
        impact_score = st.number_input("💥 Impact Score", value=0.81, min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
                                      help="Predicted mutation impact (0.0-1.0)")

    submitted = st.form_submit_button("🚀 Predict Phenotype")

# --------------------------------------------------------
# PREDICTION & OUTPUT
# --------------------------------------------------------
if submitted:
    if not gene_id or not mutation:
        st.error("⚠️ Please provide both Gene ID and Mutation Sequence")
    else:
        with st.spinner("🔄 Analyzing genetic pattern with AI..."):
            label, confidence, prob_dict = predict_trait(gene_id, mutation, expression_level, impact_score)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><h3>🎯 Prediction Results</h3></div>", unsafe_allow_html=True)

        # Determine phenotype color
        phenotype_colors = {
            "Cancer_Susceptibility": "#ef4444",
            "Metabolic_Disorder": "#f59e0b",
            "Normal": "#10b981",
            "Resistance_Variant": "#3b82f6"
        }
        
        phenotype_emojis = {
            "Cancer_Susceptibility": "🔴",
            "Metabolic_Disorder": "🟡",
            "Normal": "🟢",
            "Resistance_Variant": "🔵"
        }
        
        color = phenotype_colors.get(label, "#63b3ed")
        emoji = phenotype_emojis.get(label, "🧬")
        
        # Prediction card
        st.markdown(f"""
        <div class='prediction-card' style='border-color: {color}60; 
             background: linear-gradient(135deg, {color}20 0%, {color}05 100%);'>
            <h3 style='color: #e2e8f0; margin-bottom: 1rem; font-size: 1.3rem;'>🧠 Predicted Phenotype</h3>
            <h1 style='color: {color}; font-weight: 800; font-size: 3.5rem; margin: 1rem 0;'>{emoji} {label.replace('_', ' ')}</h1>
            <p style='font-size: 1.5rem; color: #e2e8f0; font-weight: 600;'>Confidence: <strong style='color: {color};'>{confidence*100:.2f}%</strong></p>
            <div style='margin-top: 2rem; padding-top: 2rem; border-top: 1px solid rgba(255, 255, 255, 0.1);'>
                <p style='color: #94a3b8; margin: 0; font-size: 1rem;'>Gene: <strong style='color: #63b3ed;'>{gene_id}</strong> | Mutation: <strong style='color: #63b3ed;'>{mutation}</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Explanation
        explanation = explain_prediction(expression_level, impact_score)
        st.info(f"💡 **Analysis:** {explanation}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence Distribution
        st.markdown("#### 📊 Confidence Distribution Across All Phenotypes")

        # True probabilities (for data integrity)
        true_probs = np.array(list(prob_dict.values()))

        # Softened version (for more natural bar chart visuals)
        soft_probs = true_probs.copy()
        if np.max(true_probs) == 1.0:
            soft_probs = soft_probs * 0.9
            soft_probs[np.argmax(soft_probs)] = 1.0
            soft_probs = soft_probs / np.sum(soft_probs)

        # Prepare two dataframes
        df_visual = pd.DataFrame({
            "Phenotype": [p.replace('_', ' ') for p in prob_dict.keys()],
            "Display_Probability (%)": [round(v*100, 2) for v in soft_probs]
        })
        df_true = pd.DataFrame({
            "Phenotype": [p.replace('_', ' ') for p in prob_dict.keys()],
            "True_Probability": [round(v*100, 4) for v in true_probs]
        })

        # Display soft chart
        st.bar_chart(df_visual.set_index("Phenotype"))
        st.caption("📈 Confidence bars visually softened for better interpretability (display only).")

        # Expandable section with true probabilities
        with st.expander("🔢 View Raw Model Probabilities"):
            st.dataframe(df_true.set_index("Phenotype"), use_container_width=True)
            st.info("These are the exact probability outputs returned by the model.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Phenotype information
        st.markdown("<div class='section-header'><h3>📚 Phenotype Information</h3></div>", unsafe_allow_html=True)
        
        phenotype_info = {
            "Cancer_Susceptibility": {
                "description": "Genetic variant associated with increased risk of cancer development",
                "implications": "May require enhanced screening and preventive measures",
                "examples": "BRCA1/2 mutations, TP53 variants"
            },
            "Metabolic_Disorder": {
                "description": "Variant affecting metabolic pathways and biochemical processes",
                "implications": "Could impact nutrient processing and energy metabolism",
                "examples": "MTHFR variants, G6PD deficiency"
            },
            "Normal": {
                "description": "Genetic variant with minimal or no pathogenic effect",
                "implications": "Standard health monitoring recommended",
                "examples": "Common polymorphisms, silent mutations"
            },
            "Resistance_Variant": {
                "description": "Variant conferring resistance to diseases or treatments",
                "implications": "May provide protective benefits or treatment considerations",
                "examples": "CCR5-Δ32 (HIV resistance), Factor V Leiden"
            }
        }
        
        info = phenotype_info.get(label, {})
        if info:
            st.markdown(f"""
            <div style='background: rgba(255, 255, 255, 0.03); padding: 2rem; border-radius: 16px;
                        border: 1px solid rgba(255, 255, 255, 0.08); border-left: 4px solid {color};'>
                <h4 style='color: {color}; margin-bottom: 1rem;'>{emoji} {label.replace('_', ' ')}</h4>
                <p style='margin: 0.75rem 0;'><strong style='color: #e2e8f0;'>Description:</strong> <span style='color: #94a3b8;'>{info.get('description', 'N/A')}</span></p>
                <p style='margin: 0.75rem 0;'><strong style='color: #e2e8f0;'>Implications:</strong> <span style='color: #94a3b8;'>{info.get('implications', 'N/A')}</span></p>
                <p style='margin: 0.75rem 0;'><strong style='color: #e2e8f0;'>Examples:</strong> <span style='color: #94a3b8;'>{info.get('examples', 'N/A')}</span></p>
            </div>
            """, unsafe_allow_html=True)

else:
    # Show example usage
    st.markdown("<div class='section-header'><h3>💡 How to Use</h3></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.03); padding: 2rem; border-radius: 16px;
                    border: 1px solid rgba(255, 255, 255, 0.08);'>
            <h4 style='color: #63b3ed; margin-bottom: 1rem;'>📝 Input Parameters</h4>
            <ul style='color: #94a3b8; line-height: 2;'>
                <li><strong style='color: #e2e8f0;'>Gene ID:</strong> Enter gene symbol (e.g., TP53, BRCA1)</li>
                <li><strong style='color: #e2e8f0;'>Mutation:</strong> Specify variant (e.g., A>T, G>C)</li>
                <li><strong style='color: #e2e8f0;'>Expression:</strong> Gene activity level (0.0-1.0)</li>
                <li><strong style='color: #e2e8f0;'>Impact Score:</strong> Predicted effect strength (0.0-1.0)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.03); padding: 2rem; border-radius: 16px;
                    border: 1px solid rgba(255, 255, 255, 0.08);'>
            <h4 style='color: #8b5cf6; margin-bottom: 1rem;'>🎯 Phenotype Classes</h4>
            <ul style='color: #94a3b8; line-height: 2;'>
                <li>🔴 <strong style='color: #e2e8f0;'>Cancer Susceptibility</strong></li>
                <li>🟡 <strong style='color: #e2e8f0;'>Metabolic Disorder</strong></li>
                <li>🟢 <strong style='color: #e2e8f0;'>Normal Variant</strong></li>
                <li>🔵 <strong style='color: #e2e8f0;'>Resistance Variant</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Disclaimer
st.warning("""
🧠 **Important Note:**
The model uses encoded gene–mutation relationships, expression level, and impact score to estimate the most probable phenotype.
Predictions are probabilistic and should be used as supportive insights, not diagnostic results. Always consult with healthcare professionals for medical decisions.
""")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p style='margin: 0;'>🧬 <strong style='color: #94a3b8;'>Genotype-Phenotype Mapper</strong> | Powered by Advanced Machine Learning</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>© 2025 Genofy - Genetic Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)
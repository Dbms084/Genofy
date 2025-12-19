import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import heapq
import time

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Gene Stability Predictor | Genofy", page_icon="🛡️", layout="wide")

# -----------------------------
# Modern Dark Theme CSS (Genofy-inspired)
# -----------------------------
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
    
    /* Feature Explanation */
    .feature-explanation {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0 2rem 0;
        border: 1px solid rgba(59, 130, 246, 0.2);
        animation: fadeIn 0.8s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .feature-explanation h4 {
        color: #63b3ed !important;
        margin-bottom: 0.5rem !important;
    }
    
    .feature-explanation p {
        color: #cbd5e1 !important;
        margin: 0.3rem 0 !important;
        font-size: 0.95rem;
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
    .stSlider > div > div > div {
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
    
    /* Stability Card */
    .stability-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 8px 30px rgba(0,0,0,0.2);
        animation: slideIn 0.6s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .stability-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(99, 179, 237, 0.2);
    }
    
    /* Gauge Container */
    .gauge-container {
        background: rgba(255, 255, 255, 0.03);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        animation: fadeIn 0.8s ease-out 0.2s both;
    }
    
    .gauge {
        height: 30px;
        background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
        border-radius: 15px;
        margin: 1.5rem 0;
        position: relative;
        box-shadow: 0 4px 15px rgba(99, 179, 237, 0.3);
    }
    
    .gauge-marker {
        position: absolute;
        top: -8px;
        width: 5px;
        height: 46px;
        background: #63b3ed;
        border-radius: 3px;
        box-shadow: 0 2px 8px rgba(99, 179, 237, 0.5);
        transition: left 1s ease-in-out;
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
    
    /* Ranking Animation */
    .ranking-row {
        animation: fadeInUp 0.6s ease-out;
        margin: 0.5rem 0;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model and Components
# -----------------------------
MODEL_PATH = "models/stability_index_model.pkl"
SCALER_PATH = "models/stability_index_scaler.pkl"
FEATURES_PATH = "models/stability_index_features.pkl"

def load_stability_components():
    """Load the trained stability model and preprocessing components"""
    if not all(os.path.exists(path) for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
        return None, None, None, "⚠️ Model files not found. Please train the stability model first."
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        return model, scaler, feature_names, None
    except Exception as e:
        return None, None, None, f"⚠️ Error loading model: {e}"

# -----------------------------
# DSA: Priority Queue for Top Stable Genes
# -----------------------------
class GeneStabilityHeap:
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.heap = []
    
    def add_gene(self, gene_name, stability_score, features):
        """Add gene to heap, maintaining only top N most stable genes"""
        # Use negative score for min-heap to work as max-heap for stability
        heapq.heappush(self.heap, (stability_score, gene_name, features))
        if len(self.heap) > self.max_size:
            heapq.heappop(self.heap)
    
    def get_top_genes(self):
        """Get top stable genes in descending order"""
        return sorted(self.heap, key=lambda x: x[0], reverse=True)
    
    def clear_heap(self):
        """Clear all genes from heap"""
        self.heap = []

# -----------------------------
# Page Header
# -----------------------------
st.markdown("<h1>🛡️ Gene Stability Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict gene stability under mutation stress and expression variability with advanced ML algorithms</p>", unsafe_allow_html=True)

# Feature Explanation
st.markdown("""
<div class='feature-explanation'>
    <h4>🎯 What This Tool Does</h4>
    <p>• <strong>Predict Stability:</strong> Machine learning model predicts gene stability score (0.0-1.0) based on 8 key genetic parameters</p>
    <p>• <strong>Real-time Ranking:</strong> Automatically ranks your analyzed genes using priority queue data structure</p>
    <p>• <strong>Visual Analytics:</strong> Interactive gauges and animations help understand stability levels</p>
    <p>• <strong>Research Ready:</strong> Designed for genetic researchers and bioinformaticians</p>
</div>
""", unsafe_allow_html=True)

# Load model
model, scaler, feature_names, error = load_stability_components()

if error:
    st.error(error)
    st.info("💡 Please ensure the stability model is trained and saved in the models directory")
    st.stop()

# Success metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🤖 Model Status", "Online", delta="Ready")
with col2:
    st.metric("📊 Features", len(feature_names), delta="Loaded")
with col3:
    st.metric("🎯 Target", "Stability Index", delta="0.0-1.0")

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Single Gene Prediction
# -----------------------------
st.markdown("<div class='section-header'><h3>🔍 Single Gene Stability Prediction</h3></div>", unsafe_allow_html=True)

with st.form("stability_prediction"):
    st.markdown("#### 🧬 Genetic Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amino_acid_length = st.number_input(
            "🔢 Amino Acid Length", 
            min_value=50, 
            max_value=5000, 
            value=300,
            help="Length of the protein sequence"
        )
        hydrophobicity_index = st.number_input(
            "💧 Hydrophobicity Index", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            format="%.3f",
            help="Measure of protein hydrophobicity"
        )
        gc_content = st.slider(
            "🧬 GC Content (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=50.0,
            help="Percentage of G-C base pairs"
        )
        binding_affinity = st.number_input(
            "⚡ Binding Affinity (kcal/mol)", 
            min_value=-20.0, 
            max_value=0.0, 
            value=-8.5,
            format="%.2f",
            help="Protein binding energy"
        )
    
    with col2:
        thermal_stability = st.number_input(
            "🔥 Thermal Stability (°C)", 
            min_value=0.0, 
            max_value=100.0, 
            value=37.0,
            help="Temperature stability threshold"
        )
        mutation_frequency = st.slider(
            "📈 Mutation Frequency", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.01,
            format="%.4f",
            help="Frequency of mutations in population"
        )
        expression_level = st.slider(
            "📊 Expression Level (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=75.0,
            help="Gene expression percentage"
        )
        interaction_score = st.number_input(
            "🔗 Interaction Score", 
            min_value=0.0, 
            max_value=10.0, 
            value=7.5,
            format="%.2f",
            help="Protein interaction strength"
        )
    
    gene_name = st.text_input(
        "🏷️ Gene Name (Required for Ranking)",
        placeholder="e.g., TP53, BRCA1, EGFR",
        help="Enter gene name to include in ranking system"
    )
    
    submitted = st.form_submit_button("🧠 Predict Stability Index")

# Initialize heap for top genes
if 'stability_heap' not in st.session_state:
    st.session_state.stability_heap = GeneStabilityHeap(max_size=5)

if submitted:
    try:
        # Prepare input data
        input_data = pd.DataFrame([{
            "Amino_Acid_Length": amino_acid_length,
            "Hydrophobicity_Index": hydrophobicity_index,
            "GC_Content(%)": gc_content,
            "Binding_Affinity(kcal/mol)": binding_affinity,
            "Thermal_Stability(°C)": thermal_stability,
            "Mutation_Frequency": mutation_frequency,
            "Expression_Level(%)": expression_level,
            "Interaction_Score": interaction_score
        }])
        
        # Scale features
        X_scaled = scaler.transform(input_data)
        
        # Predict stability
        stability_score = model.predict(X_scaled)[0]
        stability_score = max(0.0, min(1.0, stability_score))  # Clamp between 0-1
        
        # Add to heap if gene name provided
        if gene_name.strip():
            features_dict = {
                "Amino_Acid_Length": amino_acid_length,
                "GC_Content": gc_content,
                "Mutation_Frequency": mutation_frequency,
                "Hydrophobicity_Index": hydrophobicity_index,
                "Binding_Affinity": binding_affinity,
                "Thermal_Stability": thermal_stability,
                "Expression_Level": expression_level,
                "Interaction_Score": interaction_score
            }
            st.session_state.stability_heap.add_gene(gene_name.strip(), stability_score, features_dict)
        
        # Determine stability level
        if stability_score < 0.3:
            level = "Highly Unstable"
            color = "#ef4444"
            emoji = "🔴"
            description = "Gene is highly susceptible to mutations and environmental stress"
            gradient = "linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.05) 100%)"
        elif stability_score < 0.7:
            level = "Moderately Stable"
            color = "#f59e0b"
            emoji = "🟡"
            description = "Gene shows moderate resistance to mutations and stress"
            gradient = "linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.05) 100%)"
        else:
            level = "Highly Stable"
            color = "#10b981"
            emoji = "🟢"
            description = "Gene maintains function well under stress conditions"
            gradient = "linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.05) 100%)"
        
        # Display results with animation
        with st.spinner('🔄 Analyzing gene stability...'):
            time.sleep(1)  # Simulate processing time for better UX
        
        st.success("✅ Stability Prediction Completed!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class='stability-card' style='border-color: {color}40; background: {gradient}; backdrop-filter: blur(10px);'>
                <div style='text-align: center;'>
                    <h3 style='color: {color}; margin: 0; font-size: 1.5rem;'>{emoji} Prediction Result</h3>
                    <h1 style='color: {color}; margin: 1rem 0; font-size: 5rem; font-weight: 800;'>{stability_score:.4f}</h1>
                    <p style='font-size: 1.5rem; color: #e2e8f0; margin: 0; font-weight: 600;'>Level: <strong>{level}</strong></p>
                    <p style='color: #94a3b8; margin: 1.5rem 0 0 0; font-size: 1.05rem;'>{description}</p>
                    {f"<p style='color: #63b3ed; margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.1); font-weight: 600;'>Gene: {gene_name}</p>" if gene_name.strip() else "<p style='color: #f59e0b; margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.1);'>⚠️ No gene name provided - not added to ranking</p>"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            marker_position = stability_score * 100
            st.markdown(f"""
            <div class='gauge-container'>
                <h4 style='color: #ffffff; margin-bottom: 1.5rem;'>📊 Stability Scale</h4>
                <div class='gauge'>
                    <div class='gauge-marker' style='left: calc({marker_position}% - 2.5px);'></div>
                </div>
                <div style='display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; margin-top: 1rem;'>
                    <span>0.0<br>Unstable</span>
                    <span>0.5<br>Moderate</span>
                    <span>1.0<br>Stable</span>
                </div>
                <div style='text-align: center; color: #63b3ed; margin-top: 1.5rem; font-weight: 700; font-size: 1.2rem;'>
                    Position: {marker_position:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Top Stable Genes Ranking
# -----------------------------
st.markdown("<div class='section-header'><h3>🏆 Top Stable Genes Ranking</h3></div>", unsafe_allow_html=True)

# Add ranking explanation
st.markdown("""
<div class='feature-explanation'>
    <h4>📈 How Ranking Works</h4>
    <p>• <strong>Priority Queue:</strong> Uses min-heap data structure to efficiently track top 5 most stable genes</p>
    <p>• <strong>Automatic Updates:</strong> New predictions automatically update the rankings in real-time</p>
    <p>• <strong>Limited Memory:</strong> Only stores top performers to optimize performance</p>
    <p>• <strong>Gene Name Required:</strong> Enter gene name in prediction form to include in rankings</p>
</div>
""", unsafe_allow_html=True)

top_genes = st.session_state.stability_heap.get_top_genes()

if top_genes:
    # Add clear button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear All Rankings", type="secondary"):
            st.session_state.stability_heap.clear_heap()
            st.rerun()
    
    st.markdown("#### 📊 Current Top Stable Genes")
    
    # Create animated ranking display
    for i, (score, gene_name, features) in enumerate(top_genes):
        # Determine medal emoji
        if i == 0:
            medal = "🥇"
            badge_color = "#FFD700"
        elif i == 1:
            medal = "🥈" 
            badge_color = "#C0C0C0"
        elif i == 2:
            medal = "🥉"
            badge_color = "#CD7F32"
        else:
            medal = f"#{i+1}"
            badge_color = "#63b3ed"
        
        # Create progress bar for visual ranking
        progress = int(score * 100)
        
        st.markdown(f"""
        <div class='ranking-row' style='animation-delay: {i * 0.1}s;'>
            <div style='background: rgba(255, 255, 255, 0.03); padding: 1.5rem; border-radius: 12px; 
                        border: 1px solid rgba(255, 255, 255, 0.08); margin: 0.5rem 0;'>
                <div style='display: flex; justify-content: between; align-items: center; margin-bottom: 1rem;'>
                    <div style='display: flex; align-items: center; gap: 1rem;'>
                        <span style='background: {badge_color}; color: white; padding: 0.5rem 1rem; 
                                    border-radius: 8px; font-weight: 700; font-size: 1.1rem;'>
                            {medal}
                        </span>
                        <h4 style='color: #ffffff; margin: 0; font-size: 1.3rem;'>{gene_name}</h4>
                    </div>
                    <div style='text-align: right;'>
                        <div style='font-size: 1.5rem; font-weight: 700; color: #63b3ed;'>{score:.4f}</div>
                        <div style='font-size: 0.8rem; color: #94a3b8;'>Stability Score</div>
                    </div>
                </div>
                <div style='background: rgba(255, 255, 255, 0.1); height: 8px; border-radius: 4px; margin: 0.5rem 0;'>
                    <div style='background: linear-gradient(90deg, #10b981, #63b3ed); height: 100%; 
                                width: {progress}%; border-radius: 4px; transition: width 1s ease;'></div>
                </div>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem; 
                            font-size: 0.85rem; color: #94a3b8;'>
                    <div>🔢 AA Length: <strong>{features.get('Amino_Acid_Length', 'N/A')}</strong></div>
                    <div>🧬 GC Content: <strong>{features.get('GC_Content', 'N/A')}%</strong></div>
                    <div>📈 Mutation: <strong>{features.get('Mutation_Frequency', 'N/A'):.4f}</strong></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # DSA explanation
    with st.expander("🧠 Data Structure & Algorithm Details"):
        st.markdown(f"""
        **Current Heap State:**
        - **Heap Size:** {len(st.session_state.stability_heap.heap)} genes
        - **Max Capacity:** {st.session_state.stability_heap.max_size} genes
        - **Data Structure:** Min-Heap Priority Queue
        
        **Algorithm Complexity:**
        - **Insertion Time:** O(log n) - Efficient addition of new genes
        - **Retrieval Time:** O(n log n) - Sorting for display
        - **Space Complexity:** O(k) - Only stores top k genes
        
        **How it Works:**
        1. Each gene prediction with a name gets added to the min-heap
        2. The heap automatically maintains the top {st.session_state.stability_heap.max_size} most stable genes
        3. When capacity is exceeded, the least stable gene gets removed
        4. Display shows genes sorted by stability score (highest first)
        
        **Real-time Operations:**
        ```python
        # Adding a new gene
        heapq.heappush(heap, (score, gene_name, features))
        
        # Maintaining size limit
        if len(heap) > max_size:
            heapq.heappop(heap)  # Remove least stable
        ```
        """)
        
else:
    st.info("💡 Predict stability for genes with names to see them ranked here!")
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.03); padding: 3rem; border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.08); text-align: center; margin-top: 2rem;'>
        <h3 style='color: #94a3b8; margin: 0;'>🏆 No genes ranked yet</h3>
        <p style='color: #64748b; margin-top: 1rem; font-size: 1.1rem;'>
            Enter a gene name in the prediction form above to start building your leaderboard!
        </p>
        <div style='margin-top: 2rem; font-size: 3rem;'>🔬</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p style='margin: 0;'>🛡️ <strong style='color: #94a3b8;'>Gene Stability Predictor</strong> | Powered by Machine Learning & Priority Queues</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Stability Index: 0.0-0.3 (Unstable) | 0.3-0.7 (Moderate) | 0.7-1.0 (Stable)</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>© 2025 Genofy - Genetic Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)
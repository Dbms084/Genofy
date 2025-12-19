import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# Import helper functions
from dsa.graph_utils import find_shortest_path, analyze_path, visualize_path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Genetic Pathway Analyzer | Genofy", page_icon="🧬", layout="wide")

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
    
    /* Stats Cards */
    .stats-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(99, 179, 237, 0.2);
        border-color: rgba(99, 179, 237, 0.3);
    }
    
    /* Path Result Card */
    .path-result {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid rgba(16, 185, 129, 0.3);
        margin: 2rem 0;
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15);
        backdrop-filter: blur(10px);
    }
    
    .path-arrow {
        color: #63b3ed;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0 0.8rem;
    }
    
    /* Select Boxes */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(99, 179, 237, 0.3) !important;
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
        padding: 0.875rem 2.5rem;
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
    
    /* Analysis Box */
    .analysis-box {
        background: rgba(255, 255, 255, 0.03);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .analysis-box:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(99, 179, 237, 0.2);
    }
    
    /* Divider */
    hr {
        margin: 3rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Info Messages */
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
    
    /* Badge */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.1) 100%);
        color: #63b3ed;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Graph Explanation */
    .graph-explanation {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(124, 58, 237, 0.05) 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(139, 92, 246, 0.3);
        margin: 2rem 0;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Page Header
# -----------------------------
st.markdown("<h1>🧬 Genetic Pathway Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Discover optimal biological pathways using advanced graph algorithms and network analysis</p>", unsafe_allow_html=True)

# -----------------------------
# Load Dataset
# -----------------------------
data_path = "data/gene_interaction_dsa.csv"

if not os.path.exists(data_path):
    st.error("❌ Dataset not found! Please ensure 'gene_interaction_dsa.csv' exists in the 'data' folder.")
    st.stop()

df = pd.read_csv(data_path)

required_cols = [
    "Gene", "Protein", "Pathway_Type", "Interaction_Strength",
    "Path_Distance", "Energy_Cost", "Reliability_Score"
]

if not all(col in df.columns for col in required_cols):
    st.error("❌ Missing required columns in dataset. Please check file format.")
    st.stop()

# Success metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Dataset", "Loaded", delta="Success")
with col2:
    st.metric("🧬 Genes", len(df["Gene"].unique()))
with col3:
    st.metric("🔬 Proteins", len(df["Protein"].unique()))
with col4:
    st.metric("🔗 Interactions", len(df))

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Build Graph
# -----------------------------
G = nx.DiGraph()
for _, row in df.iterrows():
    weight = row["Path_Distance"] + row["Energy_Cost"] - (0.5 * row["Reliability_Score"])
    G.add_edge(row["Gene"], row["Protein"], weight=weight)

# Graph statistics
st.markdown("<div class='section-header'><h3>📈 Network Statistics</h3></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='stats-card'>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>🌐</div>
        <h4 style='color: #63b3ed; margin: 0; font-size: 0.95rem;'>Total Nodes</h4>
        <h2 style='color: #ffffff; margin: 0.5rem 0; font-size: 2.5rem;'>{}</h2>
        <p style='color: #94a3b8; margin: 0; font-size: 0.9rem;'>Genes + Proteins</p>
    </div>
    """.format(len(G.nodes)), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='stats-card'>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>🔗</div>
        <h4 style='color: #8b5cf6; margin: 0; font-size: 0.95rem;'>Connections</h4>
        <h2 style='color: #ffffff; margin: 0.5rem 0; font-size: 2.5rem;'>{}</h2>
        <p style='color: #94a3b8; margin: 0; font-size: 0.9rem;'>Directed Edges</p>
    </div>
    """.format(len(G.edges)), unsafe_allow_html=True)

with col3:
    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes) if len(G.nodes) > 0 else 0
    st.markdown("""
    <div class='stats-card'>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>📊</div>
        <h4 style='color: #10b981; margin: 0; font-size: 0.95rem;'>Avg Degree</h4>
        <h2 style='color: #ffffff; margin: 0.5rem 0; font-size: 2.5rem;'>{:.2f}</h2>
        <p style='color: #94a3b8; margin: 0; font-size: 0.9rem;'>Connections/Node</p>
    </div>
    """.format(avg_degree), unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Pathway Exploration Section
# -----------------------------
st.markdown("<div class='section-header'><h3>🔍 Pathway Exploration</h3></div>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; margin-bottom: 2rem;'>Select a start gene and target protein to find the optimal biological pathway using Dijkstra's algorithm</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    start_gene = st.selectbox(
        "🧬 Start Gene",
        options=sorted(df["Gene"].unique()),
        help="Select the starting gene for pathway analysis"
    )
with col2:
    target_protein = st.selectbox(
        "🔬 Target Protein",
        options=sorted(df["Protein"].unique()),
        help="Select the target protein to reach"
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Pathfinding
# -----------------------------
if st.button("🚀 Find Shortest Pathway"):
    with st.spinner("🧠 Computing optimal pathway using graph algorithms..."):
        path, distance = find_shortest_path(G, start_gene, target_protein)

    if path:
        # Success message
        st.success("✅ Optimal Biological Pathway Found!")
        
        # Display path
        st.markdown("<div class='section-header'><h3>🗺️ Pathway Route</h3></div>", unsafe_allow_html=True)
        path_html = " <span class='path-arrow'>→</span> ".join([f"<strong style='color: #63b3ed;'>{node}</strong>" for node in path])
        st.markdown(f"""
        <div class='path-result'>
            <h4 style='color: #ffffff; margin-bottom: 1.5rem; font-size: 1.2rem;'>Pathway Sequence:</h4>
            <div style='font-size: 1.3rem; color: #e2e8f0; line-height: 2;'>{path_html}</div>
            <br>
            <div style='background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 12px; margin-top: 2rem;
                        border: 1px solid rgba(255, 255, 255, 0.1);'>
                <strong style='color: #94a3b8; font-size: 1rem;'>Total Weighted Distance:</strong> 
                <span style='color: #10b981; font-size: 2rem; font-weight: 800; margin-left: 1rem;'>{distance:.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Path metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Path Length", f"{len(path)} nodes", delta="Optimal")
        with col2:
            st.metric("⚡ Transitions", f"{len(path) - 1} steps", delta="Minimal")
        with col3:
            st.metric("📊 Efficiency", f"{distance/len(path):.2f}", delta="Avg weight")
        
        st.markdown("<hr>")
        
        # -----------------------------
        # Path Analysis
        # -----------------------------
        st.markdown("<div class='section-header'><h3>📊 Detailed Pathway Analysis</h3></div>", unsafe_allow_html=True)
        analysis = analyze_path(G, path)
        
        # Display analysis in formatted way
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='analysis-box'>
                <h4 style='color: #63b3ed; font-size: 1rem; margin-bottom: 1rem;'>🔢 Node Count</h4>
                <p style='font-size: 2.5rem; font-weight: 800; color: #ffffff; margin: 0;'>{}</p>
            </div>
            """.format(analysis.get("node_count", "N/A")), unsafe_allow_html=True)
            
            st.markdown("""
            <div class='analysis-box'>
                <h4 style='color: #8b5cf6; font-size: 1rem; margin-bottom: 1rem;'>🔗 Edge Count</h4>
                <p style='font-size: 2.5rem; font-weight: 800; color: #ffffff; margin: 0;'>{}</p>
            </div>
            """.format(analysis.get("edge_count", "N/A")), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='analysis-box'>
                <h4 style='color: #10b981; font-size: 1rem; margin-bottom: 1rem;'>⚡ Total Weight</h4>
                <p style='font-size: 2.5rem; font-weight: 800; color: #ffffff; margin: 0;'>{:.3f}</p>
            </div>
            """.format(analysis.get("total_weight", 0)), unsafe_allow_html=True)
            
            st.markdown("""
            <div class='analysis-box'>
                <h4 style='color: #f59e0b; font-size: 1rem; margin-bottom: 1rem;'>📈 Avg Weight</h4>
                <p style='font-size: 2.5rem; font-weight: 800; color: #ffffff; margin: 0;'>{:.3f}</p>
            </div>
            """.format(analysis.get("average_weight", 0)), unsafe_allow_html=True)
        
        st.markdown("<hr>")
        
        # -----------------------------
        # Visualization with Explanation
        # -----------------------------
        st.markdown("<div class='section-header'><h3>🎨 Network Visualization</h3></div>", unsafe_allow_html=True)
        
        # Graph explanation
        st.markdown(f"""
        <div class='graph-explanation'>
            <h4 style='color: #8b5cf6; margin-bottom: 1rem;'>📖 Graph Interpretation Guide</h4>
            <div style='color: #e2e8f0; line-height: 1.8;'>
                <p><strong>🧬 Nodes (Circles):</strong> Represent biological entities (genes in blue, proteins in green)</p>
                <p><strong>🔗 Edges (Arrows):</strong> Show interactions between entities with direction</p>
                <p><strong>🎯 Pathway Highlight:</strong> The <span style='color: #ff6b6b;'>red path</span> shows your optimal route from <strong>{start_gene}</strong> to <strong>{target_protein}</strong></p>
                <p><strong>📏 Distance Metric:</strong> Total weighted distance of <strong>{distance:.3f}</strong> combines path distance, energy cost, and reliability</p>
                <p><strong>🔍 Layout:</strong> Nodes are positioned using force-directed layout for optimal visualization</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create and display the visualization
        fig = visualize_path(G, path)
        # Ensure the figure has dark background
        fig.patch.set_facecolor('#0f1b2d')
        st.pyplot(fig, use_container_width=True)
        
        # Additional distance explanation
        with st.expander("🔬 Understanding the Distance Metric"):
            st.markdown("""
            ### 📊 How the Pathway Distance is Calculated
            
            The total weighted distance **({distance:.3f})** is computed using Dijkstra's algorithm with edge weights determined by:
            
            ```
            Weight = Path_Distance + Energy_Cost - (0.5 × Reliability_Score)
            ```
            
            **Components:**
            - **Path Distance**: Physical/functional distance between entities
            - **Energy Cost**: Metabolic energy required for the interaction  
            - **Reliability Score**: Confidence in the interaction (higher = better)
            
            **Interpretation:**
            - Lower total distance = More efficient biological pathway
            - The algorithm finds the path with minimum cumulative weight
            - This represents the most optimal route considering multiple biological factors
            """.format(distance=distance))
        
    else:
        st.error("⚠️ No valid pathway found between the selected gene and protein.")
        st.info("💡 Try selecting different nodes or check if they're connected in the network")
        
        # Show alternative suggestions
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.03); padding: 2rem; border-radius: 16px;
                    border: 1px solid rgba(255, 255, 255, 0.08); margin-top: 2rem;'>
            <h4 style='color: #63b3ed; margin-bottom: 1rem;'>Troubleshooting Tips:</h4>
            <ul style='color: #94a3b8; line-height: 2;'>
                <li>Verify that the selected gene and protein exist in the network</li>
                <li>Check if there's a valid path between the nodes</li>
                <li>Try selecting different start/target combinations</li>
                <li>Some nodes might be disconnected in the current network</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p style='margin: 0;'>🧬 <strong style='color: #94a3b8;'>Genetic Pathway Analyzer</strong> | Powered by Graph Theory & Network Science</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>© 2025 Genofy - Advanced Genetic Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Genetic Inheritance Predictor | Genofy", page_icon="👨👩👧👦", layout="wide")

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
    
    /* Select Boxes */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        transition: all 0.3s ease !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Checkbox */
    .stCheckbox {
        color: #e2e8f0;
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
    
    /* Inheritance Card */
    .inheritance-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #3b82f6;
        transition: all 0.3s ease;
    }
    
    .inheritance-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(99, 179, 237, 0.2);
        border-left-color: #63b3ed;
    }
    
    /* Parent Card */
    .parent-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px solid rgba(59, 130, 246, 0.3);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Probability Badge */
    .probability-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.3rem;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
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
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Backend: Genetic Inheritance Logic
# -----------------------------
class GeneticInheritancePredictor:
    def __init__(self):
        self.trait_database = self._load_trait_database()
    
    def _load_trait_database(self):
        """Load common genetic traits and their inheritance patterns"""
        return {
            "eye_color": {
                "name": "👁 Eye Color",
                "type": "polygenic",
                "genes": ["OCA2", "HERC2"],
                "traits": {
                    "brown": {"genotype": "BB/Bb", "dominance": "dominant"},
                    "blue": {"genotype": "bb", "dominance": "recessive"},
                    "green": {"genotype": "GG/Gb", "dominance": "intermediate"},
                    "hazel": {"genotype": "HH/Hb", "dominance": "intermediate"}
                }
            },
            "hair_color": {
                "name": "💇 Hair Color",
                "type": "polygenic",
                "genes": ["MC1R", "TYRP1"],
                "traits": {
                    "black": {"genotype": "BKBK", "dominance": "dominant"},
                    "brown": {"genotype": "BRBR", "dominance": "dominant"},
                    "blonde": {"genotype": "BLBL", "dominance": "recessive"},
                    "red": {"genotype": "RDRD", "dominance": "recessive"}
                }
            },
            "blood_type": {
                "name": "💉 Blood Type",
                "type": "codominant",
                "genes": ["ABO"],
                "traits": {
                    "A": {"genotype": "AA/AO", "dominance": "codominant"},
                    "B": {"genotype": "BB/BO", "dominance": "codominant"},
                    "AB": {"genotype": "AB", "dominance": "codominant"},
                    "O": {"genotype": "OO", "dominance": "recessive"}
                }
            },
            "lactose_tolerance": {
                "name": "🥛 Lactose Tolerance",
                "type": "mendelian",
                "genes": ["LCT"],
                "traits": {
                    "tolerant": {"genotype": "TT/Tt", "dominance": "dominant"},
                    "intolerant": {"genotype": "tt", "dominance": "recessive"}
                }
            },
            "height_potential": {
                "name": "📏 Height Potential",
                "type": "polygenic",
                "genes": ["HMGA2", "GDF5"],
                "traits": {
                    "tall": {"genotype": "multiple", "dominance": "additive"},
                    "average": {"genotype": "multiple", "dominance": "additive"},
                    "short": {"genotype": "multiple", "dominance": "additive"}
                }
            }
        }
    
    def predict_mendelian_inheritance(self, father_trait, mother_trait, trait_type):
        """Predict inheritance for Mendelian traits"""
        predictions = {}
        
        if trait_type == "blood_type":
            blood_punnett = {
                ("A", "A"): {"A": 1.0, "B": 0.0, "AB": 0.0, "O": 0.0},
                ("A", "B"): {"A": 0.25, "B": 0.25, "AB": 0.5, "O": 0.0},
                ("A", "AB"): {"A": 0.5, "B": 0.0, "AB": 0.5, "O": 0.0},
                ("A", "O"): {"A": 0.5, "B": 0.0, "AB": 0.0, "O": 0.5},
                ("B", "B"): {"A": 0.0, "B": 1.0, "AB": 0.0, "O": 0.0},
                ("B", "AB"): {"A": 0.0, "B": 0.5, "AB": 0.5, "O": 0.0},
                ("B", "O"): {"A": 0.0, "B": 0.5, "AB": 0.0, "O": 0.5},
                ("AB", "AB"): {"A": 0.25, "B": 0.25, "AB": 0.5, "O": 0.0},
                ("AB", "O"): {"A": 0.5, "B": 0.5, "AB": 0.0, "O": 0.0},
                ("O", "O"): {"A": 0.0, "B": 0.0, "AB": 0.0, "O": 1.0}
            }
            predictions = blood_punnett.get((father_trait, mother_trait), 
                                          blood_punnett.get((mother_trait, father_trait), {}))
            
        elif trait_type == "lactose_tolerance":
            if father_trait == "tolerant" and mother_trait == "tolerant":
                predictions = {"tolerant": 0.75, "intolerant": 0.25}
            elif father_trait == "intolerant" and mother_trait == "intolerant":
                predictions = {"tolerant": 0.0, "intolerant": 1.0}
            else:
                predictions = {"tolerant": 0.5, "intolerant": 0.5}
                
        return predictions
    
    def predict_polygenic_trait(self, father_trait, mother_trait, trait_type):
        """Predict inheritance for polygenic traits"""
        predictions = {}
        
        if trait_type == "eye_color":
            eye_color_rules = {
                ("brown", "brown"): {"brown": 0.75, "green": 0.19, "blue": 0.06, "hazel": 0.0},
                ("brown", "blue"): {"brown": 0.5, "green": 0.12, "blue": 0.38, "hazel": 0.0},
                ("brown", "green"): {"brown": 0.5, "green": 0.38, "blue": 0.12, "hazel": 0.0},
                ("blue", "blue"): {"brown": 0.0, "green": 0.01, "blue": 0.99, "hazel": 0.0},
                ("green", "green"): {"brown": 0.25, "green": 0.59, "blue": 0.16, "hazel": 0.0},
                ("hazel", "hazel"): {"brown": 0.22, "green": 0.28, "blue": 0.15, "hazel": 0.35}
            }
            predictions = eye_color_rules.get((father_trait, mother_trait), 
                                            eye_color_rules.get((mother_trait, father_trait), {}))
            
        elif trait_type == "hair_color":
            hair_color_rules = {
                ("black", "black"): {"black": 0.95, "brown": 0.05, "blonde": 0.0, "red": 0.0},
                ("black", "brown"): {"black": 0.5, "brown": 0.5, "blonde": 0.0, "red": 0.0},
                ("brown", "brown"): {"black": 0.25, "brown": 0.5, "blonde": 0.25, "red": 0.0},
                ("brown", "blonde"): {"black": 0.0, "brown": 0.5, "blonde": 0.5, "red": 0.0},
                ("blonde", "blonde"): {"black": 0.0, "brown": 0.25, "blonde": 0.75, "red": 0.0},
                ("red", "red"): {"black": 0.0, "brown": 0.0, "blonde": 0.0, "red": 1.0}
            }
            predictions = hair_color_rules.get((father_trait, mother_trait), 
                                             hair_color_rules.get((mother_trait, father_trait), {}))
            
        elif trait_type == "height_potential":
            height_combinations = {
                ("tall", "tall"): {"tall": 0.8, "average": 0.2, "short": 0.0},
                ("tall", "average"): {"tall": 0.5, "average": 0.45, "short": 0.05},
                ("tall", "short"): {"tall": 0.25, "average": 0.6, "short": 0.15},
                ("average", "average"): {"tall": 0.25, "average": 0.5, "short": 0.25},
                ("average", "short"): {"tall": 0.1, "average": 0.45, "short": 0.45},
                ("short", "short"): {"tall": 0.0, "average": 0.25, "short": 0.75}
            }
            predictions = height_combinations.get((father_trait, mother_trait), 
                                                height_combinations.get((mother_trait, father_trait), {}))
            
        return predictions
    
    def predict_all_traits(self, father_traits, mother_traits):
        """Predict inheritance for all selected traits"""
        results = {}
        
        for trait_id, trait_info in self.trait_database.items():
            if trait_id in father_traits and trait_id in mother_traits:
                father_trait = father_traits[trait_id]
                mother_trait = mother_traits[trait_id]
                
                if trait_info["type"] in ["mendelian", "codominant"]:
                    predictions = self.predict_mendelian_inheritance(
                        father_trait, mother_trait, trait_id
                    )
                else:
                    predictions = self.predict_polygenic_trait(
                        father_trait, mother_trait, trait_id
                    )
                
                results[trait_id] = {
                    "name": trait_info["name"],
                    "father_trait": father_trait,
                    "mother_trait": mother_trait,
                    "predictions": predictions,
                    "type": trait_info["type"]
                }
        
        return results

    def _get_inheritance_explanation(self, inheritance_type):
        explanations = {
            "mendelian": "Follows classic Mendelian inheritance with dominant-recessive patterns",
            "codominant": "Both alleles are expressed equally in the phenotype",
            "polygenic": "Influenced by multiple genes with additive effects"
        }
        return explanations.get(inheritance_type, "Complex genetic inheritance pattern")

# -----------------------------
# Initialize Predictor
# -----------------------------
predictor = GeneticInheritancePredictor()

# -----------------------------
# Page Header
# -----------------------------
st.markdown("<h1>👨👩👧👦 Genetic Inheritance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict inherited traits based on parental genetic characteristics using Mendelian genetics and probability theory</p>", unsafe_allow_html=True)

# Feature Explanation
st.markdown("""
<div class='feature-explanation'>
    <h4>🎯 What This Tool Does</h4>
    <p>• <strong>Predict Genetic Inheritance:</strong> Calculate probability of traits being passed from parents to children</p>
    <p>• <strong>Multiple Inheritance Patterns:</strong> Supports Mendelian, Codominant, and Polygenic inheritance</p>
    <p>• <strong>Family Simulation:</strong> Generate simulated children based on genetic probabilities</p>
    <p>• <strong>Visual Analytics:</strong> Interactive charts and probability distributions</p>
    <p>• <strong>Educational Insights:</strong> Learn about genetic patterns and inheritance mechanisms</p>
</div>
""", unsafe_allow_html=True)

# Success metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🧬 Traits", len(predictor.trait_database))
with col2:
    st.metric("🧪 Genes", "15+")
with col3:
    st.metric("🎯 Patterns", "3 Types")
with col4:
    st.metric("📊 Accuracy", "High")

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Parental Traits Input
# -----------------------------
st.markdown("<div class='section-header'><h3>👪 Parental Genetic Traits</h3></div>", unsafe_allow_html=True)

with st.form("inheritance_prediction"):
    col1, col2 = st.columns(2)
    
    # Father's Traits
    with col1:
        st.markdown("""
        <div class='parent-card'>
            <h3 style='color: #63b3ed; text-align: center; margin: 0;'>👨 Father's Traits</h3>
        </div>
        """, unsafe_allow_html=True)
        
        father_traits = {}
        for trait_id, trait_info in predictor.trait_database.items():
            options = list(trait_info["traits"].keys())
            father_traits[trait_id] = st.selectbox(
                f"{trait_info['name']}",
                options=options,
                key=f"father_{trait_id}"
            )
    
    # Mother's Traits  
    with col2:
        st.markdown("""
        <div class='parent-card'>
            <h3 style='color: #8b5cf6; text-align: center; margin: 0;'>👩 Mother's Traits</h3>
        </div>
        """, unsafe_allow_html=True)
        
        mother_traits = {}
        for trait_id, trait_info in predictor.trait_database.items():
            options = list(trait_info["traits"].keys())
            mother_traits[trait_id] = st.selectbox(
                f"{trait_info['name']}",
                options=options,
                key=f"mother_{trait_id}"
            )
    
    # Additional options
    st.markdown("### ⚙ Prediction Settings")
    col1, col2 = st.columns(2)
    with col1:
        num_children = st.slider("👶 Number of Children to Simulate", 1, 10, 3)
    with col2:
        include_probabilities = st.checkbox("📈 Show Detailed Probabilities", value=True)
    
    submitted = st.form_submit_button("🧬 Predict Inheritance Patterns")

# -----------------------------
# Prediction Results
# -----------------------------
if submitted:
    with st.spinner("🔬 Analyzing genetic inheritance patterns..."):
        results = predictor.predict_all_traits(father_traits, mother_traits)
    
    st.success("✅ Inheritance Prediction Completed!")
    
    # Display Results
    st.markdown("<div class='section-header'><h3>📊 Inheritance Predictions</h3></div>", unsafe_allow_html=True)
    
    for trait_id, result in results.items():
        with st.expander(f"{result['name']} - Inheritance Analysis", expanded=True):
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                st.markdown(f"""
                <div style='text-align: center;'>
                    <h4 style='color: #e2e8f0;'>👨 Father</h4>
                    <div class='probability-badge'>{result['father_trait'].title()}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='text-align: center;'>
                    <h4 style='color: #e2e8f0;'>👩 Mother</h4>
                    <div class='probability-badge'>{result['mother_trait'].title()}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("#### 🎯 Child's Possible Traits")
                
                # Create probability bars
                for trait, probability in result['predictions'].items():
                    if probability > 0:
                        percentage = probability * 100
                        st.markdown(f"""
                        <div style='margin: 0.75rem 0;'>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 0.3rem;'>
                                <span style='font-weight: 600; color: #e2e8f0;'>{trait.title()}</span>
                                <span style='color: #63b3ed; font-weight: 700;'>{percentage:.1f}%</span>
                            </div>
                            <div style='background: rgba(255, 255, 255, 0.1); border-radius: 10px; height: 12px;'>
                                <div style='background: linear-gradient(90deg, #3b82f6, #2563eb); 
                                            border-radius: 10px; height: 12px; width: {percentage}%;
                                            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4);'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # -----------------------------
    # Family Simulation
    # -----------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'><h3>👶 Simulated Family Traits</h3></div>", unsafe_allow_html=True)
    
    # Generate simulated children
    simulated_children = []
    for child_num in range(num_children):
        child_traits = {}
        for trait_id, result in results.items():
            traits = list(result['predictions'].keys())
            probabilities = list(result['predictions'].values())
            inherited_trait = np.random.choice(traits, p=probabilities)
            child_traits[trait_id] = inherited_trait
        
        simulated_children.append({
            "child": f"Child {child_num + 1}",
            **child_traits
        })
    
    # Display simulated children
    children_df = pd.DataFrame(simulated_children)
    
    # Rename columns for display
    display_columns = {"child": "👶 Child"}
    for trait_id in results.keys():
        display_columns[trait_id] = predictor.trait_database[trait_id]["name"]
    
    children_display_df = children_df.rename(columns=display_columns)
    st.dataframe(children_display_df, use_container_width=True)
    
    # -----------------------------
    # Visualization
    # -----------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'><h3>📈 Inheritance Probability Charts</h3></div>", unsafe_allow_html=True)
    
    # Set dark theme for matplotlib
    plt.style.use('dark_background')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create bar chart for one trait
        if results:
            first_trait = list(results.keys())[0]
            trait_data = results[first_trait]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f1b2d')
            ax.set_facecolor('#111c30')
            
            traits = list(trait_data['predictions'].keys())
            probabilities = [p * 100 for p in trait_data['predictions'].values()]
            
            colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b']
            bars = ax.bar(traits, probabilities, color=colors[:len(traits)], edgecolor='#1e293b', linewidth=1.5)
            
            ax.set_ylabel('Probability (%)', fontsize=12, color='#cbd5e1', fontweight='bold')
            ax.set_title(f"{trait_data['name']} - Inheritance Probabilities", fontsize=14, fontweight='bold', color='#e2e8f0', pad=20)
            ax.set_ylim(0, 100)
            ax.tick_params(colors='#94a3b8')
            ax.grid(axis='y', alpha=0.1, color='#475569')
            
            # Add value labels on bars
            for bar, probability in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{probability:.1f}%', ha='center', va='bottom', fontweight='bold', color='#e2e8f0')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    
    with col2:
        # Pie chart for trait distribution - FIXED ERROR HERE
        if len(simulated_children) > 0:
            # Use the original trait_id key, not the display name
            eye_colors = [child['eye_color'] for child in simulated_children]
            color_counts = pd.Series(eye_colors).value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor('#0f1b2d')
            
            colors_pie = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'][:len(color_counts)]
            
            wedges, texts, autotexts = ax.pie(
                color_counts.values, 
                labels=color_counts.index,
                autopct='%1.1f%%',
                colors=colors_pie,
                startangle=90,
                textprops={'fontsize': 12, 'weight': 'bold', 'color': '#e2e8f0'}
            )
            
            # Improve text appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('👁 Eye Color Distribution in Simulated Children', fontsize=14, fontweight='bold', color='#e2e8f0', pad=20)
            st.pyplot(fig)

    # -----------------------------
    # Genetic Explanation
    # -----------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'><h3>🧬 Genetic Inheritance Explanation</h3></div>", unsafe_allow_html=True)
    
    for trait_id, result in results.items():
        trait_info = predictor.trait_database[trait_id]
        
        st.markdown(f"""
        <div class='inheritance-card'>
            <h4 style='color: #63b3ed; margin-bottom: 1rem;'>{result['name']}</h4>
            <p style='margin: 0.5rem 0;'><strong style='color: #e2e8f0;'>Inheritance Type:</strong> <span style='color: #94a3b8;'>{trait_info['type'].title()}</span></p>
            <p style='margin: 0.5rem 0;'><strong style='color: #e2e8f0;'>Key Genes:</strong> <span style='color: #94a3b8;'>{', '.join(trait_info['genes'])}</span></p>
            <p style='margin: 0.5rem 0;'><strong style='color: #e2e8f0;'>Pattern:</strong> <span style='color: #94a3b8;'>{predictor._get_inheritance_explanation(trait_info['type'])}</span></p>
        </div>
        """, unsafe_allow_html=True)

else:
    # -----------------------------
    # Educational Section (Show when no prediction)
    # -----------------------------
    st.markdown("<div class='section-header'><h3>📚 Genetic Inheritance Basics</h3></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='inheritance-card'>
            <div style='font-size: 3rem; text-align: center; margin-bottom: 1rem;'>🧬</div>
            <h4 style='color: #63b3ed; text-align: center;'>Mendelian Inheritance</h4>
            <p style='text-align: center; color: #94a3b8;'>Single gene traits with clear dominant-recessive patterns (e.g., lactose tolerance)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='inheritance-card'>
            <div style='font-size: 3rem; text-align: center; margin-bottom: 1rem;'>🔀</div>
            <h4 style='color: #8b5cf6; text-align: center;'>Codominant Traits</h4>
            <p style='text-align: center; color: #94a3b8;'>Both alleles expressed equally (e.g., AB blood type)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='inheritance-card'>
            <div style='font-size: 3rem; text-align: center; margin-bottom: 1rem;'>🎯</div>
            <h4 style='color: #10b981; text-align: center;'>Polygenic Traits</h4>
            <p style='text-align: center; color: #94a3b8;'>Influenced by multiple genes (e.g., eye color, height)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show all traits info
    st.markdown("<div class='section-header'><h3>🧬 Available Traits for Analysis</h3></div>", unsafe_allow_html=True)
    
    for trait_id, trait_info in predictor.trait_database.items():
        st.markdown(f"""
        <div class='inheritance-card'>
            <h4 style='color: #63b3ed; margin-bottom: 1rem;'>{trait_info['name']}</h4>
            <p style='margin: 0.5rem 0;'><strong style='color: #e2e8f0;'>Inheritance Type:</strong> <span style='color: #94a3b8;'>{trait_info['type'].title()}</span></p>
            <p style='margin: 0.5rem 0;'><strong style='color: #e2e8f0;'>Key Genes:</strong> <span style='color: #94a3b8;'>{', '.join(trait_info['genes'])}</span></p>
            <p style='margin: 0.5rem 0;'><strong style='color: #e2e8f0;'>Pattern:</strong> <span style='color: #94a3b8;'>{predictor._get_inheritance_explanation(trait_info['type'])}</span></p>
            <p style='margin: 0.5rem 0;'><strong style='color: #e2e8f0;'>Possible Traits:</strong> <span style='color: #94a3b8;'>{', '.join([t.title() for t in trait_info['traits'].keys()])}</span></p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p style='margin: 0;'>👨👩👧👦 <strong style='color: #94a3b8;'>Genetic Inheritance Predictor</strong> | Powered by Mendelian Genetics & Probability Theory</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Note: Predictions are probabilistic and for educational purposes</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>© 2025 Genofy - Genetic Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)
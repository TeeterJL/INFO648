!pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Load pipeline ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    with open("pipeline_bundle.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_pipeline()
cluster_scaler = bundle["cluster_scaler"]
kmeans          = bundle["kmeans"]
forest_model    = bundle["forest_model"]
numeric_features = bundle["numeric_features"]
feature_cols     = bundle["feature_cols"]

MOUNTAIN_STATES = ["Arizona","Colorado","Idaho","Montana","Nevada","New Mexico","Utah","Wyoming"]

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏔️ Mountain Region Neighborhood Growth Predictor")
st.markdown("Enter a census tract's 2020 characteristics to forecast whether it will gain ≥10% population by 2030.")

st.header("Tract Characteristics")

col1, col2 = st.columns(2)

with col1:
    pct_owner   = st.slider("% Owner-Occupied Housing",  0.0, 1.0, 0.65, 0.01)
    pct_vacant  = st.slider("% Vacant Units",            0.0, 1.0, 0.10, 0.01)
    pct_kids    = st.slider("% Children (age <18)",      0.0, 1.0, 0.25, 0.01)
    pct_seniors = st.slider("% Seniors (age 65+)",       0.0, 1.0, 0.15, 0.01)
    density     = st.number_input("Population Density (per km²)", min_value=0.0, value=500.0, step=10.0)

with col2:
    pct_white = st.slider("% White (non-Hispanic)", 0.0, 1.0, 0.70, 0.01)
    pct_black = st.slider("% Black (non-Hispanic)", 0.0, 1.0, 0.05, 0.01)
    pct_hisp  = st.slider("% Hispanic",             0.0, 1.0, 0.15, 0.01)
    state     = st.selectbox("State", MOUNTAIN_STATES)
    stype     = st.selectbox("Settlement Type", ["rural", "suburban", "urban"])

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("Predict Growth", type="primary"):
    numeric_vals = [pct_owner, pct_vacant, pct_kids, pct_seniors,
                    pct_white, pct_black, pct_hisp, density]
    
    # Assign cluster using the frozen scaler + kmeans
    scaled = cluster_scaler.transform([numeric_vals])
    cluster_label = str(kmeans.predict(scaled)[0])
    
    # Build the feature row
    row = dict(zip(numeric_features, numeric_vals))
    row["settlement_type"] = stype
    row["STATE"]           = state
    row["cluster"]         = cluster_label
    
    X_input = pd.DataFrame([row])[feature_cols + ["cluster"]]
    
    prob  = forest_model.predict_proba(X_input)[0][1]
    pred  = forest_model.predict(X_input)[0]
    
    st.divider()
    
    if pred == 1:
        st.success(f"✅ **Predicted to GROW** (≥10% population gain by 2030)")
    else:
        st.warning(f"⚠️ **Predicted NOT to grow** significantly by 2030")
    
    st.metric("Growth Probability", f"{prob:.1%}")
    st.metric("Assigned Cluster", f"Cluster {cluster_label}")
    
    cluster_names = {
        "0": "Hispanic Urban Family",
        "1": "Resort / Remote Rural",
        "2": "Suburban Family Homeowner",
        "3": "Dense Urban Renter"
    }
    st.caption(f"Cluster type: {cluster_names.get(cluster_label, 'Unknown')}")
    
    st.divider()
    st.info("**Note:** This model was trained on 2010 census data to predict 2010–2020 growth. "
            "The 2030 forecast assumes that structural relationships from the 2010s still hold.")

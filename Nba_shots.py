import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

# Set up Streamlit page
st.set_page_config(
    page_title="NBA Shot Prediction",
    page_icon="🏀",
    layout="wide"
)

# Load model and scaler with error handling
@st.cache_data
def load_model_and_scaler():
    model_path = "Gradient Boosting.pkl"
    scaler_path = "StandardScaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("⚠️ Model or scaler file not found! Ensure they are in the correct directory.")
        st.stop()

    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    if not hasattr(model, 'predict'):
        st.error('⚠️ The loaded model is invalid. Please check the file.')
        st.stop()
    
    return model, scaler

# Load model and scaler
model, scaler = load_model_and_scaler()

# Initialize session state for input fields
if "shot_dist" not in st.session_state:
    st.session_state["shot_dist"] = 15
    st.session_state["close_def_dist"] = 3
    st.session_state["shot_difficulty"] = 5
    st.session_state["shot_number"] = 5
    st.session_state["age"] = 25
    st.session_state["experience_num"] = 0
    st.session_state["player_height"] = 200
    st.session_state["player_weight"] = 90
    st.session_state["match_location"] = "Home"
    st.session_state["shot_clock_remaining"] = 10
    st.session_state["touch_time"] = 2
    st.session_state["game_minutes"] = 24

# Title and app description
st.title("🏀 NBA Shot Prediction App")
st.markdown("""
This app predicts the probability of a shot being made in an NBA game based on various shot and contextual factors.
""")

# Layout for input fields
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Shot Specifications")

    shot_dist = st.slider("Shot Distance (ft)", 0, 40, st.session_state["shot_dist"])
    close_def_dist = st.slider("Closest Defender Distance (ft)", 0, 10, st.session_state["close_def_dist"])
    shot_difficulty = st.slider("Shot Difficulty (1-10)", 1, 10, st.session_state["shot_difficulty"])
    shot_number = st.slider("Shot Number in Game", 1, 20, st.session_state["shot_number"])

    age = st.slider("Player Age", 18, 40, st.session_state["age"])
    experience_num = st.selectbox("Years of Experience", list(range(21)), index=st.session_state["experience_num"])
    player_height = st.slider("Player Height (cm)", 160, 220, st.session_state["player_height"])
    player_weight = st.slider("Player Weight (kg)", 60, 150, st.session_state["player_weight"])
    bmi = player_weight / ((player_height / 100) ** 2)

    # Hidden team-related features
    home_team_code = 0  # Default hidden value
    away_team_code = 1 - home_team_code  # Automatically opposite

    match_location = st.radio("Match Location", ["Home", "Away"], index=0 if st.session_state["match_location"] == "Home" else 1)
    shot_clock_remaining = st.slider("Shot Clock Remaining (sec)", 0, 24, st.session_state["shot_clock_remaining"])
    touch_time = st.slider("Touch Time (sec)", 0, 10, st.session_state["touch_time"])
    game_minutes = st.slider("Game Minutes", 0, 48, st.session_state["game_minutes"])

# Prediction button and result display
col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🏀 Predict Shot Outcome"):
        # Prepare input features
        input_data = np.array([[  
            float(shot_dist), float(close_def_dist), float(shot_difficulty), int(shot_number),
            int(age), int(experience_num), float(player_height), float(player_weight), float(bmi),
            int(home_team_code), int(away_team_code), int(match_location),
            float(shot_clock_remaining), float(touch_time), float(game_minutes)
        ]])

        try:
            input_data_scaled = scaler.transform(input_data)

            prediction = model.predict(input_data_scaled)
            probabilities = model.predict_proba(input_data_scaled)[0]

            outcome = "Made" if prediction[0] == 1 else "Missed"
            st.success(f"🏀 Predicted Shot Outcome: **{outcome}**")
            st.write(f"### 📊 Probability of Making the Shot: {probabilities[1] * 100:.2f}%")

            # Create bar chart for probabilities
            prob_df = pd.DataFrame({
                'Outcome': ['Missed', 'Made'],
                'Probability': probabilities * 100
            })
            
            fig = px.bar(prob_df, x='Outcome', y='Probability',
                         title='Prediction Probabilities',
                         labels={'Probability': 'Probability (%)'},
                         color='Probability',
                         color_continuous_scale='Viridis')
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"⚠️ Error in prediction: {e}")

# Reset button to clear all inputs
with col2:
    if st.button("🔄 Reset Inputs"):
        # Reset session state variables
        st.session_state["shot_dist"] = 15
        st.session_state["close_def_dist"] = 3
        st.session_state["shot_difficulty"] = 5
        st.session_state["shot_number"] = 5
        st.session_state["age"] = 25
        st.session_state["experience_num"] = 0
        st.session_state["player_height"] = 200
        st.session_state["player_weight"] = 90
        st.session_state["match_location"] = "Home"
        st.session_state["shot_clock_remaining"] = 10
        st.session_state["touch_time"] = 2
        st.session_state["game_minutes"] = 24

        st.rerun()  # Refresh UI

# Sidebar Info
st.sidebar.header("📌 About")
st.sidebar.info("""
This application predicts NBA shot outcomes based on contextual and in-game parameters using **Machine Learning**.
This will help teams determine what position such as shot distance, shot difficulty, and other contextual factors players need to use to gain a good advantage in a game.

                       
### **Model Information**
- **Algorithm:** Gradient Boosting Classifier
- **Trained on:** NBA Shot and Information Dataset              
""")

st.sidebar.header("📊 Model Performance")
st.sidebar.markdown("""
- **Algorithm Used:** Gradient Boosting
""")

# Feature Importance Section
st.subheader("📊 Feature Importance in NBA Shot Prediction")
feature_importance_path = "feature_importance.png"
if os.path.exists(feature_importance_path):
    st.image(Image.open(feature_importance_path), caption="Feature Importance", width=700)  
else:
    uploaded_file = st.file_uploader("Upload Feature Importance Image", type=['png'])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Feature Importance", width=700)
    else:
        st.warning("⚠️ Feature importance chart not found. Upload the image manually if needed.")

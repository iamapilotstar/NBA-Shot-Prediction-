import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

# 📌 Set page config
st.set_page_config(
    page_title="NBA Shot Prediction",
    page_icon="🏀",
    layout="wide"
)

# 📌 Function to Load Model & Scaler
@st.cache_data
def load_model_and_scaler():
    try:
        with open('Gradient Boosting.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('StandardScaler.pkl', 'rb') as file:
            scaler = pickle.load(file)

        if not hasattr(model, 'predict'):
            st.error('⚠️ The loaded model is not valid. Ensure the correct model is being loaded.')
            st.stop()
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"⚠️ Error: {e}. Ensure the model and scaler exist in the correct directory.")
        st.stop()

# Load the model and scaler
model, scaler = load_model_and_scaler()

# 📌 Title and description
st.title("🏀 NBA Shot Prediction App")
st.markdown("""
This app predicts the probability of a shot being made in an NBA game based on various shot and contextual factors.
""")

# 📌 Create layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Shot Specifications")
    
    shot_dist = st.slider("Shot Distance (ft)", 0, 40, 15)
    close_def_dist = st.slider("Closest Defender Distance (ft)", 0, 10, 3)
    shot_difficulty = st.slider("Shot Difficulty (1-10)", 1, 10, 5)
    shot_number = st.slider("Shot Number in Game", 1, 20, 5)
    
    age = st.slider("Player Age", 18, 40, 25)
    experience_num = st.selectbox("Years of Experience", list(range(21)))
    player_height = st.slider("Player Height (cm)", 160, 220, 200)
    player_weight = st.slider("Player Weight (kg)", 60, 150, 90)
    bmi = player_weight / ((player_height / 100) ** 2)

    home_team_code = st.radio("Home Team", ["Home", "Away"])
    home_team_code = 1 if home_team_code == "Home" else 0
    
    away_team_code = 0
    
    match_location = st.radio("Match Location", ["Home", "Away"])
    match_location = 1 if match_location == "Home" else 0

    shot_clock_remaining = st.slider("Shot Clock Remaining (sec)", 0, 24, 10)
    touch_time = st.slider("Touch Time (sec)", 0, 10, 2)
    game_minutes = st.slider("Game Minutes", 0, 48, 24)

    if st.button("Predict Shot Outcome"):
        input_data = np.array([[  
            float(shot_dist), float(close_def_dist), float(shot_difficulty), int(shot_number),
            int(age), int(experience_num), float(player_height), float(player_weight), float(bmi),
            int(home_team_code), 0, int(match_location),
            float(shot_clock_remaining), float(touch_time), float(game_minutes)
        ]])

        input_data_scaled = scaler.transform(input_data)

        if input_data_scaled.shape[1] != model.n_features_in_:
            st.error(f"⚠️ Model expects {model.n_features_in_} features, but received {input_data_scaled.shape[1]}.")
        else:
            prediction = model.predict(input_data_scaled)
            probabilities = model.predict_proba(input_data_scaled)[0]

            outcome = "Made" if prediction[0] == 1 else "Missed"
            st.success(f"Predicted Shot Outcome: **{outcome}**")
            st.write(f"### Probability of Making the Shot: {probabilities[1] * 100:.2f}%")
            
            # 📊 Probability Graph
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

# 📌 Add "About" Section in Sidebar
st.sidebar.header("📌 About")
st.sidebar.info("""
This application predicts NBA shot outcomes based on contextual and in game parameters using **Machine Learning**.
This will help teams determine what position such as shot distance, shot difficulty and other contextual factors players need to use to gain a good advantage in game.

                       
### **Model Information**
- **Algorithm:** Gradient Boosting Classifier
- **Trained on:** NBA Shot and information Dataset              
""")

# 📌 Add Model Performance Metrics
st.sidebar.header("📊 Model Performance")
st.sidebar.markdown("""
- **Algorithm Used:** Gradient Boosting
""")

# 📌 Display Feature Importance Chart
st.subheader("📊 Feature Importance in NBA Shot Prediction")
feature_importance_path = "feature_importance.png"
if os.path.exists(feature_importance_path):
    st.image(Image.open(feature_importance_path), caption="Feature Importance", use_container_width=True)
else:
    st.warning("⚠️ Feature importance chart not found. Run the script to generate it.")

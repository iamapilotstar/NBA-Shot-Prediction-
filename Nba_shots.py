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

st.markdown("**⬅️ Use the sidebar on the left to switch between prediction and model insights.**")

st.markdown("**⬅️➡️ swipe or use the arrow keys left/right in the Model Insights section to navigate between tabs.**")

st.markdown("**⬇️ Scroll or swipe down in the Model Insights section below the images and click on the **Click to see detailed analysis of data distributions**.**")



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

# Sidebar view selection
view_option = st.sidebar.radio("Select View", ["🏀 Shot Prediction", "📊 Model Insights"])

if view_option == "🏀 Shot Prediction":
    # Title and app description
    st.title("🏀 NBA Shot Prediction App")
    st.markdown("""
    This app predicts the probability of a shot being made in an NBA game based on various shot and contextual factors.
    """)

    # Layout for input fields
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

        home_team_code = 0 
        away_team_code = 1 - home_team_code 

        match_location = st.radio("Match Location", ["Home", "Away"])
        match_location = 1 if match_location == "Home" else 0

        shot_clock_remaining = st.slider("Shot Clock Remaining (sec)", 0, 24, 10)
        touch_time = st.slider("Touch Time (sec)", 0, 10, 2)
        game_minutes = st.slider("Game Minutes", 0, 48, 24)

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

                if input_data_scaled.shape[1] != model.n_features_in_:
                    st.error(f"⚠️ Model expects {model.n_features_in_} features, but received {input_data_scaled.shape[1]}.")
                else:
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
        if st.button("🔄 Reset Predictions"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    # Sidebar Info
    st.sidebar.header("📌 About")
    st.sidebar.info("""
    This application predicts NBA shot outcomes based on contextual and in-game parameters using **Machine Learning**.
    This will help teams determine what position such as shot distance, shot difficulty, and other contextual factors players need to use to gain a good advantage in a game.
                           
    ### **Model Information**
    - **Algorithm:** Gradient Boosting Classifier
    - **Trained on:** NBA Shot and Information Dataset              
    """)

elif view_option == "📊 Model Insights":
    st.markdown("### 📈 Deep Dive: How the Model Learns from Clinical Data")

    image_paths = {
        "Height vs Shot Distance": "Height vs Shot distance.png",
        "Weight vs Shot Distance": "Weight vs Shot distance.png",
        "Position vs Shot Distance": "Position vs Shot disance.png",
        "Simpson's Paradox - Overall": "Simpsons Paradox for overall.png",
        "Simpson's Paradox - Position C": "Simpsons Paradox for Position C.png",
        "Simpson's Paradox - Position PG": "Simpsons Paradox for Position PG.png",
        "Evolution of Shot Types": "Evolution of Shot types.png",
        "Confusion Matrix": "Confusion matrix.png",
        "Model Comparison - Test Accuracy": "Model Accuracy.png",
        "ROC Curve": "AUC-ROC.png",
        "Feature Importance": "feature_importance.png",
        "Feature Correlation Heatmap": "Correlation Heatmap.png"
    }

    tab_labels = list(image_paths.keys())
    tabs = st.tabs(tab_labels)

    for tab, tab_key in zip(tabs, tab_labels):
        with tab:
            st.subheader(tab_key)
            img_file = image_paths[tab_key]
            if os.path.exists(img_file):
                st.image(Image.open(img_file), width=700)
            else:
                st.error(f"⚠️ Image file not found: {img_file}")

            if tab_key == "Height vs Shot Distance":
                with st.expander("Click to see detailed analysis of data distributions"):
                    st.write("- **Taller players tend to take shots closer to the basket** as they have a natural advantage near the rim.")
                    st.write("- **There is a slight positive correlation** between height and shot distance, indicating that taller players have an **efficiency edge near the basket**.")
            
            elif tab_key == "Weight vs Shot Distance":
                with st.expander("Click to see detailed analysis of data distributions"):
                    st.write("- **Heavier players** generally take shots closer to the basket, likely due to their dominant presence on the court.")
                    st.write("- **Guards and lighter players tend to take shots from farther away from the basket.** Our analysis also reveals that taller players are naturally heavier.")
                    st.write("- **Weight does not strongly influence FG%**, highlighting that **skill and positioning matter more** than physique in basketball.")

            elif tab_key == "Position vs Shot Distance":
                with st.expander("Click to see detailed analysis of data distributions"):
                    st.write("- **Player position significantly influences shot selection** and effectiveness.")
                    st.write("- **Guards typically take longer shots**, including more three-pointers, while centers take the shortest ones by leveraging their **proximity to the basket**.")
                    st.write("- **Centers and power forwards have higher FG%** due to taking more high-percentage shots near the hoop. Guards contribute more to long-range scoring.")

            elif tab_key == "Simpson's Paradox - Overall":
                with st.expander("Click to see detailed analysis of data distributions"):
                    st.write("- **Simpson’s Paradox** highlights how overall trends can be misleading when data is not segmented properly.")
                    st.write("- **At the overall level, heavier and taller players tend to have higher FG%.**")
                    st.write("- However, when analyzed by position, this relationship **does not necessarily hold**. Centers—who are the tallest and heaviest—already take high-percentage shots, so within their position, weight and height **do not strongly impact FG% further**.")
                    st.write("- **Similarly, for point guards, taller players do not always have higher FG%.** This is because PGs take more difficult shots, and their shooting efficiency depends more on shot selection rather than height alone.")
                    st.write("- **This paradox highlights the importance of breaking down data by player roles rather than relying only on overall trends.**")

            elif tab_key == "Simpson's Paradox - Position C":
                with st.expander("Click to see detailed analysis of data distributions"):
                    st.write("- **At the overall level, heavier players appear to have a positive correlation with FG%,** but within their own position, this trend does not necessarily continue.")
                    st.write("- **Centers already take high-percentage shots near the rim,** so among centers alone, weight/height differences **do not have a significant impact on FG%**.")
                    st.write("- **This paradox occurs because positional roles dictate shot selection.** While centers are naturally the heaviest players and have high FG%, their efficiency is **driven more by shot type than just weight alone.**")

            elif tab_key == "Simpson's Paradox - Position PG":
                with st.expander("Click to see detailed analysis of data distributions"):
                    st.write("- **Point guards, at an overall level, appear to have a lower FG% compared to other positions.**")
                    st.write("- However, when analyzed by position, **taller point guards do not always have a significant FG% advantage over shorter ones**.")
                    st.write("- **This is because PGs take more difficult shots—long-range threes, pull-ups, and contested jumpers—which reduces their overall shooting efficiency.**")
                    st.write("- **This highlights why breaking down data by position is essential.** FG% is not just about height/weight, but about **the type of shots a player is taking.**")

            elif tab_key == "Evolution of Shot Types":
                with st.expander("Click to see detailed analysis of data distributions"):
                    st.write("- The **NBA has evolved into a perimeter-oriented game**, with a sharp increase in **three-point attempts over the years**.")
                    st.write("- By 2024, **nearly 40% of all shot attempts are from beyond the arc, compared to nearly 0% in 1980**.")

            elif tab_key == "Confusion Matrix":
                with st.expander("Click to see detailed analysis of data distributions"):
                    st.write("- The confusion matrix evaluates **how well the model classifies made and missed shots**.")
                    st.write("- **Made shots (Actual - True Positive): 4195**")
                    st.write("- **Missed shots (Actual - True Negative): 12532**")
                    st.write("- **Incorrectly Predicted Misses as Make (False Positives - Type 1 Error): 2145**")
                    st.write("- **Incorrectly Predicted Make as Misses (False Negatives - Type 2 Error): 8018**")
                    st.write("- **Inference:** The model is **better at predicting missed shots**. Teams can leverage these insights to optimize both defensive and offensive strategies.")
          
            elif tab_key == "Model Comparison - Test Accuracy":
                with st.expander("Click to see detailed analysis of data distributions"):
                    st.write("- **Gradient Boosting was chosen as the final model (~62.2%)**, while Random Forest had a larger train accuracy and test accuracy gap.")
                    st.write("- **Gradient Boosting performs better due to its strong generalization, lower overfitting, and better handling of complex relationships in NBA shot prediction.**")

            elif tab_key == "ROC Curve":
                with st.expander("Click to see detailed analysis of data distributions"):
                    st.write("- The **ROC curve evaluates trade-offs between precision and recall**.")
                    st.write("- **AUC scores between 0.61 and 0.65, which is reasonable** for NBA analytics and demonstrate a decent level of predictive power in shot success analysis.")

            elif tab_key == "Feature Importance":
                with st.expander("Click to see detailed analysis of data distributions"):
                    st.write("- The most influential features are **shot distance, close defender distance, and shot difficulty**, while player attributes surprisingly played a less significant role.")
                    st.write("- **The model relies heavily on defensive and distance-based metrics,** indicating that external game conditions influence shot success more than individual player attributes.")

            elif tab_key == "Feature Correlation Heatmap":
                with st.expander("Click to see detailed analysis of data distributions"):
                    st.write("- The heatmap highlights correlations between features, revealing potential multicollinearity, where highly correlated variables (e.g., PLAYER_HEIGHT, PLAYER_WEIGHT, and BMI) may introduce redundancy. This can affect linear models by inflating variance and making coefficient estimates unreliable. To mitigate this, feature selection techniques like removing one correlated variable or using transformations (e.g., BMI instead of height and weight separately) can be applied. While multicollinearity impacts interpretability in regression models, tree-based models remain largely unaffected.")

# NBA-Shot-Prediction-
🏀 NBA Shot Prediction App
This NBA Shot Prediction App uses Machine Learning to predict whether a shot will be Made or Missed based on various shot-related and contextual factors.

Live Demo: https://shotspredict.streamlit.app/

📊 Model Used: Gradient Boosting Classifier- 64%

📌 Project Overview
Basketball teams and analysts constantly look for ways to improve shot selection and maximize scoring efficiency. This application helps users analyze key shot factors such as:
•	Shot Distance (How far the shot is)
•	Defender Distance (How close the nearest defender is)
•	Shot Clock Pressure (Time left on the shot clock)
•	Game Context (Time remaining in the game, player experience, etc.)

Using this information, the model predicts the probability of whether the shot will be made or missed.

📊 How It Works

•	User Inputs Shot Details → Distance, defender proximity, shot clock, shot difficulty etc.
•	Data is Scaled & Processed → Features are transformed to match the trained model.
•	Machine Learning Model Predicts the Outcome → Outputs Shot Made / Shot Missed.
•	Displays Probability & Visualization → Shows a probability bar chart for better insight.


Key Findings:
•	 Shot Distance & Defender Proximity are the strongest predictors of shot success.
•	 Game context (shot difficulty, shot clock, touch time) plays a role but is less dominant.
•	 Player attributes (height, weight, experience) surprisingly have minimal impact.
•	 Home-court advantage has negligible influence on shot success.


Features
•	User-friendly interface built using Streamlit.
•	Real-time shot prediction using a trained Gradient Boosting Model.
•	Feature importance visualization to understand which factors impact shot success.
•	Interactive sliders for selecting shot parameters.


Tech Stack
•	Machine Learning: Scikit-Learn, Gradient Boosting Classifier
•	Web App Framework: Streamlit
•	Data Visualization: Plotly, Seaborn, Matplotlib
•	Backend Processing: Pandas, NumPy, Pickle

⚠️ Please refer requirements.txt file.

📢 Future Improvements:
•	Add More richer features (e.g., Player Fatigue, Defensive Intensity)
•	Use Deep Learning models (LSTMs or Neural Networks for advanced modeling)
•	Incorporate more advanced hyperparameter tuning techniques.

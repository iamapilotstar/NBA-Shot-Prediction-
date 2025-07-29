# 🏀 CourtVision: Real-Time NBA Shot Prediction & Analysis Tool

## 💡The Problem:

Basketball teams and analysts need deeper insights into shot selection and player tendencies to enhance strategy and improve scoring efficiency. Traditional stats often miss the context behind shot success.

## 🔧The Solution

I developed a machine learning-powered web application that predicts whether an NBA shot will be successful based on player, shot, and game context. The model uses Gradient Boosting and is deployed via Streamlit, with SHAP-based interpreta

Live Demo: https://nba-shot-prediction-w1ai.onrender.com

Report: https://bit.ly/3IuJY06 -Real-Time-NBA-Shot-Prediction.pdf

## 📌 Key Results
✅ Model Accuracy: 62.4% on real-world NBA data (strong given the randomness of in-game action)

✅ Algorithm: Gradient Boosting (outperformed Logistic Regression, Random Forest)

## 🧪 Features:

• Shot Distance & Angle

• Defender Distance

• Touch Time

• Shot Clock Pressure

• Player Experience & Attributes

📊 Model Used: Gradient Boosting Classifier- Accuracy: 62% (Reflecting the inherent uncertainty of shot-making and game outcomes in basketball).

## 📌 Project Overview
Basketball teams and analysts constantly look for ways to improve shot selection and maximize scoring efficiency. This application helps users analyze key shot factors such as:
•	Shot Distance (How far the shot is)
•	Defender Distance (How close the nearest defender is)
•	Shot Clock Pressure (Time left on the shot clock)
•	Game Context (Time remaining in the game, player experience, etc.)

Using this information, the model predicts the probability of whether the shot will be made or missed.

## ⚙️ How It Works

•	User Inputs Shot Details → Distance, defender proximity, shot clock, shot difficulty etc.
•	Data is Scaled & Processed → Features are transformed to match the trained model.
•	Machine Learning Model Predicts the Outcome → Outputs Shot Made / Shot Missed.
•	Displays Probability & Visualization → Shows a probability bar chart for better insight.


## Key Findings:
•	 Shot Distance & Defender Proximity are the strongest predictors of shot success.
•	 Game context (shot difficulty, shot clock, touch time) plays a role but is less dominant.
•	 Player attributes (height, weight, experience) surprisingly have minimal impact.
•	 Home-court advantage has negligible influence on shot success.




## Tech Stack
•	Machine Learning: Scikit-Learn, Gradient Boosting Classifier
•	Web App Framework: Streamlit
•	Data Visualization: Plotly, Seaborn, Matplotlib
•	Backend Processing: Pandas, NumPy, Pickle

⚠️ Please refer requirements.txt file.

## 📢 Future Improvements:
•	Add More richer features (e.g., Player Fatigue, Defensive Intensity)
•	Use Deep Learning models (LSTMs or Neural Networks for advanced modeling)
•	Incorporate more advanced hyperparameter tuning techniques.


## 📁 Folder Structure

```bash
NBA-Shot-Prediction/
│
├── App and Analysis/
│   ├── Nba_shots.py
│   └── Machine_Learning_for_sports_data_portfolio.ipynb
│
├── Models/
│   ├── Gradient Boosting.pkl
│   └── StandardScaler.pkl
│
├── Images/
│   ├── Evolution of Shot types.png
│   ├── Height vs Shot distance.png
│   └── Simpsons Paradox for overall.png
    └── Model Accuracy.png
│
├── Requirements.txt
└── README.md


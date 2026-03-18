# 🩺 Multi-Disease Predictor — Machine Learning Application

## 📌 Project Overview

**Multi-Disease Predictor** is an end-to-end Machine Learning system that predicts the risk of multiple diseases using clinical health parameters.

Currently supported predictions:

- 🧠 Diabetes Risk Detection  
- ❤️ Cardiovascular Risk Detection  

The project demonstrates a complete ML lifecycle:

- 📊 Data preprocessing & feature engineering  
- 🧠 Model training and evaluation  
- 💾 Model serialization  
- 🖥️ Interactive Streamlit frontend  
- ⚡ Real-time prediction engine  
- 🏗️ Modular and scalable architecture  

This application shows how Machine Learning can assist early diagnosis and preventive healthcare decisions.

---

## 🚀 Live Demo

🔗 Try the deployed application here:  
👉 **https://multi-disease-predictorgit-5gn5jhprkhrekg9eczdn7x.streamlit.app**

---

## 🎯 Problem Statement

Predict disease risk based on patient medical attributes.

### 🧬 Diabetes Prediction

- **1 → Diabetes Detected**
- **0 → No Diabetes**

### ❤️ Cardiovascular Prediction

- **1 → High Cardio Risk**
- **0 → Low Cardio Risk**

---

## 📊 Input Features

### 🧬 Diabetes Model Features

| Feature | Description |
|----------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Skin fold thickness |
| Insulin | Serum insulin level |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic risk score |
| Age | Age in years |

---

### ❤️ Cardiovascular Model Features

| Feature | Description |
|----------|-------------|
| Age | Age in years |
| Sex | Gender |
| CP | Chest pain type |
| Trestbps | Resting blood pressure |
| Chol | Serum cholesterol |
| FBS | Fasting blood sugar |
| RestECG | ECG results |
| Thalach | Maximum heart rate |
| Exang | Exercise-induced angina |
| Oldpeak | ST depression |
| Slope | ST segment slope |
| CA | Number of vessels |
| Thal | Thalassemia type |

---

## 🧠 Machine Learning Approach

### 🧬 Diabetes Model

- Algorithm: Random Forest Classifier  
- Preprocessing:
  - Zero values treated as missing  
  - Imputation  
  - Feature scaling  
- Pipeline-based training  

### ❤️ Cardio Risk Model

- Algorithm: Random Forest Classifier  
- Preprocessing:
  - Feature scaling  
- Pipeline-based training  

---

## ⚙️ System Architecture

User → Streamlit Frontend → Prediction Engine → ML Models → Risk Output

---

## 📂 Project Structure

```
multi-disease-predictor/
│
├── .venv/                 # Virtual environment  
├── data/                  # Datasets  
├── logs/                  # Application logs  
├── models/                # Trained model files (.joblib)  
├── notebooks/             # Exploratory analysis notebooks  
│
├── src/                   # Source code  
│   ├── backend/           # Prediction logic / services  
│   ├── config/            # Configuration settings  
│   ├── frontend/          # Streamlit UI  
│   ├── training/          # Model training scripts  
│   ├── utils/             # Utility functions  
│   └── __init__.py  
│
├── .env                   # Environment variables (local only)  
├── .gitignore  
├── env_template.txt       # Sample environment config  
├── requirements.txt       # Dependencies  
└── README.md  
```

---

## 🧪 Prediction Output

The system returns:

- Predicted class (0 or 1)  
- Probability score  
- Diagnosis message  

### Example — Diabetes

Prediction: 1  
Probability: 0.82  
Diagnosis: Diabetes Risk Detected  

### Example — Cardio Risk

Prediction: 0  
Probability: 0.21  
Diagnosis: Low Cardiovascular Risk  

---

## 🧰 Tech Stack

### 🐍 Machine Learning

- Python  
- Pandas  
- NumPy  
- Scikit-learn  

### 🖥️ Application

- Streamlit  

### ⚙️ Utilities & Tools

- Joblib (model serialization)  
- Logging  
- Modular Python architecture  

---

## ▶️ How to Run Locally

### 1️⃣ Clone the Repository

git clone https://github.com/bala-ml/multi-disease-predictor.git  
cd multi-disease-predictor  

### 2️⃣ Create Virtual Environment

python -m venv .venv  
.venv\Scripts\activate      # Windows  
source .venv/bin/activate   # Mac/Linux  

### 3️⃣ Install Dependencies

pip install -r requirements.txt  

### 4️⃣ Run the Application

streamlit run src/frontend/app.py  

Open in browser:  
http://localhost:8501  

---

## 🌐 Deployment

This application can be deployed as a single service on:

- Streamlit Community Cloud (Recommended)  
- Hugging Face Spaces  
- Render  
- Any Python-compatible cloud platform  

No separate backend deployment is required.

---

## 💡 Future Improvements

- Support for additional diseases  
- Advanced ensemble models  
- Explainable AI (SHAP / LIME)  
- Personalized health recommendations  
- Mobile-friendly interface  
- Integration with wearable health devices  

---

## 👤 Author

**Balaji I**  
🎯 Aspiring Machine Learning Engineer  
📍 India  

---

## ⭐ Acknowledgment

This project is developed for educational and portfolio purposes to demonstrate real-world applications of Machine Learning in healthcare risk prediction.
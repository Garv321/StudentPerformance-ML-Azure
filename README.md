# 🎓 Student Performance ML — Azure Deployment (FastAPI, MLflow, CI/CD)

An end-to-end Machine Learning system for predicting student performance, built with a production-grade pipeline and deployed on Microsoft Azure Web App with CI/CD.

---

## 🚀 Project Overview

- Data preprocessing, feature engineering, and transformation  
- Model training with hyperparameter tuning  
- Experiment tracking using MLflow  
- REST API deployment using FastAPI  
- CI/CD deployment on Azure Web App  

---


---

## 🧠 Key Features

- End-to-end ML pipeline  
- MLflow experiment tracking  
- FastAPI REST API  
- Azure deployment with CI/CD  
- Model drift monitoring  
- Production-ready architecture  

---

## 📂 Project Structure

student-ml-azure/
│── data/                     
│── artifacts/                
│── src/
│   ├── components/
│   ├── pipeline/
│── app.py                    
│── requirements.txt
│── .github/workflows/        
│── mlflow.db / mlruns/       
│── README.md

---

## ⚙️ Tech Stack

Python, Pandas, NumPy, Scikit-learn, MLflow, FastAPI, Azure, GitHub Actions

---

## 🛠️ Setup Instructions

1. Clone Repository  
git clone https://github.com/<your-username>/StudentPerformance-ML-Azure.git  

2. Install Dependencies  
pip install -r requirements.txt  

3. Run Training  
python -m src.pipeline.train_pipeline  

4. Start MLflow  
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db  

5. Run API  
uvicorn app:app --reload  

---

## 🔮 API Usage

POST /predict

Input:
{
  "gender": "female",
  "race_ethnicity": "group B",
  "parental_level_of_education": "bachelor's degree",
  "lunch": "standard",
  "test_preparation_course": "none",
  "reading_score": 72,
  "writing_score": 74
}

Output:
{
  "prediction": 75.3
}

---

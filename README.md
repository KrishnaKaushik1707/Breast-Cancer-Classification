# 🩺 Breast Cancer Prediction System

An AI-powered healthcare application that predicts whether a breast tumor is **Benign** or **Malignant** using Machine Learning techniques. The application is built using **Scikit-learn** and deployed with **Streamlit** through an interactive and modern healthcare dashboard.

---

## 🚀 Live Demo

Add your Streamlit deployment link here:

[Live Application](https://breast-cancer-classification-bpfmvhv4waqtxp2c2knn8d.streamlit.app/)

---

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide. Early diagnosis can significantly improve treatment outcomes.

This project uses machine learning algorithms trained on the **Breast Cancer Wisconsin Dataset** to classify tumors as:

- 🟢 Benign
- 🔴 Malignant

The system provides an intuitive dashboard where users can enter tumor measurements and receive instant predictions.

---

## ✨ Features

- Modern Healthcare Dashboard UI
- Real-Time Tumor Classification
- Interactive Input Forms
- Confidence Score Visualization
- Responsive Design
- Machine Learning-Based Predictions
- Professional Streamlit Interface
- GitHub & Cloud Deployment Ready

---

## 🧠 Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Train-Test Split
4. Model Training
5. Hyperparameter Tuning
6. Model Evaluation
7. Model Serialization using Pickle
8. Streamlit Deployment

---

## 📊 Dataset

**Dataset:** Breast Cancer Wisconsin Dataset

Source:

```python
from sklearn.datasets import load_breast_cancer
```

Dataset Characteristics:

- Total Samples: 569
- Features: 30
- Classes: 2
- Target Labels:
  - 0 → Malignant
  - 1 → Benign

---

## 📈 Input Features

The model uses 30 diagnostic measurements divided into three groups:

### Mean Measurements

- Mean Radius
- Mean Texture
- Mean Perimeter
- Mean Area
- Mean Smoothness
- Mean Compactness
- Mean Concavity
- Mean Concave Points
- Mean Symmetry
- Mean Fractal Dimension

### Error Measurements

- Radius Error
- Texture Error
- Perimeter Error
- Area Error
- Smoothness Error
- Compactness Error
- Concavity Error
- Concave Points Error
- Symmetry Error
- Fractal Dimension Error

### Worst Measurements

- Worst Radius
- Worst Texture
- Worst Perimeter
- Worst Area
- Worst Smoothness
- Worst Compactness
- Worst Concavity
- Worst Concave Points
- Worst Symmetry
- Worst Fractal Dimension

---

## 🛠️ Tech Stack

### Machine Learning

- Python
- NumPy
- Scikit-learn
- Pickle

### Deployment

- Streamlit

### Development Tools

- VS Code
- Git
- GitHub

---

## 📂 Project Structure

```text
Breast-Cancer-Prediction/
│
├── app.py
├── trained_model.sav
├── cancer.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Breast-Cancer-Prediction.git

cd Breast-Cancer-Prediction
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Mac/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Application will start locally at:

```text
http://localhost:8501
```

---

## 📊 Model Performance

| Metric | Value |
|----------|----------|
| Accuracy | Add Your Accuracy |
| Precision | Add Value |
| Recall | Add Value |
| F1 Score | Add Value |

---

## 🔮 Future Improvements

- SHAP Explainability
- Feature Importance Visualization
- Multiple Model Comparison
- User Authentication
- PDF Report Generation
- Cloud Database Integration
- Medical Report Upload Support

---

## 👨‍💻 Author

### Krishna Kaushik

B.Tech CSE (Artificial Intelligence & Machine Learning)

JNTUH College of Engineering Hyderabad

GitHub:
https://github.com/KrishnaKaushik1707

LinkedIn:
https://www.linkedin.com/in/krishna-kaushik-097884333/

---

## ⭐ Support

If you found this project useful:

- Star the repository
- Share feedback
- Connect on LinkedIn

---

## 📜 License

This project is developed for educational and learning purposes.

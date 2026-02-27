# 🛡️ Slingshot Secure – AI-Powered Phishing Email Detection System

## 📌 Project Overview

Slingshot Secure is a machine learning–powered phishing detection system built to identify malicious email content using both linguistic intelligence and behavioral pattern analysis.

The system combines Natural Language Processing (NLP) with behavioral threat indicators to produce a real-time threat score and risk classification. It is designed as a practical, deployable prototype that demonstrates secure AI application in real-world cybersecurity scenarios.

This project was developed as part of a technical submission for AMD Slingshot, focusing on performance efficiency, clarity of architecture, and explainable AI.

---

## 🚀 Key Highlights

- Real-time phishing email classification  
- Hybrid feature engineering (Text + Behavioral signals)  
- Threat score generation (0–100 scale)  
- Risk classification: Safe | Suspicious | High Risk  
- Clean interactive UI built using Streamlit  
- Model persistence using Joblib  
- Performance metrics displayed after training  

---

## 🏗️ System Architecture

The solution follows a structured ML pipeline:

1. Data Preprocessing  
   - Lowercasing  
   - URL removal  
   - Special character filtering  
   - Whitespace normalization  

2. Behavioral Feature Engineering  
   - Urgency keyword frequency  
   - Authority keyword frequency  
   - Exclamation mark count  

3. Text Vectorization  
   - TF-IDF (3000 max features)  

4. Feature Fusion  
   - TF-IDF vectors combined with behavioral indicators  

5. Model Training  
   - Logistic Regression classifier  
   - 80/20 train-test split  

6. Evaluation Metrics  
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  

---

## 🧠 Why Hybrid Features?

Most phishing detectors rely purely on text classification.

This system improves detection by combining:
- Language patterns  
- Psychological triggers (urgency, authority pressure)  
- Behavioral signals  

This approach improves generalization and mimics how humans detect suspicious emails.

---

## 📂 Project Structure

```
slingshot-secure/
│
├── app.py
├── phishing_model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Information

Due to GitHub file size limitations, the dataset is hosted externally.

You can use one of the following publicly available phishing datasets:

Kaggle Dataset:
https://www.kaggle.com/datasets/subhajournal/phishingemails

Alternative Dataset:
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

After downloading, rename the file to:

Phishing_Email.csv

Place it in the root directory before training.

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/slingshot-secure.git
cd slingshot-secure
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

If no model exists, click **"Train Model Now"** inside the application.

---

## 📈 Model Performance

During training, the system automatically displays:
- Accuracy  
- Precision  
- Recall  
- F1 Score  

This ensures transparency and performance validation.

---

## 🔐 Risk Classification Logic

| Threat Score | Risk Level |
|--------------|------------|
| 0 – 30       | Safe       |
| 31 – 70      | Suspicious |
| 71 – 100     | High Risk  |

---

## 🧰 Technologies Used

- Python  
- Streamlit  
- Scikit-learn  
- NumPy  
- Pandas  
- Joblib  

---

## 💡 Design Decisions

- Logistic Regression chosen for interpretability and efficiency  
- TF-IDF limited to 3000 features for performance balance  
- Behavioral features added to improve phishing pattern detection  
- Automatic label detection for flexible dataset compatibility  

---

## 📌 Future Improvements

- Deep learning implementation (LSTM / BERT)  
- Email header analysis  
- URL domain reputation scoring  
- Real-time API deployment  
- Cloud-based scaling  
- GPU acceleration (AMD optimization ready)  

---

## 👨‍💻 Author

Manmohan Bora  |  Shreya Chikane
Data Analyst | ML Enthusiast | AI Security Explorer  

---

## 📜 License

This project is intended for academic and research demonstration purposes.

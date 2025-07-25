# 🚗 Auto Insurance Fraud Detection using Machine Learning

## 📌 Project Overview
This project focuses on detecting fraudulent insurance claims using machine learning techniques. The goal is to build a robust classification model that can help an insurance company flag suspicious claims for further investigation.

We followed a full ML pipeline including data cleaning, feature engineering, model building, evaluation, and final deployment.

---

## 📂 Dataset Description

Three datasets were used:
1. **Train Dataset** – Used to train the models.
2. **Validation Dataset** – Used to validate and tune model performance.
3. **Test Dataset** – Used to evaluate the final selected model.

The target variable is: **`Fraud_Ind`**  
- `Y` → Fraudulent Claim  
- `N` → Legitimate Claim

---

## 🔧 Project Workflow

### 1. 📥 Data Loading
- Loaded training, validation, and test datasets using Pandas.

### 2. 🧹 Data Cleaning
- Handled missing values.
- Removed duplicate rows.
- Treated outliers (if any).

### 3. 🔄 Feature Engineering
Created new features to enhance model performance:
- `Claim_Duration_Days`
- `Claim_Amount_per_Injury`
- `Is_Weekend_Accident`
- `Police_Contacted_Flag`
- `Total_Damage_Score`

Also performed:
- Label encoding of categorical variables.
- Date parsing for duration features.

### 4. 📊 Feature Scaling
- Applied **StandardScaler** to numerical columns.

### 5. 🤖 Model Training
Trained and evaluated 10+ classification models.  
The best performing model was selected based on validation metrics.

✅ **Selected Model**: `RandomForestClassifier`

### 6. 🧪 Model Evaluation
Evaluated the final model on the **test set** using:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- AUC Score

### 7. 📈 KPI Visualization
Plotted:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Classification Report

---

## 🧠 Final Model Details

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

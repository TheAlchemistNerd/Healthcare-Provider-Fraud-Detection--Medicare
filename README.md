# 🏥 Healthcare Provider Fraud Detection Using Machine Learning

## 📘 Project Overview

This project presents the development of a machine learning-based **healthcare provider fraud detection system** using real-world **Medicare claims data**. With rising concerns about fraudulent billing and provider misconduct in healthcare systems, particularly Medicare, this system aims to identify potential fraud patterns by analyzing financial anomalies and healthcare claim behaviors.

By applying supervised learning algorithms such as **Decision Trees**, **Random Forests**, and **XGBoost**, the system highlights suspicious activities like **unusually high reimbursements**, **duplicate claims**, and **abnormally high deductible payments**.

The project encapsulates data preprocessing, feature engineering, model training, evaluation, and interpretation of results in the context of real-world fraud detection.

---

## 🎯 Objectives

* To analyze Medicare inpatient and outpatient claims data for patterns indicative of fraudulent behavior.
* To develop classification models to identify suspicious claims or provider behavior.
* To compare model performance and determine the most effective algorithm for fraud detection.
* To provide insights and interpretability that can aid healthcare fraud investigators and data scientists.

---

## 📦 Dataset

The data used in this project comes from publicly available **Medicare claims data**, including both **inpatient** and **outpatient** files. It contains anonymized information on:

* Provider identifiers
* Beneficiary demographic details
* Diagnosis and procedure codes
* Reimbursements and deductible amounts
* Claim dates and types

**Target variable**: A binary indicator denoting whether a claim or provider was involved in fraudulent behavior.

---

## 🧰 Tools and Technologies

| Tool/Library       | Purpose                            |
| ------------------ | ---------------------------------- |
| Python             | Main programming language          |
| Pandas             | Data loading and manipulation      |
| NumPy              | Numerical computing                |
| Scikit-learn       | Model training and evaluation      |
| XGBoost            | Gradient boosting model            |
| Matplotlib/Seaborn | Data visualization                 |
| Jupyter Notebook   | Interactive analysis and reporting |

---

## 🧹 Data Preprocessing

The raw Medicare data underwent a structured preprocessing pipeline, including:

* **Handling missing values**: Imputed or removed where necessary.
* **Feature selection**: Focused on financial and clinical attributes relevant to fraud.
* **Feature engineering**: Created indicators for duplicate claims, claim frequency per provider, and statistical outliers in reimbursement amounts.
* **Normalization**: Applied standard scaling to numeric features.
* **Encoding**: Transformed categorical variables using label encoding and one-hot encoding.

We also dealt with **class imbalance** using **SMOTE** (Synthetic Minority Over-sampling Technique) and **undersampling**, since fraudulent cases are significantly rarer than non-fraudulent ones.

---

## 🤖 Machine Learning Models

The following supervised classification algorithms were implemented and compared:

### 1. **Decision Trees**

* Simple, interpretable model
* Captured basic fraud patterns and decision rules

### 2. **Random Forest**

* Ensemble model with multiple decision trees
* Reduced overfitting and improved generalization

### 3. **XGBoost**

* Gradient boosting algorithm optimized for speed and accuracy
* Best performance on evaluation metrics

### 4. **Logistic Regression**

* Baseline statistical model
* Useful for understanding linear relationships

### 5. **Support Vector Machine (SVM)**

* Performed poorly due to high dimensionality and class imbalance

---

## 📊 Evaluation Metrics

Due to the class imbalance, standard accuracy was not sufficient. Therefore, we used the following metrics:

* **Precision**: How many predicted frauds were actually fraud
* **Recall**: How many actual frauds were correctly predicted
* **F1-Score**: Harmonic mean of precision and recall
* **AUC-ROC Curve**: Trade-off between true and false positives

📈 **Best Results**:

* **XGBoost** had the highest AUC and F1-score.
* **Decision Trees** were most interpretable and still effective.

---

## 📌 Key Findings

* High **deductible and reimbursement** amounts were strong indicators of potential fraud.
* Duplicate and frequent claims by the same provider flagged suspicious behavior.
* Fraud was more common among certain provider types and geographic regions.
* **XGBoost and Decision Trees** outperformed other models in identifying fraud patterns.

---

## 💡 Project Insights

* Fraud detection benefits from a **hybrid approach**: rule-based + machine learning.
* Interpretability matters: domain experts prefer models that explain *why* a case is flagged.
* Feature engineering was critical in improving model accuracy.
* The same techniques can be extended to other healthcare systems and insurance fraud domains.

---

## ⚠️ Limitations

* **Dataset Bias**: Only Medicare data was used; generalization to private insurers is limited.
* **Assumption of Accuracy**: I assumed the fraud labels in the data are correct and comprehensive.
* **Static Analysis**: Temporal evolution of fraud patterns (e.g., over time) was not analyzed in this phase.

---

## 🧪 Future Work

To improve and extend this work, future research can include:

* Incorporating **time-series features** to detect evolving fraud patterns.
* Leveraging **deep learning** and **autoencoders** for unsupervised anomaly detection.
* Adding **claims from private insurers** or integrating external datasets for broader generalizability.
* Using **LIME** or **SHAP** for explainability and transparency in model decisions.
* Creating a **real-time streaming pipeline** for fraud detection using Apache Kafka or similar tools.

---

## 🚀 How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/TheAlchemistNerd/Healthcare-Provider-Fraud-Detection--Medicare.git
cd Healthcare-Provider-Fraud-Detection--Medicare
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Load the data:

* Place your Medicare claims CSV files in the `/data` directory.
* Update the notebook paths accordingly.

4. Run the Jupyter Notebook:

```bash
jupyter notebook Healthcare-Provider-Fraud-Detection-Medicare.ipynb
```

---

## 📁 Project Structure

```
📦 healthcare-fraud-detection/
├── data/
│   └── medicare_claims.csv
├── notebooks/
│   └── fraud_detection_pipeline.ipynb
├── models/
│   └── saved_model_xgboost.pkl
├── visuals/
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

---

## 🧑‍🔬 Authors and Contributions

* **\Nevil Maloba** – Data preprocessing, model development, evaluation, and documentation.

---

## 📜 License

This project is licensed under the MIT License. You are free to use, modify, and distribute the work for non-commercial and academic purposes.

---

## 🙋 Acknowledgments

* U.S. Centers for Medicare & Medicaid Services (CMS) for data access
* Scikit-learn and XGBoost communities for extensive documentation
* Open-source community and Kaggle healthcare challenges

---

## 📫 Contact

If you have any questions, suggestions, or collaboration ideas, feel free to contact:

📧 Email: [nevillemaloba@yahoo.com](mailto:nevillemaloba@yahoo.com)
🐙 GitHub: [github.com/TheAlchemistNerd](https://github.com/TheAlchemistNerd)
🔗 LinkedIn: [linkedin.com/in/Nevil Maloba](https://linkedin.com/in/nevil-maloba-3a268716b)

# Task 3: Customer Churn Prediction

# Project Overview:
This project develops machine learning models to predict customer churn for subscription-based services. By analyzing historical customer data including usage behavior and demographics, the models identify customers at risk of churning, enabling proactive retention strategies.

# Objective
Predict whether a customer will churn (leave the service) or be retained based on their demographic information, account details, and behavior patterns.

# Dataset Information
- Source: Churn_Modelling.csv
- Total Records: 10,000 customers
- Features: 14 attributes
- Target Variable: Exited (0 = Retained, 1 = Churned)
- Churn Rate: ~20% of customers churned
- No Missing Values: Complete dataset

# Key Features:
- Demographics: Age, Gender, Geography
- Account Information: Credit Score, Balance, Estimated Salary
- Behavior: Tenure, Number of Products, Has Credit Card, Is Active Member

# Technologies Used
- *Python 3.x
- Libraries:
  - pandas & numpy - Data manipulation
  - matplotlib & seaborn - Visualization
  - scikit-learn - Machine learning models and preprocessing
  - joblib - Model persistence

# Machine Learning Models

# 1. Logistic Regression
- Training Accuracy: 80.50%
- Testing Accuracy: 80.50%
- Precision: 0.5642
- RecalL: 0.1966
- F1-Score: 0.2920
- ROC-AUC: 0.7710
# 2. Random Forest Classifier
- Training Accuracy: 100.00%
- Testing Accuracy: 86.45%
- Precision: 0.7779
- Recall: 0.4652
- F1-Score: 0.5826
- ROC-AUC: 0.8469
# 3. Gradient Boosting Classifier(BEST MODEL)
- Training Accuracy: 87.04%
- Testing Accuracy: 86.75%
- Precision: 0.7718
- Recall: 0.4749
- F1-Score: 0.5886
- ROC-AUC: 0.8673

# Key Findings

# Feature Importance (Top 5):
1. Age - Older customers show higher churn rates
2. Number of Products - Customers with 3+ products more likely to churn
3. Is Active Member - Inactive members have significantly higher churn
4. Balance - Account balance correlates with retention
5. Geography - Location influences churn behavior

# Insights:
- Customers aged 40+ have higher churn rates
- Geography significantly impacts churn (Germany shows higher churn than France/Spain)
- Inactive members are 2x more likely to churn
- Customers with only 1 product have better retention
- Gender has minimal impact on churn prediction

# 📁 Project Structure
Task_03_Customer_Churn/
│
├── data/
│   └── Churn_Modelling.csv          # Dataset
│
├── notebooks/
│   └── customer_churn_prediction.ipynb   # Main Jupyter notebook
│
├── models/
│   ├── logistic_regression_model.pkl     # Trained LR model
│   ├── random_forest_model.pkl           # Trained RF model
│   ├── gradient_boosting_model.pkl       # Trained GB model
│   ├── standard_scaler.pkl               # Feature scaler
│   ├── label_encoder_geography.pkl       # Geography encoder
│   └── label_encoder_gender.pkl          # Gender encoder
│
├── reports/
│   ├── target_distribution.png
│   ├── numerical_features_distribution.png
│   ├── categorical_features_vs_churn.png
│   ├── correlation_matrix.png
│   ├── logistic_regression_results.png
│   ├── logistic_regression_feature_importance.png
│   ├── random_forest_results.png
│   ├── random_forest_feature_importance.png
│   ├── gradient_boosting_results.png
│   ├── gradient_boosting_feature_importance.png
│   ├── model_comparison_visualization.png
│   ├── model_comparison.csv
│   ├── model_predictions.csv
│   └── project_summary_report.txt
│
└── README.md                             # Project documentation

# Workflow
# 1. Data Preprocessing
- Removed irrelevant columns (RowNumber, CustomerId, Surname)
- Encoded categorical variables (Geography, Gender)
- Applied Standard Scaling to numerical features
- Split data: 80% training, 20% testing (stratified)
# 2. Exploratory Data Analysis
- Analyzed target variable distribution
- Examined numerical and categorical features
- Created correlation matrix
- Identified key patterns and relationships
# 3. Model Development
- Trained three algorithms: Logistic Regression, Random Forest, Gradient Boosting
- Performed hyperparameter optimization
- Evaluated using multiple metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
# 4. Model Evaluation
- Generated confusion matrices
- Plotted ROC curves
- Compared model performances
- Selected best model based on comprehensive metrics

# Visualizations
The project includes 11 visualizations covering:
- Target variable distribution
- Feature distributions and relationships
- Correlation heatmaps
- Confusion matrices for all models
- ROC curves comparison
- Feature importance charts
- Model performance comparison

# Business Recommendations

1. Focus on High-Risk Segments:
   - Customers aged 40+
   - Inactive members
   - Customers in Germany
2. Retention Strategies:
   - Engage inactive members with personalized campaigns
   - Offer incentives to customers with 3+ products
   - Implement age-specific retention programs
3. Monitoring:
   - Track customer activity regularly
   - Set up alerts for customers predicted to churn
   - A/B test retention strategies

# Model Deployment Potential
The Gradient Boosting model with 86.75% accuracy and 0.8673 ROC-AUC score is production-ready and can be integrated into:
- Customer relationship management (CRM) systems
- Automated alert systems
- Dashboard applications for business intelligence

# How to Use
Training the Models:
```python
# Load the notebook
jupyter notebook customer_churn_prediction.ipynb

# Run all cells sequentially
# Models will be automatically saved to models/ folder
Making Predictions:
import joblib
import pandas as pd

# Load the best model
model = joblib.load('models/gradient_boosting_model.pkl')
scaler = joblib.load('models/standard_scaler.pkl')

# Load encoders
le_geo = joblib.load('models/label_encoder_geography.pkl')
le_gender = joblib.load('models/label_encoder_gender.pkl')

# Prepare new data (ensure same preprocessing)
# ... encode and scale features ...

# Make prediction
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)[:, 1]

print(f"Churn Prediction: {'Churned' if prediction[0] == 1 else 'Retained'}")
print(f"Churn Probability: {probability[0]:.2%}")

# Future Improvements
Implement SMOTE or other techniques to handle class imbalance
Try advanced algorithms (XGBoost, LightGBM, Neural Networks)
Feature engineering to create new predictive features
Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
Deploy model as REST API using Flask/FastAPI
Create real-time dashboard for monitoring predictions

# Author
Jaswini G

Completed: November 2025
 Results Summary:
Successfully developed three machine learning models with the Gradient Boosting Classifier achieving the best performance (86.75% accuracy, 0.8673 ROC-AUC). The model effectively identifies customers at risk of churning and provides actionable insights for business retention strategies.

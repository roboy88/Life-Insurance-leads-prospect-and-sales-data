# Life-Insurance-leads-prospect-and-sales-data
Using data analysis to derive from the data the most promising demi graphic and product to focus on in # Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset (ensure the path to the CSV is correct)
data = pd.read_csv('data/insurance_leads.csv')

# Display first few rows of the data
print("First 5 rows of data:")
print(data.head())

# Data Preprocessing: Handle categorical variables (one-hot encoding)
data = pd.get_dummies(data, drop_first=True)

# Define features and target for lead conversion prediction
X_conversion = data.drop(['lead_conversion', 'insurance_type'], axis=1) # Features
y_conversion = data['lead_conversion'] # Target (1 = converted, 0 = not converted)

# Split data into training and test sets for conversion prediction
X_train_conv, X_test_conv, y_train_conv, y_test_conv = train_test_split(X_conversion, y_conversion, test_size=0.3, random_state=42)

# Feature Scaling for conversion prediction
scaler = StandardScaler()
X_train_conv_scaled = scaler.fit_transform(X_train_conv)
X_test_conv_scaled = scaler.transform(X_test_conv)

# Train a Random Forest Classifier for lead conversion prediction
rf_conversion = RandomForestClassifier(n_estimators=100, random_state=42)
rf_conversion.fit(X_train_conv_scaled, y_train_conv)

# Predict on test data
y_pred_conv = rf_conversion.predict(X_test_conv_scaled)

# Evaluate model for lead conversion prediction
print("Lead Conversion Prediction Accuracy:", accuracy_score(y_test_conv, y_pred_conv))
print("Lead Conversion Confusion Matrix:\n", confusion_matrix(y_test_conv, y_pred_conv))
print("Lead Conversion Classification Report:\n", classification_report(y_test_conv, y_pred_conv))

# Hyperparameter tuning for lead conversion prediction using GridSearchCV
param_grid_conv = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]}
grid_search_conv = GridSearchCV(estimator=rf_conversion, param_grid=param_grid_conv, cv=5, verbose=2, n_jobs=-1)
grid_search_conv.fit(X_train_conv_scaled, y_train_conv)

print("Best Parameters for Lead Conversion:", grid_search_conv.best_params_)
best_rf_conv = grid_search_conv.best_estimator_

# Final Prediction using the best model for conversion
y_pred_best_conv = best_rf_conv.predict(X_test_conv_scaled)

# Final evaluation for conversion prediction
print("Final Lead Conversion Accuracy:", accuracy_score(y_test_conv, y_pred_best_conv))
print("Final Lead Conversion Confusion Matrix:\n", confusion_matrix(y_test_conv, y_pred_best_conv))
print("Final Lead Conversion Classification Report:\n", classification_report(y_test_conv, y_pred_best_conv))

# Feature Importance Visualization for conversion prediction
importances_conv = best_rf_conv.feature_importances_
indices_conv = np.argsort(importances_conv)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances for Lead Conversion Prediction')
plt.bar(range(X_train_conv.shape[1]), importances_conv[indices_conv], align='center')
plt.xticks(range(X_train_conv.shape[1]), X_conversion.columns[indices_conv], rotation=90)
plt.tight_layout()
plt.show()

# Now, define features and target for insurance type prediction
X_insurance = data.drop(['lead_conversion', 'insurance_type'], axis=1) # Features
y_insurance = data['insurance_type'] # Target (insurance type: term, whole life, IUL)

# Split data for insurance type prediction
X_train_ins, X_test_ins, y_train_ins, y_test_ins = train_test_split(X_insurance, y_insurance, test_size=0.3, random_state=42)

# Train Random Forest Classifier for insurance type prediction
rf_insurance_type = RandomForestClassifier(n_estimators=100, random_state=42)
rf_insurance_type.fit(X_train_ins, y_train_ins)

# Predict insurance type on the test set
y_pred_type = rf_insurance_type.predict(X_test_ins)

# Evaluate performance for insurance type prediction
print("Insurance Type Prediction Classification Report:\n", classification_report(y_test_ins, y_pred_type))

# Export predictions to CSV for insurance type prediction
predictions_type = pd.DataFrame({
    'Actual Insurance Type': y_test_ins,
    'Predicted Insurance Type': y_pred_type
})
predictions_type.to_csv('insurance_type_predictions.csv', index=False)
print("Insurance type predictions saved to insurance_type_predictions.csv")

# Feature Importance Visualization for insurance type prediction
importances_type = rf_insurance_type.feature_importances_
indices_type = np.argsort(importances_type)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances for Insurance Type Prediction')
plt.bar(range(X_train_ins.shape[1]), importances_type[indices_type], align='center')
plt.xticks(range(X_train_ins.shape[1]), X_insurance.columns[indices_type], rotation=90)
plt.tight_layout()
plt.show()

# Saving both conversion and insurance type models
import joblib

joblib.dump(best_rf_conv, 'conversion_model.pkl')
joblib.dump(rf_insurance_type, 'insurance_type_model.pkl')

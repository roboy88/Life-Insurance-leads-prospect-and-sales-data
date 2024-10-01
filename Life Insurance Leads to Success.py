# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# Load dataset (ensure the path to the CSV is correct)
# Change the path according to where you've saved your dataset
data = pd.read_csv('data/insurance_leads.csv')

# Display first few rows of the data
print("First 5 rows of data:")
print(data.head())

# Data Preprocessing: Handle categorical variables (one-hot encoding)
data = pd.get_dummies(data, drop_first=True)

# Define features and target (lead conversion)
X = data.drop(['lead_conversion', 'insurance_type'], axis=1) # Features
y = data['lead_conversion'] # Target (1 = converted, 0 = not converted)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling (optional but recommended for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = rf.predict(X_test_scaled)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Hyperparameter tuning using GridSearchCV
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_

# Final Prediction using the best model
y_pred_best = best_rf.predict(X_test_scaled)

# Final evaluation
print("Final Accuracy:", accuracy_score(y_test, y_pred_best))
print("Final Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("Final Classification Report:\n", classification_report(y_test, y_pred_best))

# Feature Importance Visualization
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

# Now let's predict which insurance type (term, whole, IUL) is best for each demographic
# Define features and target (insurance type)
X = data.drop(['lead_conversion', 'insurance_type'], axis=1) # Features
y = data['insurance_type'] # Target (insurance type: term, whole life, IUL)

# Split data for insurance type prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Classifier for insurance type prediction
rf_insurance_type = RandomForestClassifier(n_estimators=100, random_state=42)
rf_insurance_type.fit(X_train, y_train)

# Predict insurance type on the test set
y_pred_type = rf_insurance_type.predict(X_test)

# Evaluate performance for insurance type prediction
print("Classification Report for Insurance Type Prediction:\n", classification_report(y_test, y_pred_type))

# Export predictions to CSV if needed
predictions = pd.DataFrame({
'Actual': y_test,
'Predicted': y_pred_type
})
predictions.to_csv('insurance_lead_predictions.csv', index=False)
print("Predictions saved to insurance_lead_predictions.csv")
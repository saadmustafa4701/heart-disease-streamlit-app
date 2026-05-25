import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, precision_score, recall_score, f1_score
import joblib

# Load the cleaned dataset
df = pd.read_csv('Heart_Disease_Prediction_cleaned.csv')

# Separate features and target
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Perform k-fold cross-validation
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Define scorers
accuracy_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
precision_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='precision')
recall_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='recall')
f1_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='f1')

print(f"K-Fold Cross-Validation Results (k={k}):")
print(f"Accuracy: {accuracy_scores.mean():.2f} ± {accuracy_scores.std():.2f}")
print(f"Precision: {precision_scores.mean():.2f} ± {precision_scores.std():.2f}")
print(f"Recall: {recall_scores.mean():.2f} ± {recall_scores.std():.2f}")
print(f"F1-Score: {f1_scores.mean():.2f} ± {f1_scores.std():.2f}")

# Train on full data
model.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved.")
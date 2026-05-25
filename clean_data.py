import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Heart_Disease_Prediction.csv')

# Display initial info
print("Initial dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())

# Handle missing values: drop rows with any missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Convert categorical target to numeric if needed
if df['Heart Disease'].dtype == 'object':
    le = LabelEncoder()
    df['Heart Disease'] = le.fit_transform(df['Heart Disease'])

# Identify numerical columns for outlier detection
numerical_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']

# Function to remove outliers using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for each numerical column
for col in numerical_cols:
    df = remove_outliers_iqr(df, col)

# Display cleaned dataset info
print("Cleaned dataset shape:", df.shape)
print("Missing values after cleaning:\n", df.isnull().sum())

# Save the cleaned dataset
df.to_csv('Heart_Disease_Prediction_cleaned.csv', index=False)

print("Cleaned dataset saved as 'Heart_Disease_Prediction_cleaned.csv'")
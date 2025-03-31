import pandas as pd

data = pd.read_csv('gym_members_exercise_tracking.csv')

# Display all column names
print("Columns in the dataset:")
print(data.columns)

# Adjust display settings to show more columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)

# Display concise summary of the DataFrame
print("\nDataset Info:")
data.info()

# Check for null values in the dataset
print("\nNull values in each column:")
print(data.isnull().sum())

# Display descriptive statistics of the dataset
print("\nDescriptive Statistics:")
print(data.describe())

# Display the first few rows again with updated settings
print(data.head())
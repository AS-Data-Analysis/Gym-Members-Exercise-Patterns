import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('gym_members_exercise_tracking.csv')

print("Columns in the dataset:")
print(df.columns)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)

print("\nDataset Info:")
df.info()

print("\nNull values in each column:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe())

bins = np.linspace(min(df["Age"]), max(df["Age"]), 9)

age_groups = ["18-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56+"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=age_groups, include_lowest=True)

df.to_csv('final_df.csv', index=False)
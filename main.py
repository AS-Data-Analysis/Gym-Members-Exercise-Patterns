import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Load the dataset
df = pd.read_csv('gym_members_exercise_tracking.csv')

# Set display options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Display dataset information
df.info()

# Select continuous columns and exclude specific ones
continuous_columns = df.select_dtypes(include=[np.number]).columns
continuous_columns = continuous_columns.drop(['Workout_Frequency (days/week)',
                                              'Experience_Level'])

# Normalize continuous columns using MinMaxScaler
scaler = MinMaxScaler()
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

# Set up a 2x7 grid for plotting
fig, axes = plt.subplots(2, 7, figsize=(15, 10))
axes = axes.flatten()

# Remove 'Calories_Burned' from continuous columns
continuous_columns = continuous_columns.drop('Calories_Burned')

# Select categorical columns and append additional ones
categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.append(pd.Index(['Workout_Frequency (days/week)',
                                                  'Experience_Level']))

# Combine continuous and categorical columns for plotting
all_columns = list(continuous_columns) + list(categorical_columns)

correlation_columns_to_check = continuous_columns.append(pd.Index(['Calories_Burned']))
print(df[correlation_columns_to_check].corr()['Calories_Burned'])

# Plot scatter plots for continuous variables and boxplots for categorical variables
for i, col in enumerate(all_columns):
    if col in continuous_columns:
        sns.scatterplot(data=df, x=col, y='Calories_Burned', ax=axes[i])
        sns.regplot(data=df, x=col, y='Calories_Burned', scatter=False, color='red', ax=axes[i])
    else:
        sns.boxplot(data=df, x=col, y='Calories_Burned', ax=axes[i])

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# Calculate Pearson coefficient and p-value for each continuous variable with 'Calories_Burned'
for col in continuous_columns:
    pearson_coef, p_value = stats.pearsonr(df[col], df['Calories_Burned'])
    print(f"{col}:\npearson_coef = {pearson_coef}, p-value = {p_value}\n")

# Save the updated dataframe to a CSV file
df.to_csv('final_df.csv', index=False)
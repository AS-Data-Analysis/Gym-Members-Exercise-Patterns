from requirements import *

# Load the dataset
df = pd.read_csv('gym_members_exercise_tracking.csv')

# Set display options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Display dataset information
df.info()

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Select continuous columns and exclude specific ones
continuous_columns = df.select_dtypes(include=[np.number]).columns
continuous_columns = continuous_columns.drop(['Workout_Frequency (days/week)',
                                              'Experience_Level'])

# Normalize continuous columns using MinMaxScaler on training data
scaler = MinMaxScaler()
train_df[continuous_columns] = scaler.fit_transform(train_df[continuous_columns])

# Apply the same transformation to the testing data
test_df[continuous_columns] = scaler.transform(test_df[continuous_columns])

# Set up a 2x7 grid for plotting
fig, axes = plt.subplots(3, 7, figsize=(15, 10))
axes = axes.flatten()

# Select categorical columns and append additional ones
categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.append(pd.Index(['Workout_Frequency (days/week)',
                                                  'Experience_Level']))


# Plot scatter plots for continuous variables and boxplots for categorical variables
for i, col in enumerate(df.columns):
    if col in continuous_columns and col != 'Calories_Burned':
        sns.scatterplot(data=df, x=col, y='Calories_Burned', ax=axes[i])
        sns.regplot(data=df, x=col, y='Calories_Burned', scatter=False, color='red', ax=axes[i])
    else:
        sns.boxplot(data=df, x=col, y='Calories_Burned', ax=axes[i])

# Adjust layout and save the plots to a file
plt.tight_layout()
plt.show()

# Calculate Pearson coefficient and p-value for each continuous variable with 'Calories_Burned'
for col in continuous_columns:
    pearson_coef, p_value = stats.pearsonr(df[col], df['Calories_Burned'])
    print(f"{col}:\npearson_coef = {pearson_coef}, p-value = {p_value}\n")

# Calculate the full correlation matrix for all continuous features
correlation_matrix = df[continuous_columns].corr()

# Create a heatmap for the full correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, fmt='.2f')
plt.title('Full Correlation Matrix: Continuous Features')
plt.show()

# Perform one-way ANOVA for 'Workout_Frequency (days/week)' categories
categories = df['Workout_Frequency (days/week)'].unique()
grouped_data = [df[df['Workout_Frequency (days/week)'] == category]['Calories_Burned'] for category in categories]

f_stat, p_value = stats.f_oneway(*grouped_data)

print(f"One-way ANOVA results for 'Workout_Frequency (days/week)':")
print(f"F-statistic: {f_stat}")
print(f"P-value: {p_value}")

# Perform one-way ANOVA for 'Experience_Level' categories
categories_exp = df['Experience_Level'].unique()
grouped_data_exp = [df[df['Experience_Level'] == category]['Calories_Burned'] for category in categories_exp]

f_stat_exp, p_value_exp = stats.f_oneway(*grouped_data_exp)

print(f"One-way ANOVA results for 'Experience_Level':")
print(f"F-statistic: {f_stat_exp}")
print(f"P-value: {p_value_exp}")

lm = LinearRegression()

X_train = train_df[['Session_Duration (hours)', 'Fat_Percentage']]
y_train = train_df['Calories_Burned']

lm.fit(X_train, y_train)

# Prepare testing data for prediction
X_test = test_df[['Session_Duration (hours)', 'Fat_Percentage']]
y_test = test_df['Calories_Burned']

# Predict on the testing data
y_pred = lm.predict(X_test)

# Find index of 'Calories_Burned' in continuous columns
target_index = list(continuous_columns).index('Calories_Burned')

# Get the inverse-transformed actual values
y_test_original = scaler.inverse_transform(test_df[continuous_columns])[:, target_index]

# To inverse-transform predicted values, we need them in a DataFrame with same columns
temp_df = test_df.copy()
temp_df['Calories_Burned'] = y_pred  # replace scaled predicted column

# Get the inverse-transformed predicted values
y_pred_original = scaler.inverse_transform(temp_df[continuous_columns])[:, target_index]

# Calculate and print R² score and Mean Squared Error
r2 = r2_score(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
print(f"R² Score: {r2}")
print(f"Mean Absolute Error: {mae}")


# Create a KDE plot for actual vs. predicted values
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test_original, label='Actual', color='blue', fill=True, linewidth=2)
sns.kdeplot(y_pred_original, label='Predicted', color='orange', fill=True, linewidth=2)
plt.title('Distribution of Actual vs Predicted Calories Burned')
plt.xlabel('Calories Burned')
plt.ylabel('Density')
plt.legend()
plt.show()
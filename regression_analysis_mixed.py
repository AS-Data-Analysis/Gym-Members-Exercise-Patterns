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

# Prepare training and testing data
X_train = train_df[['Session_Duration (hours)', 'Fat_Percentage', 'Workout_Frequency (days/week)', 'Experience_Level']]
y_train = train_df['Calories_Burned']
X_test = test_df[['Session_Duration (hours)', 'Fat_Percentage', 'Workout_Frequency (days/week)', 'Experience_Level']]
y_test = test_df['Calories_Burned']

pipeline = Pipeline(steps=[
    ('regressor', RandomForestRegressor(random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Test Set R² Score: {r2}")
print(f"Test Set Mean Absolute Error: {mae}")

# Create a KDE plot for actual vs. predicted values
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, label='Actual', color='blue', fill=True, linewidth=2)
sns.kdeplot(y_pred, label='Predicted', color='orange', fill=True, linewidth=2)
plt.title('Distribution of Actual vs Predicted Calories Burned')
plt.xlabel('Calories Burned')
plt.ylabel('Density')
plt.legend()
plt.savefig('kde_plot_mixed.png')
# plt.show()

# Create a scatterplot of actual vs. predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.title('Actual vs Predicted Calories Burned')
plt.xlabel('Actual Calories Burned')
plt.ylabel('Predicted Calories Burned')
plt.legend()
# plt.show()

# Create a residual plot
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred, y=y_test, color='blue', line_kws={'color': 'red', 'lw': 2})
plt.title('Residual Plot')
plt.xlabel('Predicted Calories Burned')
plt.ylabel('Residuals')
plt.show()
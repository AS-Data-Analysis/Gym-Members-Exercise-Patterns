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
X_train = train_df[['Session_Duration (hours)', 'Fat_Percentage']]
y_train = train_df['Calories_Burned']
X_test = test_df[['Session_Duration (hours)', 'Fat_Percentage']]
y_test = test_df['Calories_Burned']

# Initialize variables to track the best degree, R² score, and MAE
best_degree = None
best_r2 = float('-inf')
best_mae = float('inf')

# Define K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Loop over degrees from 1 to 15
for degree in range(1, 16):
    print(f"Evaluating Degree: {degree}")
    
    # Update the preprocessor with the current degree
    preprocessor = ColumnTransformer(
        transformers=[
            ('poly_scaler', Pipeline(steps=[
                ('scaler', MinMaxScaler()),
                ('poly', PolynomialFeatures(degree=degree))
            ]), ['Session_Duration (hours)', 'Fat_Percentage'])
        ],
        remainder='passthrough'
    )
    
    # Define the pipeline with the updated preprocessor
    poly_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Perform cross-validation
    r2_scores = cross_val_score(poly_pipeline, X_train, y_train, cv=kf, scoring='r2')
    mae_scores = -cross_val_score(poly_pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
    
    # Calculate average R² score and MAE
    avg_r2 = r2_scores.mean()
    avg_mae = mae_scores.mean()
    
    print(f"Degree: {degree}, Average R² Score: {avg_r2}, Average MAE: {avg_mae}")
    
    # Update the best degree if the current R² score is higher and MAE is lower
    if avg_r2 > best_r2 or (avg_r2 == best_r2 and avg_mae < best_mae):
        best_degree = degree
        best_r2 = avg_r2
        best_mae = avg_mae
        best_model = poly_pipeline

# Print the best degree, R² score, and MAE
print(f"Best Degree: {best_degree}")
print(f"Best Average R² Score: {best_r2}")
print(f"Best Average Mean Absolute Error: {best_mae}")

# Fit the best model on the entire training dataset
best_model.fit(X_train, y_train)

# Generate predictions on the test set
y_pred = best_model.predict(X_test)

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
plt.savefig('kde_plot_continuous.png', dpi=300, bbox_inches='tight')
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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

# Print the best degree, R² score, and MAE
print(f"Best Degree: {best_degree}")
print(f"Best Average R² Score: {best_r2}")
print(f"Best Average Mean Absolute Error: {best_mae}")

# Train the final model using the best degree
final_preprocessor = ColumnTransformer(
    transformers=[
        ('poly_scaler', Pipeline(steps=[
            ('scaler', MinMaxScaler()),
            ('poly', PolynomialFeatures(degree=best_degree))
        ]), ['Session_Duration (hours)', 'Fat_Percentage'])
    ],
    remainder='passthrough'
)

final_pipeline = Pipeline(steps=[
    ('preprocessor', final_preprocessor),
    ('regressor', LinearRegression())
])

final_pipeline.fit(X_train, y_train)

# Generate predictions on the test set
y_pred = final_pipeline.predict(X_test)

# Create a KDE plot for actual vs. predicted values
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, label='Actual', color='blue', fill=True, linewidth=2)
sns.kdeplot(y_pred, label='Predicted', color='orange', fill=True, linewidth=2)
plt.title('Distribution of Actual vs Predicted Calories Burned')
plt.xlabel('Calories Burned')
plt.ylabel('Density')
plt.legend()
plt.savefig('pipeline_actual_vs_predicted_kde.png')
plt.show()
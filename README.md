# Gym Members Exercise Patterns Analysis

## Overview
This project analyzes the exercise patterns of gym members to understand the relationships between various factors and the target variable, `Calories_Burned`.

## Steps Performed
1. **Dataset Loading**: The dataset `gym_members_exercise_tracking.csv` was loaded into a pandas DataFrame.
2. **Data Preprocessing**:
   - Continuous variables were normalized using `MinMaxScaler`.
   - Categorical variables were encoded using one-hot encoding.
   - Missing values and irrelevant columns were handled appropriately.
3. **Visualization**:
   - Scatter plots and boxplots were created to visualize relationships between variables.
   - Correlation analysis was performed to identify significant relationships.
4. **Correlation Analysis**:
   - Pearson correlation coefficients and p-values were calculated for continuous variables with the target variable.

## Target and Independent Variables
- **Target Variable**: `Calories_Burned`
- **Independent Variables**:
  - `Session_Duration (hours)`
  - `Fat_Percentage`

## Tools and Libraries Used
- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

## Output
The processed dataset is saved as `final_df.csv`, and visualizations are displayed to analyze the relationships between variables.
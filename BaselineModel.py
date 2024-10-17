import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load datasets
df1 = pd.read_csv('!Assignment 3/dataset1.csv')
df2 = pd.read_csv('!Assignment 3/dataset2.csv')
df3 = pd.read_csv('!Assignment 3/dataset3.csv')

# Merge datasets based on ID
merged_df = df1.merge(df2, on='ID', how='inner').merge(df3, on='ID', how='inner')

# Calculate total weekly screen time for computer, gaming, smartphones, and TV
merged_df['total_computer'] = merged_df['C_we'] * 2 + merged_df['C_wk'] * 5
merged_df['total_gaming'] = merged_df['G_we'] * 2 + merged_df['G_wk'] * 5
merged_df['total_smartphone'] = merged_df['S_we'] * 2 + merged_df['S_wk'] * 5
merged_df['total_TV'] = merged_df['T_we'] * 2 + merged_df['T_wk'] * 5

# Function to remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from each screen time variable
for column in ['total_computer', 'total_gaming', 'total_smartphone', 'total_TV']:
    merged_df = remove_outliers(merged_df, column)

# Prepare baseline models for each device
models = {}

# Define the dependent variable
y = merged_df['Engs']

# Calculate total screen time across all devices
merged_df['total_screentime'] = (merged_df['total_computer'] + 
                                 merged_df['total_gaming'] + 
                                 merged_df['total_smartphone'] + 
                                 merged_df['total_TV'])

# Fit baseline models for each device
for device in ['total_screentime']:
    X = sm.add_constant(merged_df[device])  # Add a constant term for intercept
    model = sm.OLS(y, X).fit()  # Fit the model
    models[device] = model  # Store the model
    
    # Predicting values
    predictions = model.predict(X)

    # Calculate MAE and MSE
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)

    # Print the summary of each model
    print(f"Baseline Model for {device}:")
    print(model.summary())
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print("\n" + "=" * 80 + "\n")

# Calculate the mean of Engs (average engagement score)
mean_engs = merged_df['Engs'].mean()

# Create the scatter plot of total screen time vs average Engs score
plt.figure(figsize=(10, 6))
plt.scatter(merged_df['total_screentime'], merged_df['Engs'], color='powderblue', label='Actual Engs Scores')

# Plot the horizontal line representing the average Engs score
plt.axhline(y=mean_engs, color='red', linestyle='--', label=f'Average Engs Score = {mean_engs:.2f}')

# Set plot labels and title
plt.title('Scatter Plot of Total Screen Time vs Average Engs Score')
plt.xlabel('Total Screen Time (hours)')
plt.ylabel('Engs Score')
plt.legend()

# Show the plot
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

# Merge datasets based on ID
merged_df = df1.merge(df2, on='ID', how='inner').merge(df3, on='ID', how='inner')

# Calculate total weekly screen time for computer, gaming, and smartphones
merged_df['total_computer'] = merged_df['C_we'] * 2 + merged_df['C_wk'] * 5
merged_df['total_gaming'] = merged_df['G_we'] * 2 + merged_df['G_wk'] * 5
merged_df['total_smartphone'] = merged_df['S_we'] * 2 + merged_df['S_wk'] * 5

# Function to remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from each screen time variable
for column in ['total_computer', 'total_gaming', 'total_smartphone']:
    merged_df = remove_outliers(merged_df, column)

# Prepare data for regression
X = merged_df[['total_computer', 'total_gaming', 'total_smartphone']]
y = merged_df['Engs']

# Add constant term to the features
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
mae = np.mean(np.abs(y - y_pred))
mse = np.mean((y - y_pred)**2)
r2 = model.rsquared

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Print model coefficients
print("\nModel Coefficients:")
for name, coef in zip(X.columns, model.params):
    print(f"{name}: {coef:.4f}")
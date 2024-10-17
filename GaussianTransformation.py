import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import PowerTransformer
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

# Prepare data for regression
X = merged_df[['total_computer', 'total_TV', 'total_gaming', 'total_smartphone']]
y = merged_df['Engs']

# Build the linear regression using statsmodels (Before Power Transformation)
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Print model summary before transformation
print("Model Summary Before Power Transformation:\n")
print(model.summary())

# Predictions before transformation
y_pred_before = model.predict(X)

# Calculate MAE and MSE before transformation
mae_before = mean_absolute_error(y, y_pred_before)
mse_before = mean_squared_error(y, y_pred_before)
print(f"\nMAE before transformation: {mae_before:.4f}")
print(f"MSE before transformation: {mse_before:.4f}\n")

# Apply Power Transformer (Yeo-Johnson Transformation)
PowerTransform = PowerTransformer()
X_no_const = X.drop(['const'], axis=1)

x_pow = PowerTransform.fit_transform(X_no_const.values)

df_x_pow = pd.DataFrame(x_pow, index=X_no_const.index, columns=X_no_const.columns)

# Rebuild the linear regression using statsmodels (After Power Transformation)
df_x_pow = sm.add_constant(df_x_pow)
model_transformed = sm.OLS(y, df_x_pow).fit()

# Print model summary after transformation
print("Model Summary After Power Transformation:\n")
print(model_transformed.summary())

# Predictions after transformation
y_pred_after = model_transformed.predict(df_x_pow)

# Calculate MAE and MSE after transformation
mae_after = mean_absolute_error(y, y_pred_after)
mse_after = mean_squared_error(y, y_pred_after)
print(f"\nMAE after transformation: {mae_after:.4f}")
print(f"MSE after transformation: {mse_after:.4f}")

# Plot histograms before Power Transformation
plt.figure(figsize=(12, 8))
for i, column in enumerate(X_no_const.columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(X_no_const[column], kde=True, bins=20)
    plt.title(f'Histogram of {column} (Before Power Transform)')
plt.tight_layout()
plt.show()

# Plot histograms after Power Transformation
plt.figure(figsize=(12, 8))
for i, column in enumerate(df_x_pow.columns[1:], 1):  # skip the 'const' column
    plt.subplot(2, 2, i)
    sns.histplot(df_x_pow[column], kde=True, bins=20)
    plt.title(f'Histogram of {column} (After Power Transform)')
plt.tight_layout()
plt.show()
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
X = merged_df[['total_computer', 'total_gaming', 'total_smartphone', 'total_TV']]
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

# Create two separate figures for scatter plots
variables = ['total_computer', 'total_gaming', 'total_smartphone', 'total_TV']
titles = ['Computer', 'Gaming', 'Smartphone', 'TV']

# First figure: Computer and Gaming
fig1, axs1 = plt.subplots(1, 2, figsize=(16, 8))
# Second figure: Smartphone and TV
fig2, axs2 = plt.subplots(1, 2, figsize=(16, 8))

for i, (var, title) in enumerate(zip(variables, titles)):
    if i < 2:
        ax = axs1[i]
    else:
        ax = axs2[i-2]
    
    # Scatter plot
    ax.scatter(X[var], y, color='blue', alpha=0.5, label='Actual')
    
    # Fit a simple regression for this variable
    X_single = sm.add_constant(X[var])
    model_single = sm.OLS(y, X_single).fit()
    
    # Generate points for the regression line
    x_range = np.linspace(X[var].min(), X[var].max(), 100)
    X_range = sm.add_constant(x_range)
    y_pred = model_single.predict(X_range)
    
    # Plot the regression line
    ax.plot(x_range, y_pred, color='red', label='Regression Line')
    
    ax.set_xlabel(f"{title} Weekly Screen Time (hours)")
    ax.set_ylabel("Engs Score (Wellbeing)")
    ax.set_title(f"Wellbeing Score vs {title} Screen Time")
    ax.legend()

fig1.suptitle("Wellbeing vs Computer and Gaming Screen Time", fontsize=16)
fig2.suptitle("Wellbeing vs Smartphone and TV Screen Time", fontsize=16)

fig1.tight_layout()
fig2.tight_layout()
plt.show()


# Print model coefficients
print("Model Coefficients:")
for name, coef in zip(X.columns, model.params):
    print(f"{name}: {coef:.4f}")

# Select relevant columns for correlation
columns_to_correlate = ['total_computer', 'total_gaming', 'total_smartphone', 'total_TV', 'Engs']
correlation_data = merged_df[columns_to_correlate]

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.4f', cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap: Screen Time vs Wellbeing')
plt.tight_layout()
plt.show()
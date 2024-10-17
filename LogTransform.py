import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

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

# Calculate total screen time across all devices
merged_df['total_screentime'] = (merged_df['total_computer'] + 
                                 merged_df['total_gaming'] + 
                                 merged_df['total_smartphone'] + 
                                 merged_df['total_TV'])

# Remove respondents with 0 screen time for any device
merged_df = merged_df[(merged_df['total_computer'] > 0) & 
                      (merged_df['total_gaming'] > 0) & 
                      (merged_df['total_smartphone'] > 0) & 
                      (merged_df['total_TV'] > 0)]

# Function to remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from total screen time
merged_df = remove_outliers(merged_df, 'total_screentime')

# Log-transform and visualize one variable at a time
def log_transform_and_plot(df, col_to_log, label):
    df_transformed = df.copy()
    
    # Apply log transformation to the selected column
    df_transformed[col_to_log] = np.log(df_transformed[col_to_log])

    # Prepare data for regression
    X = df_transformed[['total_computer', 'total_TV', 'total_gaming', 'total_smartphone']]
    y = df_transformed['Engs']

    # Add constant (intercept) to the model
    X = sm.add_constant(X)

    # Fit the multilinear regression model
    model = sm.OLS(y, X).fit()

    # Print the regression results
    print(f"Regression results with log-transformed {label}:")
    print(model.summary())

    # Visualize the relationship between Engs and total screen time
    plt.figure(figsize=(10, 6))
    sns.regplot(x=df_transformed['total_screentime'], y=df_transformed['Engs'], scatter_kws={'s': 50}, line_kws={'color': 'red'})
    plt.title(f'Relationship Between Total Screen Time and Engs (Log-transformed {label})')
    plt.xlabel(f'Total Screen Time (Log-transformed {label})')
    plt.ylabel('Engagement Score (Engs)')
    plt.show()

# Log-transform total_computer and plot
log_transform_and_plot(merged_df, 'total_computer', 'Computer Screen Time')

# Log-transform total_TV and plot
log_transform_and_plot(merged_df, 'total_TV', 'TV Screen Time')

# Log-transform total_gaming and plot
log_transform_and_plot(merged_df, 'total_gaming', 'Gaming Screen Time')

# Log-transform total_smartphone and plot
log_transform_and_plot(merged_df, 'total_smartphone', 'Smartphone Screen Time')
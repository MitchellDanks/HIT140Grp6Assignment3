import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

# Merge datasets based on ID
merged_df = df1.merge(df2, on='ID', how='inner').merge(df3, on='ID', how='inner')

# Calculate total weekly screen time for each device
merged_df['total_computer'] = merged_df['C_we'] * 2 + merged_df['C_wk'] * 5
merged_df['total_gaming'] = merged_df['G_we'] * 2 + merged_df['G_wk'] * 5
merged_df['total_smartphone'] = merged_df['S_we'] * 2 + merged_df['S_wk'] * 5
merged_df['total_TV'] = merged_df['T_we'] * 2 + merged_df['T_wk'] * 5

# Prepare data for plotting
plot_data = merged_df[['total_computer', 'total_gaming', 'total_smartphone', 'total_TV']]

# Create box plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=plot_data, palette='Set3')

plt.title('Distribution of Weekly Screen Time by Device')
plt.ylabel('Hours per Week')
plt.xlabel('Device')
plt.xticks(range(4), ['Computer', 'Gaming', 'Smartphone', 'TV'], rotation=45)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Print summary statistics
print(plot_data.describe())

# Print the number of records
print(f"\nTotal number of records: {len(merged_df)}")

# Function to identify outliers using IQR method
def identify_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers)

# Print number of outliers for each device
for column in plot_data.columns:
    num_outliers = identify_outliers(plot_data, column)
    print(f"Number of outliers in {column}: {num_outliers}")
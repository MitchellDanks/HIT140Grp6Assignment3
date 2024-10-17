import pandas as pd
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

# Function to calculate average screen time for each device for a given Engs score
def FindAVG_PerDevice(df, Engs_Score):
    # Filter the dataset for the specific Engs score
    filtered_df = df[df['Engs'] == Engs_Score]
    
    # Calculate the mean screen time for each device
    avg_computer = filtered_df['total_computer'].mean()
    avg_gaming = filtered_df['total_gaming'].mean()
    avg_smartphone = filtered_df['total_smartphone'].mean()
    avg_TV = filtered_df['total_TV'].mean()
    
    return avg_computer, avg_gaming, avg_smartphone, avg_TV

# Store average screentime for each Engs score in a dictionary
Length_Engs_Score = range(1, 6)  # Engs score range from 1 to 5
AVG_Screentime_Data = {
    'Engs': [],
    'Computer': [],
    'Gaming': [],
    'Smartphone': [],
    'TV': []
}

# Calculate average screentime for each Engs score
for Engs_Score in Length_Engs_Score:
    avg_computer, avg_gaming, avg_smartphone, avg_TV = FindAVG_PerDevice(merged_df, Engs_Score)
    AVG_Screentime_Data['Engs'].append(Engs_Score)
    AVG_Screentime_Data['Computer'].append(avg_computer)
    AVG_Screentime_Data['Gaming'].append(avg_gaming)
    AVG_Screentime_Data['Smartphone'].append(avg_smartphone)
    AVG_Screentime_Data['TV'].append(avg_TV)

# Convert the dictionary to a DataFrame for easy plotting
AVG_Screentime_df = pd.DataFrame(AVG_Screentime_Data)

# Plotting scatterplots for each device's average screentime against Engs scores
plt.figure(figsize=(10, 8))

# Scatterplot for Computer screentime
plt.subplot(2, 2, 1)
sns.scatterplot(x='Engs', y='Computer', data=AVG_Screentime_df, color='b')
plt.title('Average Computer Screentime by Engs Score')
plt.ylabel('Average Hours')
plt.xlabel('Engs Score')
plt.grid(True)
# Scatterplot for Gaming screentime
plt.subplot(2, 2, 2)
sns.scatterplot(x='Engs', y='Gaming', data=AVG_Screentime_df, color='g')
plt.title('Average Gaming Screentime by Engs Score')
plt.ylabel('Average Hours')
plt.xlabel('Engs Score')
plt.grid(True)

# Scatterplot for Smartphone screentime
plt.subplot(2, 2, 3)
sns.scatterplot(x='Engs', y='Smartphone', data=AVG_Screentime_df, color='r')
plt.title('Average Smartphone Screentime by Engs Score')
plt.ylabel('Average Hours')
plt.xlabel('Engs Score')
plt.grid(True)

# Scatterplot for TV screentime
plt.subplot(2, 2, 4)
sns.scatterplot(x='Engs', y='TV', data=AVG_Screentime_df, color='y')
plt.title('Average TV Screentime by Engs Score')
plt.ylabel('Average Hours')
plt.xlabel('Engs Score')
plt.grid(True)

# Adjust layout for better spacing between plots
plt.tight_layout()

# Show the plots
plt.show()
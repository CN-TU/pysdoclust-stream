import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read the CSV files
csv_file1 = './results/results_real.csv'
csv_file2 = './results/results_ex.csv'

df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

# Concatenate the two DataFrames vertically (along rows)
df = pd.concat([df1, df2], axis=0, ignore_index=True)

# Reset the index
df.reset_index(drop=True, inplace=True)

# Function to extract folder structure information
def extract_folder_info(filepath):
    # Split the file path to get folder names
    parts = filepath.split('/')
    
    if len(parts) > 2:  # Ensure we have enough parts
        data_collection = parts[1]  # 'real'
        type_info = parts[2].split('.')[0]  # 'base' from 'base_clean'
        return pd.Series([data_collection, type_info])
    return pd.Series([None, None])  # Return None if the structure is unexpected

# Apply the function to extract new columns
df[['data_collection', 'type']] = df['filename'].apply(extract_folder_info)

# Exclude rows where the algorithm is 'GT'
df = df[df['algorithm'] != 'GT']

# Map the `noisy` column to 'yes' for normal and 'no' for clean
df['algorithm'] = df['algorithm'].map({'STREAMKMeans': 'stKMns', 'SDOstreamc': 'SDOstcl', 'CluStream': 'CluSt', 'DenStream': 'DenSt', 'DBStream': 'DBst'})

save_dir = 'plots'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Define the indices to plot
indices = ['ARI', 'iXB', 'iPS', 'irCIP', 'TS', 'time']  # Adjust this list according to your available columns

# Create a separate box plot for each index with hue='noisy'
for index in indices:
    # Create a new figure for each index
    plt.figure(figsize=(8, 6))
    
    # Create the box plot for the current index
    sns.barplot(data=df, x='type', y=index, hue='algorithm', palette='tab10', dodge=True, estimator='median', errorbar = 'ci')
    
    plt.xlabel('Dataset')
    plt.ylabel(f'{index}')
    plt.xticks(rotation=30)
    if index == 'iPS':
        plt.yscale('symlog')  # Set y-axis to symlog scale
    if index =='iXB' or index == 'irCIP' or index == 'time':
        plt.yscale('log')  # Set y-axis to log scale
        if not index == 'time':
            plt.gca().invert_yaxis()  # Reverse the y-axis for iXB and irCIP


    # Save the plot as an SVG file with a suitable filename
    file_name = f'barplot_real_ex_{index}.svg'
    plt.savefig(os.path.join(save_dir, file_name), format='svg')
    
    # Close the figure to free up memory
    plt.close()

print(f"Box plots have been saved to the directory: {save_dir}")

# Group by 'type' (dataset), 'index', and 'algorithm', then calculate the median for each index
median_df = df.groupby(['type', 'algorithm'])[indices].median().stack().reset_index()
median_df.columns = ['type', 'algorithm', 'index', 'value']

# Pivot the table to make 'index' and 'algorithm' the row index and 'type' the columns
pivot_median_df = median_df.pivot_table(index=['index', 'algorithm'], columns='type', values='value')


# Convert the DataFrame to LaTeX format
latex_table = pivot_median_df.to_latex(float_format="%.2f", caption="Median Indices per Dataset and Algorithm", label="tab:median_indices", index=True)

# Create a directory to save the LaTeX file if it doesn't exist
save_dir = 'plots'
os.makedirs(save_dir, exist_ok=True)

# Save the LaTeX table to a .tex file
latex_filename = os.path.join(save_dir, 'median_real_ex_table.tex')
with open(latex_filename, 'w') as f:
    f.write(latex_table)

print(f"LaTeX table has been saved to {latex_filename}")
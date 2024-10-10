import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read the CSV file
csv_file = './results/results_syn.csv'  # Replace with your actual file path
df = pd.read_csv(csv_file)

# Function to extract folder structure information
def extract_folder_info(filepath):
    # Split the file path to get folder names
    parts = filepath.split('/')
    
    # Assuming the structure is consistent, we can extract:
    # synthetic from 'data/synthetic/'
    # base from 'base_clean/' (or similar)
    # clean from 'clean/' (or noisy)
    if len(parts) > 2:  # Ensure we have enough parts
        data_collection = parts[1]  # 'synthetic'
        type_info = parts[2].split('_')[0]  # 'base' from 'base_clean'
        noisy_info = parts[2].split('_')[1]  # 'clean' from 'base_clean'
        return pd.Series([data_collection, type_info, noisy_info])
    return pd.Series([None, None, None])  # Return None if the structure is unexpected

# Apply the function to extract new columns
df[['data_collection', 'type', 'noisy']] = df['filename'].apply(extract_folder_info)

# Exclude rows where the algorithm is 'GT'
df = df[df['algorithm'] != 'GT']

# Map the `noisy` column to 'yes' for normal and 'no' for clean
df['noisy'] = df['noisy'].map({'normal': 'yes', 'clean': 'no'})
df['algorithm'] = df['algorithm'].map({'STREAMKMeans': 'stKMns', 'SDOstreamc': 'SDOstcl', 'CluStream': 'CluSt', 'DenStream': 'DenSt', 'DBStream': 'DBst'})

save_dir = 'plots'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Define the indices to plot
indices = ['ARI', 'TS', 'iPS', 'iXB', 'irCIP', 'time']  # Adjust this list according to your available columns

# Create a separate box plot for each index with hue='noisy'
for index in indices:
    # Create a new figure for each index
    plt.figure(figsize=(8, 6))
    
    # Create the box plot for the current index
    sns.boxplot(data=df, x='algorithm', y=index, hue='noisy', palette='tab10', dodge=True)
    
    # Set titles and labels
    # plt.title(f'Box Plot of {index} Scores (Noisy)')
    plt.xlabel('Algorithm')
    plt.ylabel(f'{index}')
    plt.xticks(rotation=30)
    if index == 'iPS':
        plt.yscale('symlog')  # Set y-axis to symlog scale
    if index =='iXB' or index == 'irCIP' or index == 'time':
        plt.yscale('log')  # Set y-axis to log scale
        if not index == 'time':
            plt.gca().invert_yaxis()  # Reverse the y-axis for iXB and irCIP

    # Save the plot as an SVG file with a suitable filename
    file_name = f'boxplot_syn_{index}_noisy.svg'
    plt.savefig(os.path.join(save_dir, file_name), format='svg')
    
    # Close the figure to free up memory
    plt.close()

# Create a separate box plot for each index with hue='noisy'
for index in indices:
    # Create a new figure for each index
    plt.figure(figsize=(8, 6))
    
    # Create the box plot for the current index
    sns.violinplot(data=df, x='algorithm', y=index, hue='noisy', palette='tab10', split=True, inner='quart', dodge=True)
    
    # Set titles and labels
    # plt.title(f'Box Plot of {index} Scores (Noisy)')
    plt.xlabel('Algorithm')
    plt.ylabel(f'{index}')
    plt.xticks(rotation=30)
    if index == 'iPS':
        plt.yscale('symlog')  # Set y-axis to symlog scale
    if index =='iXB' or index == 'irCIP' or index == 'time':
        plt.yscale('log')  # Set y-axis to log scale
        if not index == 'time':
            plt.gca().invert_yaxis()  # Reverse the y-axis for iXB and irCIP

    # Save the plot as an SVG file with a suitable filename
    file_name = f'violinplot_syn_{index}_noisy.svg'
    plt.savefig(os.path.join(save_dir, file_name), format='svg')
    
    # Close the figure to free up memory
    plt.close()

# Filter the DataFrame for noisy = 'no'
df_sequential_no = df[df['type'] != 'sequential']

# Create a separate box plot for each index with hue='noisy'
for index in indices:
    # Create a new figure for each index
    plt.figure(figsize=(8, 6))
    
    # Create the box plot for the current index
    sns.boxplot(data=df_sequential_no, x='algorithm', y=index, hue='noisy', palette='tab10', dodge=True)
    # sns.violinplot(data=df_sequential_no, x='algorithm', y=index, hue='noisy', palette='tab10', split=True, inner='quart', dodge=True)
    
    # Set titles and labels
    # plt.title(f'Box Plot of {index} Scores (Noisy)')
    plt.xlabel('Algorithm')
    plt.ylabel(f'{index}')
    plt.xticks(rotation=30)
    if index == 'iPS':
        plt.yscale('symlog')  # Set y-axis to symlog scale
    if index =='iXB' or index == 'irCIP' or index == 'time':
        plt.yscale('log')  # Set y-axis to log scale
        if not index == 'time':
            plt.gca().invert_yaxis()  # Reverse the y-axis for iXB and irCIP

    # Save the plot as an SVG file with a suitable filename
    file_name = f'boxplot_syn_noseq_{index}_noisy.svg'
    plt.savefig(os.path.join(save_dir, file_name), format='svg')
    
    # Close the figure to free up memory
    plt.close()

# Create a separate box plot for each index with hue='noisy'
for index in indices:
    # Create a new figure for each index
    plt.figure(figsize=(8, 6))
    
    # Create the box plot for the current index
    sns.violinplot(data=df_sequential_no, x='algorithm', y=index, hue='noisy', palette='tab10', split=True, inner='quart', dodge=True)
    
    # Set titles and labels
    # plt.title(f'Box Plot of {index} Scores (Noisy)')
    plt.xlabel('Algorithm')
    plt.ylabel(f'{index}')
    plt.xticks(rotation=30)
    if index == 'iPS':
        plt.yscale('symlog')  # Set y-axis to symlog scale
    if index =='iXB' or index == 'irCIP' or index == 'time':
        plt.yscale('log')  # Set y-axis to log scale
        if not index == 'time':
            plt.gca().invert_yaxis()  # Reverse the y-axis for iXB and irCIP

    # Save the plot as an SVG file with a suitable filename
    file_name = f'violinplot_syn_noseq_{index}_noisy.svg'
    plt.savefig(os.path.join(save_dir, file_name), format='svg')
    
    # Close the figure to free up memory
    plt.close()

# Create a separate box plot for each index with hue='type'
for index in indices:
    # Create a new figure for each index
    plt.figure(figsize=(8, 6))
    
    # Create the box plot for the current index with hue='type'
    sns.boxplot(data=df, x='algorithm', y=index, hue='type', palette='Set2', dodge=True)
    
    # Set titles and labels
    # plt.title(f'Box Plot of {index} Scores (Type)')
    plt.xlabel('Algorithm')
    plt.ylabel(f'{index}')
    plt.xticks(rotation=30)
    if index == 'iPS':
        plt.yscale('symlog')  # Set y-axis to symlog scale
    if index =='iXB' or index == 'irCIP' or index == 'time':
        plt.yscale('log')  # Set y-axis to log scale
        if not index == 'time':
            plt.gca().invert_yaxis()  # Reverse the y-axis for iXB and irCIP
    # Save the plot as an SVG file with a suitable filename
    file_name = f'boxplot_syn_{index}_type.svg'
    plt.savefig(os.path.join(save_dir, file_name), format='svg')
    
    # Close the figure to free up memory
    plt.close()

# Filter the DataFrame for noisy = 'no'
df_noisy_no = df[df['noisy'] == 'no']

# Create a separate box plot for each index with hue='type'
for index in indices:
    # Create a new figure for each index
    plt.figure(figsize=(8, 6))
    
    # Create the box plot for the current index with hue='type'
    sns.boxplot(data=df_noisy_no, x='algorithm', y=index, hue='type', palette='Set2', dodge=True)
    
    # Set titles and labels
    # plt.title(f'Box Plot of {index} Scores (Type)')
    plt.xlabel('Algorithm')
    plt.ylabel(f'{index}')
    plt.xticks(rotation=30)
    if index == 'iPS':
        plt.yscale('symlog')  # Set y-axis to symlog scale
    if index =='iXB' or index == 'irCIP' or index == 'time':
        plt.yscale('log')  # Set y-axis to log scale
        if not index == 'time':
            plt.gca().invert_yaxis()  # Reverse the y-axis for iXB and irCIP

    # Save the plot as an SVG file with a suitable filename
    file_name = f'boxplot_syn_clean_{index}_type.svg'
    plt.savefig(os.path.join(save_dir, file_name), format='svg')
    
    # Close the figure to free up memory
    plt.close()

print(f"Box plots have been saved to the directory: {save_dir}")

# Filter the DataFrame for noisy = 'no'
df_noisy_yes = df[df['noisy'] == 'yes']

# Create a separate box plot for each index with hue='type'
for index in indices:
    # Create a new figure for each index
    plt.figure(figsize=(8, 6))
    
    # Create the box plot for the current index with hue='type'
    sns.boxplot(data=df_noisy_yes, x='algorithm', y=index, hue='type', palette='Set2', dodge=True)
    
    # Set titles and labels
    # plt.title(f'Box Plot of {index} Scores (Type)')
    plt.xlabel('Algorithm')
    plt.ylabel(f'{index}')
    plt.xticks(rotation=30)
    if index == 'iPS':
        plt.yscale('symlog')  # Set y-axis to symlog scale
    if index =='iXB' or index == 'irCIP' or index == 'time':
        plt.yscale('log')  # Set y-axis to log scale
        if not index == 'time':
            plt.gca().invert_yaxis()  # Reverse the y-axis for iXB and irCIP

    # Save the plot as an SVG file with a suitable filename
    file_name = f'boxplot_syn_normal_{index}_type.svg'
    plt.savefig(os.path.join(save_dir, file_name), format='svg')
    
    # Close the figure to free up memory
    plt.close()

print(f"Box plots have been saved to the directory: {save_dir}")

# Define the data types
data_types = ['base', 'moving', 'nonstat', 'sequential']

# Initialize a list to store the table data
table_data = []

# Calculate the median values for each metric, algorithm, and data type
for index in indices:
    for algorithm in df['algorithm'].unique():
        row = [algorithm]  # Start the row with the algorithm name
        
        for data_type in data_types:
            # Filter for the current algorithm and data type
            filtered = df[(df['algorithm'] == algorithm) & (df['type'] == data_type)]
            
            # Calculate the median values for noisy 'no' and 'yes'
            normal_median = filtered[filtered['noisy'] == 'no'][index].median()
            clean_median = filtered[filtered['noisy'] == 'yes'][index].median()
            
            # Format the entry as 'v0 / v1'
            entry = f"{normal_median:.2f} / {clean_median:.2f}" if pd.notna(normal_median) and pd.notna(clean_median) else "N/A"
            row.append(entry)
        
        table_data.append(row)

# Create a new column that combines 'type' and 'noisy'
df['type_noisy'] = df['type'] + " (" + df['noisy'] + ")"  # e.g., "base (yes)"

# Create a new column that combines 'type' and 'noisy'
df['type_noisy'] = df['type'] + " (" + df['noisy'] + ")"  # e.g., "base (yes)"

# Group by 'type_noisy' and 'algorithm' to calculate the median for each index
median_df = df.groupby(['type_noisy', 'algorithm'])[indices].median().stack().reset_index()
median_df.columns = ['type_noisy', 'algorithm', 'index', 'value']

# Pivot the table to make 'index' and 'algorithm' the row index and 'type_noisy' the columns
pivot_median_df = median_df.pivot_table(index=['index', 'algorithm'], columns='type_noisy', values='value')

# Convert the DataFrame to LaTeX format
latex_table = pivot_median_df.to_latex(float_format="%.2f", caption="Median Indices per Dataset and Algorithm", label="tab:median_indices", index=True)

# Create a directory to save the LaTeX file if it doesn't exist
save_dir = 'plots'
os.makedirs(save_dir, exist_ok=True)

# Save the LaTeX table to a .tex file
latex_filename = os.path.join(save_dir, 'median_syn_table_easy.tex')
with open(latex_filename, 'w') as f:
    f.write(latex_table)

print(f"LaTeX table has been saved to {latex_filename}")

# Reset index to facilitate aggregation
pivot_median_df.reset_index(inplace=True)

# Initialize a DataFrame to store the aggregated results
aggregated_data = {}

# Aggregate values by combining 'yes' and 'no' for each type
for index in indices:
    for algorithm in pivot_median_df['algorithm']:
        # Get values for the corresponding index and algorithm
        base_yes = pivot_median_df.loc[(pivot_median_df['index'] == index) & (pivot_median_df['algorithm'] == algorithm), 'base (yes)'].values
        base_no = pivot_median_df.loc[(pivot_median_df['index'] == index) & (pivot_median_df['algorithm'] == algorithm), 'base (no)'].values
        nonstat_yes = pivot_median_df.loc[(pivot_median_df['index'] == index) & (pivot_median_df['algorithm'] == algorithm), 'nonstat (yes)'].values
        nonstat_no = pivot_median_df.loc[(pivot_median_df['index'] == index) & (pivot_median_df['algorithm'] == algorithm), 'nonstat (no)'].values
        moving_yes = pivot_median_df.loc[(pivot_median_df['index'] == index) & (pivot_median_df['algorithm'] == algorithm), 'moving (yes)'].values
        moving_no = pivot_median_df.loc[(pivot_median_df['index'] == index) & (pivot_median_df['algorithm'] == algorithm), 'moving (no)'].values
        sequential_yes = pivot_median_df.loc[(pivot_median_df['index'] == index) & (pivot_median_df['algorithm'] == algorithm), 'sequential (yes)'].values
        sequential_no = pivot_median_df.loc[(pivot_median_df['index'] == index) & (pivot_median_df['algorithm'] == algorithm), 'sequential (no)'].values

        # Create combined strings for each type with values rounded to two decimals
        aggregated_data[(index, algorithm)] = {
            'base': f"{round(base_yes[0], 2)} / {round(base_no[0], 2)}" if base_yes.size > 0 and base_no.size > 0 else "N/A",
            'nonstat': f"{round(nonstat_yes[0], 2)} / {round(nonstat_no[0], 2)}" if nonstat_yes.size > 0 and nonstat_no.size > 0 else "N/A",
            'moving': f"{round(moving_yes[0], 2)} / {round(moving_no[0], 2)}" if moving_yes.size > 0 and moving_no.size > 0 else "N/A",
            'sequential': f"{round(sequential_yes[0], 2)} / {round(sequential_no[0], 2)}" if sequential_yes.size > 0 and sequential_no.size > 0 else "N/A"
        }

# Create a new DataFrame for the aggregated results
aggregated_df = pd.DataFrame(aggregated_data).T
aggregated_df.index.names = ['index', 'algorithm']

# Convert the DataFrame to LaTeX format
latex_table = aggregated_df.to_latex(float_format="%.2f", caption="Aggregated Median Indices per Dataset and Algorithm", label="tab:aggregated_median_indices", index=True)

# Create a directory to save the LaTeX file if it doesn't exist
save_dir = 'plots'
os.makedirs(save_dir, exist_ok=True)

# Save the LaTeX table to a .tex file
latex_filename = os.path.join(save_dir, 'median_syn_table.tex')
with open(latex_filename, 'w') as f:
    f.write(latex_table)

print(f"LaTeX table has been saved to {latex_filename}")
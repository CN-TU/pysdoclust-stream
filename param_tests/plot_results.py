import json
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

def compute_param_importance(df, par_cols, dataset_name=None, file_path=None, suffix=None):
    """
    Compute and aggregate parameter importance using Lasso regression.

    Parameters:
    - df: DataFrame containing parameters and target values.
    - par_cols: List of parameter columns to be used for Lasso regression.
    - dataset_name: Optional name of the dataset for saving results.
    - file_path: Optional path to save the parameter importance results.

    Returns:
    - lasso_importance: Series with Lasso regression importance values.
    - combined_importance: DataFrame with aggregated parameter importance, where indices are renamed according to specific rules.
    - correlation_matrix: DataFrame of the correlation matrix for parameters and values.
    """


    # Prepare X and Y for Lasso regression
    X = df[[col for col in df.columns if col in par_cols]]  # All parameter columns
    Y = df['values']  # Target values

    # Normalize the X variables
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Perform Lasso regression with normalized features
    lasso = LassoCV()
    lasso.fit(X_normalized, Y)

    # Get the importance of each parameter
    lasso_importance = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
    lasso_importance = lasso_importance / lasso_importance.max()  # Normalize to [0, 1]

    # Calculate the correlation matrix for the specified columns
    corr_cols = par_cols + ['values']
    correlation_matrix = df[corr_cols].corr()  # Use absolute values

    # Extract the correlations of each parameter with 'values'
    corr_importance = correlation_matrix['values'].abs().drop('values').sort_values(ascending=False)
    corr_importance = corr_importance / corr_importance.max()  # Normalize to [0, 1]

    # Create a combined DataFrame with Lasso and Correlation importance
    combined_importance = pd.DataFrame({
        'lasso_importance': lasso_importance,
        'correlation_importance': corr_importance
    })
    # print(combined_importance)

    # Replace NaN values with 0 for any parameter missing from one metric
    combined_importance = combined_importance.fillna(0)

    def get_parent_directory(dir):
        """
        Extract the parent directory from the given filepath by removing the last subfolder.

        Parameters:
        - filepath: The full path from which to extract the parent directory.

        Returns:
        - Parent directory path.
        """
        # Ensure the path does not have a trailing slash, so dirname works correctly
        dir = dir.rstrip(os.sep)

        # Get the parent directory
        parent_directory = os.path.dirname(dir)
    
        return parent_directory + os.sep  # Append separator for consistency

    # Save Lasso importance values to a JSON file if `file_path` and `dataset_name` are provided
    if file_path and dataset_name:
        # Define the JSON file path
        parent_path = get_parent_directory(file_path)
        if suffix:
            json_file_path = f"{parent_path}/param_importance_" + suffix + ".json"
        else:
            json_file_path = f"{parent_path}/param_importance.json"

        # Load existing data if the file exists, otherwise create an empty dictionary
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as json_file:
                importance_dict = json.load(json_file)
        else:
            importance_dict = {}

        # Normalize the lasso_importance by the sum of all values
        lasso_importance_normalized = lasso_importance / lasso_importance.sum()

        # Save the normalized importance to a dictionary
        importance_dict[dataset_name] = lasso_importance_normalized.to_dict()
        
        # Save the updated dictionary back to the file
        with open(json_file_path, "w") as json_file:
            json.dump(importance_dict, json_file, indent=4)

        # Print confirmation message
        print(f"Lasso importance values have been saved in the file: {json_file_path}")

    return lasso_importance, combined_importance, correlation_matrix

def plot_param_importance_all(file_path, par_cols, dataset_to_color=None, json_file='param_importance.json', output_file='param_importance.svg'):
    """
    Plots parameter importance from a JSON file and saves the figure as an SVG.

    Args:
    file_path (str): Path to the directory where output files should be saved.
    json_file (str): Name of the JSON file containing parameter importance data.
    output_file (str): Name of the output file for saving the plot as SVG.
    """
    
    def get_parent_directory(dir):
        """
        Extract the parent directory from the given filepath by removing the last subfolder.

        Parameters:
        - dir: The full path from which to extract the parent directory.

        Returns:
        - Parent directory path.
        """
        # Ensure the path does not have a trailing slash, so dirname works correctly
        dir = dir.rstrip(os.sep)

        # Get the parent directory
        parent_directory = os.path.dirname(dir)

        return parent_directory + os.sep  # Append separator for consistency

    # Get the parent directory for saving output files
    parent_dir = get_parent_directory(file_path)

    # Construct the full paths for the JSON input file and SVG output file
    json_path = os.path.join(parent_dir, json_file)
    output_path = os.path.join(parent_dir, output_file)

    # Step 1: Load the JSON data from the file
    with open(json_path) as f:
        data = json.load(f)

    # Convert the JSON data to a DataFrame
    df = pd.DataFrame(data).T  # Transpose to have datasets as rows
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Dataset'}, inplace=True)

    # Reverse the par_cols list
    custom_order_param = par_cols[::-1]  # This reverses the list    
    custom_order_dataset = ['cong', 'retail', 'fert', 'flow', 'occupancy']

    # Melt the DataFrame for seaborn
    df_melted = df.melt(id_vars='Dataset', var_name='Parameter', value_name='Value')

    # Map categories in 'Parameter' to numerical values based on custom_order_param
    df_melted['category_order_param'] = df_melted['Parameter'].map(dict(zip(custom_order_param, range(len(custom_order_param)))))

    # Map categories in 'Dataset' to numerical values based on custom_order_dataset
    df_melted['category_order_dataset'] = df_melted['Dataset'].map(dict(zip(custom_order_dataset, range(len(custom_order_dataset)))))

    # Sort by both 'category_order_param' and 'category_order_dataset'
    df_melted.sort_values(['category_order_param', 'category_order_dataset'], inplace=True)

    # Drop the temporary sorting columns
    df_melted.drop(['category_order_param', 'category_order_dataset'], axis=1, inplace=True)

    # Step 2: Create the Bar Plot
    plt.figure(figsize=(10, 6))
    if dataset_to_color:
        sns.barplot(data=df_melted, y='Parameter', x='Value', hue='Dataset', orient='h', palette=dataset_to_color)
    else:
        sns.barplot(data=df_melted, y='Parameter', x='Value', hue='Dataset', orient='h')
    
    # Use the parameter names from the melted DataFrame to create the new tick labels
    unique_parameters = df_melted['Parameter'].unique()
    new_tick_labels = [
        label.replace('log_', '').replace('sq_', '').replace('outlier_threshold', 'outl_thr').replace('replacement_rate', 'rpl_rate')
        for label in unique_parameters
    ]

    # Set the new y-axis tick labels
    plt.yticks(ticks=np.arange(len(unique_parameters)), labels=new_tick_labels, ha='right')

    # Step 3: Customize the Plot
    plt.xlabel('importance')
    plt.ylabel('')
    # plt.yticks(rotation=30)
    # plt.xticks(rotation=30)
    plt.legend(title='dataset')
    plt.tight_layout()

    # Step 4: Save the Plot as SVG in the parent directory
    plt.savefig(output_path, format='svg')
    plt.close()  # Close the figure after saving to avoid display issues
    print(f"Parameter importance plot saved to: {output_path}")

def plot_histogram(df, file_path, dataset_name, bins=30, color=None):
    """
    Plots a histogram of the specified column in the DataFrame and saves it as an SVG file.

    Parameters:
    - df: pd.DataFrame: The DataFrame containing the data.
    - column: str: The column name to plot.
    - file_path: str: The directory path to save the histogram.
    - bins: int: The number of bins for the histogram (default is 30).
    """
    # Set the style for the plot
    sns.set_theme(style="whitegrid")

    # Create the histogram
    plt.figure(figsize=(10, 6))  # Optional: set the figure size
    if color:
        sns.histplot(df['values'], bins=bins, stat='probability', kde=True, color=color) 
    else:
        sns.histplot(df['values'], bins=bins, stat='probability', kde=True)  # Adjust the number of bins as needed

    # Add labels and title
    plt.xlabel('ari')
    plt.ylabel('relative frequency')
    # plt.title(f'Histogram of {column}')

    # Save the figure as an SVG file for the histogram
    svg_file_path = f"{file_path}/{dataset_name}_histogram_ARI.svg"  # Ensure file_path is defined
    plt.savefig(svg_file_path, format='svg')
    print(f"Histogram plot saved to: {svg_file_path}")

def plot_parameter_importance(importance, file_path, dataset_name, suffix=None, color=None):
    """
    Plots parameter importance and saves it as an SVG file.

    Parameters:
    - importance_vector: Series containing the importance values.
    - file_path: Path to save the SVG file.
    """
    # Plot theparameter importance
    plt.figure(figsize=(10, 6))
    if color:
        importance.plot(kind='barh', figsize=(12, 8), color=color)
    else:
        importance.plot(kind='barh', figsize=(12, 8))
    plt.xlabel('Coefficient Magnitude')

    # Create new tick labels
    new_tick_labels = [
        label.replace('log_', '').replace('sq_', '').replace('outlier_threshold', 'outl_thr')
        .replace('chi_prop', 'chi_prop').replace('values', 'ari').replace('replacement_rate', 'rpl_rate') for label in importance.index
    ]

    # Rotate tick labels for better readability    
    plt.yticks(ticks=np.arange(len(new_tick_labels)), labels=new_tick_labels, rotation=30, ha='right')

    # Save the figure as an SVG file
    if suffix==None:
        svg_file_path = file_path + f"/{dataset_name}_importance.svg"
    else:
        svg_file_path = file_path + f"/{dataset_name}_importance_" + suffix + ".svg"
    plt.savefig(svg_file_path, format='svg')
    plt.close()  # Close the plot to free up memory
    print(f"Aggregated parameter importance plot saved to: {svg_file_path}")

def plot_parameter_correlations(correlation_matrix, ordered_params, file_path, dataset_name, suffix=None, cmap=None):
    """
    Plots a heatmap of parameter correlations with specified order and saves it as an SVG file.

    Parameters:
    - correlation_matrix: The correlation matrix to visualize.
    - lasso_importance: Series containing the Lasso importance index.
    - threshold: The threshold value to include in the title.
    - file_path: Path to save the SVG file.
    """    
    
    # Reindex the correlation matrix
    ordered_correlation_matrix = correlation_matrix.loc[ordered_params, ordered_params]

    # Create an absolute correlation matrix for coloring
    abs_correlation_matrix = np.abs(ordered_correlation_matrix)

    # Mask the upper diagonal
    mask = np.triu(np.ones_like(abs_correlation_matrix, dtype=bool), k=0)

    # Visualize the correlations with a heatmap
    plt.figure(figsize=(10, 6))
    if cmap:
        hm = sns.heatmap(abs_correlation_matrix, annot=ordered_correlation_matrix, cmap=cmap, vmin=0, vmax=1, fmt=".2f", mask=mask, cbar=False)
    else:
        hm = sns.heatmap(abs_correlation_matrix, annot=ordered_correlation_matrix, cmap='Blues', vmin=0, vmax=1, fmt=".2f", mask=mask, cbar=False)

    # Add a color bar label
    cbar = plt.colorbar(hm.collections[0])
    cbar.set_label('Absolute Correlation Coefficient', rotation=270, labelpad=15)

    # Strip 'log_' and 'sq_' prefixes from tick labels
    new_tick_labels = [
        label.replace('log_', '').replace('sq_', '').replace('outlier_threshold', 'outl_thr')
        .replace('chi_prop', 'chi_prop').replace('values', 'ari').replace('replacement_rate', 'rpl_rate') for label in ordered_correlation_matrix.columns
    ]

    # Rotate tick labels for better readability
    plt.xticks(ticks=np.arange(len(new_tick_labels)) + 0.5, labels=new_tick_labels, rotation=30, ha='center')
    plt.yticks(ticks=np.arange(len(new_tick_labels)) + 0.5, labels=new_tick_labels, rotation=30, ha='right')

    # Save the figure as an SVG file for parameter correlations
    if suffix==None:
        svg_file_path = file_path + f"/{dataset_name}_correlations.svg"
    else:
        svg_file_path = file_path + f"/{dataset_name}_correlations_" + suffix + ".svg"
    plt.savefig(svg_file_path, format='svg')
    plt.close()  # Close the plot to free up memory
    print(f"Parameter correlations plot saved to: {svg_file_path}")

def create_parallel_coordinates_plot(df, params, file_path, dataset_name, cmap='Viridis', nticks=5, jitter_strength=0.1, suffix=None):
    """
    Create a parallel coordinates plot.

    Parameters:
    - df: The DataFrame containing the data to plot.
    - params: List of parameters for the plot.
    - color_values: The values to color the lines in the plot.
    - file_path: Path to save the SVG file.
    """

    log_par_cols = ['log_T', 'log_chi_prop', 'log_x', 'log_k', 'log_x', 'log_replacement_rate']
    sq_par_cols = ['sq_outlier_threshold']
    cat_par_cols  =['log_k', 'log_x', 'sq_outlier_threshold']

    # Define a function to add jitter to values based on the scale of the column
    def add_jitter_to_values(values, strength=jitter_strength):
        # Calculate jitter strength as a small percentage of the range of values
        jitter_strength = strength * (values.max() - values.min())
        # Add random noise to the values
        return values + np.random.uniform(-jitter_strength, jitter_strength, size=len(values))
    
    # Conditional squaring of scaled values: square the normalized values (values/cmax)
    # color_values = np.where(df['values'] > 0, np.square(np.square(df['values'] / np.max(df['values']))), 0)
    color_values = np.where(df['values'] > 0, np.square(df['values'] / np.max(df['values'])), 0)

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=color_values,
                colorscale=cmap,  # You can change the colorscale if desired
                #showscale=True
            ),
            dimensions=[
                dict(
                    label=param.replace('log_', '').replace('sq_', '') 
                          if param in log_par_cols or param in sq_par_cols 
                          else param,
                    values=add_jitter_to_values(df[param]) 
                          if param in cat_par_cols 
                          else df[param],  # Apply jitter to parameter values
                    tickvals=np.linspace(
                        df[param].min(),
                        df[param].max(),
                        nticks
                    ),
                    ticktext=[
                        f"{2**val:.2f}" if param in log_par_cols 
                        else f"{val**0.5:.2f}" if param in sq_par_cols 
                        else f"{val:.2f}" 
                        for val in np.linspace(
                            df[param].min(),
                            df[param].max(),
                            nticks
                        )
                    ]
                ) for param in params
            ]
        )
    )

    # Adjust the layout for a more compact visualization
    if cmap == 'viridis':
        fig.update_layout(
            width=600,  # Adjust the width of the entire plot
            height=400,  # Adjust the height of the entire plot
            font=dict(color='#D3D3D3'),  # Set the color of all tick text here '#D3D3D3' '#A9A9A9' gray, '#D75B0D' orange
        )
    else:
        fig.update_layout(
            width=600,  # Adjust the width of the entire plot
            height=400,  # Adjust the height of the entire plot
        )

    # Save the figure as an SVG file for parallel coordinates
    if suffix:
        svg_file_path = f"{file_path}/{dataset_name}_parcords_{params[1]}_" + suffix +".svg"
    else:
        svg_file_path = f"{file_path}/{dataset_name}_parcords_{params[1]}.svg"
    fig.write_image(svg_file_path, format='svg')
    print(f"Parallel coordinates plot saved to: {svg_file_path}")

def create_parallel_coordinates_for_top_correlations(df, correlation_matrix, par_cols, file_path, dataset_name, dataset_to_cmap=None, suffix=None):
    """
    Create parallel coordinates plots for the top two correlated parameters with each parameter.

    Parameters:
    - df: The input DataFrame containing the data.
    - correlation_matrix: The DataFrame representing the correlation matrix.
    - par_cols: A list of parameter columns to consider for correlations.
    - file_path: The path where the plots will be saved.
    """

    if dataset_to_cmap:
        cmap = dataset_to_cmap[dataset_name]
    else:
        cmap = 'Viridis'

    for param in par_cols:
        # Get the absolute correlations of the current parameter with all others
        abs_correlations = correlation_matrix[param].abs()
        abs_correlations = abs_correlations[abs_correlations.index != param]
        abs_correlations = abs_correlations[abs_correlations.index != 'values']
        
        # Get the top two correlated parameters
        top_two = abs_correlations.nlargest(2).index.tolist()
        params = [top_two[0], param, top_two[1]]
        
        # Call the function to create the parallel coordinates plot
        create_parallel_coordinates_plot(df, params, file_path, dataset_name, cmap=cmap, suffix=suffix)

def save_best_param_settings(df, dataset_name, file_path):
    """
    Save the best parameter settings based on replacement rate thresholds and k values.

    Parameters:
    - df: The input DataFrame containing the data.
    - thresholds: A list of threshold values for filtering.
    - ks: A list of k values to iterate over.
    - orig_par_cols: A list of original parameter columns to include in the output.
    - file_path: The path where the CSV files will be saved.
    """
    # Define thresholds and k values to evaluate
    thresholds = [1.0, 0.75, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]

    # Specify the original parameter columns to include in the output
    orig_par_cols = ['T', 'x', 'outlier_threshold', 'chi_prop', 'qv', 'zeta'] # without k

    if dataset_name == 'retail':
        ks = [25, 35, 50, 70, 100, 140, 200]
    elif dataset_name == 'fert':
        ks = [25, 35, 50, 70, 100, 140, 200]
    elif dataset_name == 'cong':
        ks = [35, 50, 70, 100, 140, 200, 280, 400, 560, 800]
    elif dataset_name == 'occupancy':
        ks = [35, 50, 70, 100, 140, 200, 280, 400, 560, 800]
    elif dataset_name == 'flow':
        ks = [35, 50, 70, 100, 140, 200, 280, 400, 560, 800]
    else:
        raise ValueError(f"Invalid dataset_name: '{dataset_name}'. Please choose a valid dataset name.")

    # Create an empty list to hold the results
    results = []

    # Iterate over each threshold and k value
    for threshold in thresholds:
        for k in ks:
            # Filter the dataframe based on both replacement_rate threshold and specific k value
            filtered_df = df[(df['replacement_rate'] <= threshold) & (df['k'] == k)]
            
            # Check if the filtered dataframe is not empty
            if not filtered_df.empty:
                # Find the row corresponding to the maximum value in the 'values' column
                max_row = filtered_df.loc[filtered_df['values'].idxmax()]

                # Create a dictionary for each row with the required columns and parameters
                max_row_dict = {
                    'Threshold': threshold,
                    'k': k,
                    'Best Value': f"{max_row['values']:.3f}",
                    'Replacement Rate': max_row['replacement_rate'],
                    'Entry': f"{max_row['values']:.3f} / {max_row['replacement_rate']:.2f}"
                }

                # Add all the original parameter columns to the dictionary
                max_row_dict.update({param: max_row[param] for param in orig_par_cols if param in max_row})

                # Append the row to results
                results.append(max_row_dict)

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)
    
    # Reorder the columns to the specified format
    columns_order = ['Threshold', 'k', 'Best Value', 'Replacement Rate', 'Entry'] + orig_par_cols
    results_df = results_df[columns_order]

    # Remove duplicates based on all columns except 'Threshold', keeping the one with the lower threshold
    results_df = results_df.sort_values(by='Threshold', ascending=False).drop_duplicates(subset=results_df.columns.difference(['Threshold']), keep='last')

    # Pivot the table to create the desired matrix
    matrix_df = results_df.pivot(index='k', columns='Threshold', values='Entry')

    # Save the results to a CSV file
    matrix_file_path = file_path + f"/{dataset_name}_matrix.csv"
    matrix_df.to_csv(matrix_file_path, index=True)
    print(f"Matrix saved to: {matrix_file_path}")

    # Pivot the table to create the desired matrix
    matrix_df = results_df.pivot(index='k', columns='Threshold', values='Best Value')
    matrix_df = matrix_df.fillna('')

    # Add dataset_name as part of the index
    matrix_df['dataset_name'] = dataset_name
    matrix_df.set_index('dataset_name', append=True, inplace=True)

    # Reorder the index to have dataset_name first
    matrix_df = matrix_df.reorder_levels(['dataset_name', 'k'])

    # Save the LaTeX table
    latex_file_path = file_path + f"/{dataset_name}_table.tex"
    with open(latex_file_path, 'w') as f:
        f.write(matrix_df.to_latex(index=True, escape=False))
    print(f"LaTeX table saved to: {latex_file_path}")
    
    # Reorder the columns to the specified format
    columns_order = ['Threshold', 'k', 'Best Value', 'Replacement Rate'] + orig_par_cols
    results_df = results_df[columns_order]

    # Sort by 'Best Value' descending and 'Replacement Rate' ascending
    results_df = results_df.sort_values(by=['Best Value', 'Replacement Rate'], ascending=[False, True])

    # Save the results to a CSV file
    table_file_path = file_path + f"/{dataset_name}_table.csv"
    results_df.to_csv(table_file_path, index=False)
    print(f"Table saved to: {table_file_path}")    

def preprocess_data(file_path, filter_per=1):
    """
    Preprocess the data from the specified JSON file, applying necessary transformations and filtering.

    Parameters:
    - file_path: The path to the JSON file containing trial data.
    - filter_per: The percentage of top values to keep in the DataFrame (default is 1 for 100%).

    Returns:
    - df: A DataFrame containing the preprocessed data.
    """
    # Load data from JSON file
    with open(file_path + '/result.json', 'r') as file:
        data = json.load(file)

    dataset_name = file_path.split('/')[-2]
    
    # Extract trials and their corresponding values
    trials0 = data.get("trials", [])    
    values = data.get("values", [])
    
    best_params = data.get("best_params", {})
    best_params["replacement_rate"] = min(1, best_params.get("k", 1) / best_params.get("T", 1))

    trials1 = [
        {**trial, "replacement_rate": min(1, trial.get("k", 1) / trial.get("T", 1))}
        for trial in trials0
    ]
    
    trials = {
        "params": trials1,
        "values": values
    }
    
    # Create a new dictionary to hold the structured data
    trials_for_df = {key: [trial[key] for trial in trials1] for key in best_params.keys()}
    trials_for_df['values'] = trials['values']
    
    # Create a DataFrame
    df = pd.DataFrame(trials_for_df)

    # Define parameter columns for transformations
    par_cols = ['sq_outlier_threshold', 
                'zeta', 
                'qv', 
                'log_x',
                'log_T', 
                'log_k',
                'log_chi_prop']

    # Apply log transformation to selected variables
    df['log_T'] = np.log2(df['T'])  # Log base 2 transformation
    df['log_k'] = np.log2(df['k'])
    df['log_x'] = np.log2(df['x'])
    df['log_chi_prop'] = np.log2(df['chi_prop'])
    df['log_replacement_rate'] = np.log2(df['replacement_rate'])
    df['sq_outlier_threshold'] = np.square(df['outlier_threshold'])

    # # Create the histogram
    # plt.figure(figsize=(10, 6))  # Optional: set the figure size
    # sns.histplot(df['replacement_rate'], bins=25, stat='probability', kde=True)  # Adjust the number of bins as needed
    # plt.show()

    # Calculate the threshold value for the top 'filter_per' percent of 'values'
    threshold_value = np.percentile(df['values'], 100 * (1 - filter_per))

    # Filter the DataFrame to include only rows with 'values' greater than or equal to the threshold
    df = df[df['values'] >= threshold_value]

    f = 0.7
    dataset_to_color = {
        'cong': plt.cm.Blues(f),      # Drifting Conglomerates, blue
        'retail': plt.cm.Greens(f),    # Retail, orange
        'fert': plt.cm.Greys(f),       # Fertility vs Income, green
        'flow': plt.cm.Purples(f),       # Network Traffic Flows, red
        'occupancy': plt.cm.Reds(f),   # Occupancy, purple
    }
    
    # Define the color mapping for the datasets
    dataset_to_cmap = {
        'cong': 'Blues',      # Drifting Conglomerates
        'retail': 'Greens',  # Retail
        'fert': 'Greys',     # Fertility vs Income
        'flow': 'Purples',       # Network Traffic Flows
        'occupancy': 'Reds'  # Occupancy
    }


    return df, par_cols, dataset_name, dataset_to_color, dataset_to_cmap

def plot_experiment_results(file_path):
    """
    Reads the JSON file containing experiment results and returns a dictionary with the parsed data.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: Parsed dictionary with fixed parameters, best parameters, best value, and trial information.
    """

    df, par_cols, dataset_name, dataset_to_color, dataset_to_cmap = preprocess_data(file_path, filter_per=1)
    df = df[df['replacement_rate'] < 0.5]

    # Calculate the threshold value for the top 'filter_per' percent of 'values'
    filter_per = 1
    threshold_value = np.percentile(df['values'], 100 * (1 - filter_per))

    # Filter the DataFrame to include only rows with 'values' greater than or equal to the threshold
    df = df[df['values'] >= threshold_value]

    color = dataset_to_color[dataset_name] # None
    cmap = dataset_to_cmap[dataset_name] # None

    # Get the length of the DataFrame
    length_of_df = df.shape[0]
    print(f"Length of df: {length_of_df}")

    save_best_param_settings(df, dataset_name, file_path)

    # Plot histogram of scores
    plot_histogram(df, file_path, dataset_name, color=color)

    # Lasso param importance 
    lasso_importance, _, _ = compute_param_importance(df, par_cols, dataset_name=dataset_name, file_path=file_path)
    plot_parameter_importance(lasso_importance, file_path, dataset_name, color=color) 
    plot_param_importance_all(file_path, par_cols, dataset_to_color=dataset_to_color)

    # Filter the DataFrame for good scores
    max_score = df['values'].max()
    threshold = 0.75 * max_score    
    good_scores_df = df[df['values'] >= threshold]

    # Lasso param importance top
    lasso_importance, _, correlation_matrix = compute_param_importance(good_scores_df, par_cols, dataset_name=dataset_name, file_path=file_path, suffix="top")
    plot_parameter_importance(lasso_importance, file_path, dataset_name, suffix="top", color=color)        
    plot_param_importance_all(file_path, par_cols, dataset_to_color=dataset_to_color, json_file='param_importance_top.json', output_file='param_importance_top.svg')
   
    # Correlation Matrix plot
    # ordered_params = [param for param in reversed(lasso_importance.index)] # ['values'] + [param for param in reversed(lasso_importance.index)]
    plot_parameter_correlations(correlation_matrix, par_cols, file_path, dataset_name, cmap=cmap)

    # Paralle coordinates plots
    create_parallel_coordinates_for_top_correlations(df, correlation_matrix, par_cols, file_path, dataset_name, dataset_to_cmap=dataset_to_cmap)

    # Replace 'T' with 'replacement_rate' in par_cols
    # par_cols = [param if param != 'log_T' else 'log_replacement_rate' for param in par_cols]
    par_cols = ['replacement_rate'] + par_cols[:4] + par_cols[5:]
    # par_cols = ['replacement_rate'] + par_cols

    suffix = 'rr'

    # Lasso param importance rr
    lasso_importance, _, _ = compute_param_importance(df, par_cols, dataset_name=dataset_name, file_path=file_path, suffix=suffix)
    plot_parameter_importance(lasso_importance, file_path, dataset_name, suffix=suffix, color=color) 
    plot_param_importance_all(file_path, par_cols, dataset_to_color=dataset_to_color, json_file='param_importance_rr.json', output_file='param_importance_rr.svg')

    # Lasso param importance top rr
    # par_cols = par_cols[:5] + ['log_T'] + par_cols[5:]
    lasso_importance, _, correlation_matrix = compute_param_importance(good_scores_df, par_cols, dataset_name=dataset_name, file_path=file_path, suffix=suffix + "_top")
    plot_parameter_importance(lasso_importance, file_path, dataset_name, suffix=suffix + "_top", color=color)     
    plot_param_importance_all(file_path, par_cols, dataset_to_color=dataset_to_color, json_file='param_importance_rr_top.json', output_file='param_importance_rr_top.svg')
   
    # Correlation Matrix plot
    # ordered_params = [param for param in reversed(lasso_importance.index)] # ['values'] + [param for param in reversed(lasso_importance.index)]
    plot_parameter_correlations(correlation_matrix, par_cols, file_path, dataset_name, suffix=suffix, cmap=cmap)   
    
    # Paralle coordinates plots
    create_parallel_coordinates_for_top_correlations(df, correlation_matrix, par_cols, file_path, dataset_name, dataset_to_cmap=dataset_to_cmap, suffix=suffix)

def main():
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python read_experiment_results.py <path_to_result.json>")
        sys.exit(1)
    
    # Read the file path from the command-line arguments
    file_path = sys.argv[1]
    
    # Read the experiment results
    plot_experiment_results(file_path)
    
    # # Print the results
    # print("Fixed Parameters:", experiment_results["fixed_params"])
    # print("Best Parameters:", experiment_results["best_params"])
    # print("Best Value:", experiment_results["best_value"])

if __name__ == "__main__":
    main()

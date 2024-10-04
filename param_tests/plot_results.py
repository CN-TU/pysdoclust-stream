import json
import sys
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

def plot_heat(trials, param_x, param_y, n_bins=5, n_ticks=5, filepath=None):
    """Generate a heatmap of parameter combinations and their scores."""
    
    # Extract parameter values and scores from trials
    values_x = []
    values_y = []
    scores = []

    log_params = ['T', 'chi_prop', 'outlier_threshold']
    
    for trial in trials["params"]:
        values_x.append(trial[param_x])
        values_y.append(trial[param_y])

    for value in trials["values"]:
        scores.append(value)

    is_log_x = param_x in log_params
    is_log_y = param_y in log_params

    # Determine the type of bins (linear or logarithmic) based on the parameter names
    if is_log_x:
        xedges = np.logspace(np.log10(min(values_x)), np.log10(max(values_x)), n_bins + 1)
    else:
        xedges = np.linspace(min(values_x), max(values_x), n_bins + 1)

    if is_log_y:
        yedges = np.logspace(np.log10(min(values_y)), np.log10(max(values_y)), n_bins + 1)
    else:
        yedges = np.linspace(min(values_y), max(values_y), n_bins + 1)

    # Initialize 2D lists to store all the scores in each bin
    bin_scores = [[[] for _ in range(n_bins)] for _ in range(n_bins)]

    # Place each trial score into its respective bin
    for i in range(len(values_x)):
        x_idx = np.digitize(values_x[i], xedges) - 1
        y_idx = np.digitize(values_y[i], yedges) - 1
        if x_idx < n_bins and y_idx < n_bins:  # Ensure index is within bounds
            bin_scores[x_idx][y_idx].append(scores[i])
    
    # Calculate the max for each bin
    heatmap_data = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            if bin_scores[i][j]:  # Only calculate median if there are scores               
                heatmap_data[i, j] = np.max(bin_scores[i][j])
            else:
                heatmap_data[i, j] = np.nan  # Assign NaN if no scores

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(heatmap_data.T, cmap="Blues", annot=True, vmin=0, vmax=1)

    # Create tick positions and labels (use logspace if applicable)
    x_ticks = np.linspace(0, n_bins, n_ticks)
    y_ticks = np.linspace(0, n_bins, n_ticks)
    if is_log_x:
        x_labels = [f'{np.round(val, 3)}' for val in np.logspace(np.log10(min(values_x)), np.log10(max(values_x)), n_ticks)]        
    else:        
        x_labels = [f'{np.round(val, 3)}' for val in np.linspace(min(values_x), max(values_x), n_ticks)]
    
    if is_log_y:
        y_labels = [f'{np.round(val, 3)}' for val in np.logspace(np.log10(min(values_y)), np.log10(max(values_y)), n_ticks)]        
    else:        
        y_labels = [f'{np.round(val, 3)}' for val in np.linspace(min(values_y), max(values_y), n_ticks)]

    # Set the tick positions and labels
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Set the tick labels to show bin boundaries
    ax.set_xticklabels(x_labels, ha='center', rotation=0)  # Rotate x-axis labels by 30 degrees
    ax.set_yticklabels(y_labels, ha='center', rotation=0)  # Rotate y-axis labels by 90 degrees

     # Adjust padding for tick labels (move them farther from the plot)
    ax.tick_params(axis='x', pad=10)  # Move x-axis labels away from the plot by 10 points
    ax.tick_params(axis='y', pad=20)  # Move y-axis labels away from the plot by 10 points

    ax.xaxis.tick_bottom()  # Ensure x-axis ticks are at the bottom
    ax.yaxis.tick_left()    # Ensure y-axis ticks are on the left
    
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    # plt.title(f'Heatmap of {param_x} vs {param_y} ({metric})')

    if filepath:
        plt.savefig(f"{filepath}/heat_{param_x}_{param_y}.svg", format='svg')
        print(f"Heatmap saved to {filepath}/heat_{param_x}_{param_y}.svg")
        plt.close()
    else:
        plt.show()

def plot_experiment_results(file_path):
    """
    Reads the JSON file containing experiment results and returns a dictionary with the parsed data.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: Parsed dictionary with fixed parameters, best parameters, best value, and trial information.
    """
    with open(file_path + '/result.json', 'r') as file:
        data = json.load(file)    

    dataset_name = file_path.split('/')[-1]
    
    # Extract trials and their corresponding values
    trials0 = data.get("trials", [])    
    values = data.get("values", [])
    
    best_params = data.get("best_params", [])
    best_params["replacement_rate"] = min(1, best_params.get("k", 1) / best_params.get("T", 1))

    trials1 = [
        {**trial, "replacement_rate": min(1, trial.get("k", 1) / trial.get("T", 1))}
        for trial in trials0
    ]
    
    trials = {
        "params": trials1,
        "values": values
    }
    
    # Generate all combinations of parameters
    params = list(best_params.keys())
    param_combinations = list(itertools.combinations(params, 2))
   
    # Iterate over all combinations and create plots
    for pair in param_combinations:    
        plot_heat(trials, pair[0], pair[1], n_bins=6, n_ticks=5, filepath=file_path)
   
    # Create a new dictionary to hold the structured data
    trials_for_df = {key: [trial[key] for trial in trials1] for key in best_params.keys()}
    trials_for_df['values'] = trials['values']
    
    # Create a DataFrame
    df = pd.DataFrame(trials_for_df)

    orig_par_cols = ['k', 
                'T', 
                'x', 
                'outlier_threshold', 
                'chi_prop',
                'qv', 
                'zeta']
    
    # Define thresholds and k values to evaluate
    thresholds = [1.0, 0.75, 0.5, 0.33, 0.25, 0.1]
    ks = [25, 35, 50, 70, 100, 140, 200]

    # Specify the original parameter columns to include in the output
    orig_par_cols = ['k', 'T', 'x', 'outlier_threshold', 'chi_prop', 'qv', 'zeta']

    # Create an empty list to hold the results
    results = []

    # Iterate over each threshold and k value
    for threshold in thresholds:
        for k in ks:
            # Filter the dataframe based on both replacement_rate threshold and specific k value
            filtered_df = df[(df['replacement_rate'] < threshold) & (df['k'] == k)]
            
            # Check if the filtered dataframe is not empty
            if not filtered_df.empty:
                # Find the row corresponding to the maximum value in the 'values' column
                max_row = filtered_df.loc[filtered_df['values'].idxmax()]

                # Create a dictionary for each row with the required columns and parameters
                max_row_dict = {
                    'Threshold': threshold,
                    'k': k,
                    'Best Value': max_row['values'],
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
    columns_order = ['Threshold', 'k', 'Best Value', 'Replacement Rate', 'Entry'] + orig_par_cols[1:]
    results_df = results_df[columns_order]

    # Remove duplicates based on all columns except 'Threshold', keeping the one with the lower threshold
    results_df = results_df.sort_values(by='Threshold', ascending=False).drop_duplicates(subset=results_df.columns.difference(['Threshold']), keep='last')

    # Pivot the table to create the desired matrix
    matrix_df = results_df.pivot(index='k', columns='Threshold', values='Entry')

    # Save the results to a CSV file
    matrix_file_path = file_path + "best_param_settings_matrix.csv"
    matrix_df.to_csv(matrix_file_path, index=True)
    print(f"Matrix saved to: {matrix_file_path}")
    
    # Reorder the columns to the specified format
    columns_order = ['Threshold', 'k', 'Best Value', 'Replacement Rate'] + orig_par_cols
    results_df = results_df[columns_order]

    # Sort by 'Best Value' descending and 'Replacement Rate' ascending
    results_df = results_df.sort_values(by=['Best Value', 'Replacement Rate'], ascending=[False, True])

    # Save the results to a CSV file
    table_file_path = file_path + "best_param_settings_table.csv"
    results_df.to_csv(table_file_path, index=False)
    print(f"Table saved to: {table_file_path}")

    # Apply log transformation to selected variables
    df['log_T'] = np.log2(df['T'])  # Natural logarithm (log base e)
    df['log_k'] = np.log2(df['k'])
    df['log_x'] = np.log2(df['x'])
    df['log_chi_prop'] = np.log2(df['chi_prop'])
    # df['log_outlier_threshold'] = np.log2(df['outlier_threshold'])
    df['sq_outlier_threshold'] = np.square(df['outlier_threshold'])

    # Set the filter percentage
    filter_per = 0.8  # for example, top 10%

    # Calculate the threshold value for the top 'filter_per' percent of 'values'
    threshold_value = np.percentile(df['values'], 100 * (1 - filter_per))

    # Filter the DataFrame to include only rows with 'values' greater than or equal to the threshold
    df = df[df['values'] >= threshold_value]
    
    log_par_cols = ['log_T', 'log_chi_prop', 'log_x', 'log_k', 'log_x']
    par_cols = ['log_k', 
                'log_T', 
                'log_chi_prop', 
                'sq_outlier_threshold', 
                'qv', 
                'zeta', 
                'log_x']
    sq_par_cols = ['sq_outlier_threshold']
    cat_par_cols  =['log_k', 'log_x', 'sq_outlier_threshold']

    # Prepare X and Y for Lasso regression
    X = df[[col for col in df.columns if col in par_cols]]  # All parameter columns
    Y = df['values']  # Target values
    
    # Normalize the X variables
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Perform Lasso regression with normalized features
    lasso = LassoCV()  # Set alpha to a preferred value or tune it as needed
    lasso.fit(X_normalized, Y)
    
    # Get the importance of each parameter
    param_importance = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)

    # Aggregate parameter importance by grouping 'log_param' with 'param' and 'sq_param' with 'param'
    aggregated_importance = param_importance.copy()

    # Create a mapping for aggregation
    aggregation_mapping = {}
    for param in param_importance.index:
        # Map 'log_param' and 'sq_param' to their base 'param'
        base_param = param.replace('log_', '').replace('sq_', '')
        if base_param not in aggregation_mapping:
            aggregation_mapping[base_param] = []
        aggregation_mapping[base_param].append(param)

    # Aggregate the importance scores
    aggregated_importance = pd.Series({
        base_param: param_importance[params].sum() for base_param, params in aggregation_mapping.items()
    })

    # Plot aggregated parameter importance
    plt.figure(figsize=(10, 6))
    aggregated_importance.sort_values(ascending=False).plot(kind='barh', color='skyblue')
    plt.xlabel('Aggregate Coefficient Magnitude')

    # Strip 'log_' and 'sq_' prefixes from tick labels
    new_tick_labels = [label.replace('log_', '').replace('sq_', '').replace('outlier_threshold', 'outl_thresh').replace('chi_prop', 'chi_prop').replace('values', 'score') for label in aggregated_importance.index]

    # Rotate tick labels for better readability    
    plt.yticks(ticks=np.arange(len(new_tick_labels)), labels=new_tick_labels, rotation=30, ha='right')

    # Save the figure as an SVG file for aggregated parameter importance
    svg_file_path_aggregated = file_path + "/parameter_importance_aggregated.svg"
    plt.savefig(svg_file_path_aggregated, format='svg')
    print(f"Aggregated parameter importance plot saved to: {svg_file_path_aggregated}")

    # Plot parameter importance
    plt.figure(figsize=(10, 6))
    param_importance.plot(kind='barh', color='skyblue')
    # plt.title('Parameter Importance using Lasso Regression')
    plt.xlabel('Coefficient Magnitude')

    # Strip 'log_' and 'sq_' prefixes from tick labels
    new_tick_labels = [label.replace('outlier_threshold', 'outl_thresh').replace('chi_prop', 'chi_prop').replace('values', 'score') for label in param_importance.index]

    # Rotate tick labels for better readability    
    plt.yticks(ticks=np.arange(len(new_tick_labels)), labels=new_tick_labels, rotation=30, ha='right')

    # Save the figure as an SVG file for parameter importance
    svg_file_path = file_path + "/parameter_importance.svg"
    plt.savefig(svg_file_path, format='svg')
    print(f"Parameter importance plot saved to: {svg_file_path}")
    
    # Calculate the maximum score
    max_score = df['values'].max()

    # Define a threshold for good scores (90% of the best score)
    threshold = 0.9 * max_score

    # Filter the DataFrame for good scores
    good_scores_df = df[df['values'] >= threshold]

    corr_cols = ['log_k', 
                'log_T', 
                'log_x', 
                'log_chi_prop', 
                'sq_outlier_threshold', 
                'qv', 
                'zeta',
                'values']
    
    # Calculate the correlation matrix for the specified columns
    correlation_matrix = good_scores_df[corr_cols].corr().abs()  # Use absolute values

    # Reverse the order based on the importance list and add 'values' at the beginning
    ordered_params = ['values'] + [param for param in reversed(param_importance.index)]

    # Reindex the correlation matrix
    ordered_correlation_matrix = correlation_matrix.loc[ordered_params, ordered_params]

    # Mask the uppper diagonal
    mask = np.triu(np.ones_like(ordered_correlation_matrix, dtype=bool), k=0)

    # Visualize the correlations with a heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(ordered_correlation_matrix, annot=True, cmap='Blues', fmt=".2f", vmin=0, vmax=1, mask=mask)
    plt.title('Parameter Correlations for Good Scores (Threshold: {:.2f})'.format(threshold))

    # Strip 'log_' and 'sq_' prefixes from tick labels
    new_tick_labels = [label.replace('log_', '').replace('sq_', '').replace('outlier_threshold', 'outl_thresh').replace('chi_prop', 'chi_prop').replace('values', 'score') for label in ordered_correlation_matrix.columns]

    # Rotate tick labels for better readability
    plt.xticks(ticks=np.arange(len(new_tick_labels)) + 0.5, labels=new_tick_labels, rotation=30, ha='center')
    plt.yticks(ticks=np.arange(len(new_tick_labels)) + 0.5, labels=new_tick_labels, rotation=30, ha='right')

    # Save the figure as an SVG file for parameter correlations
    svg_file_path = file_path + '/parameter_correlations.svg'  # Change the path as needed
    plt.savefig(svg_file_path, format='svg')
    print(f"Parameter correlations plot saved to: {svg_file_path}")
    
    # Select the top 3 most important parameters
    nparams = 3
    top_x_params = param_importance.index[:2*nparams].tolist()
    top_x_params = [param for param in top_x_params if param in par_cols]
    top_x_params = top_x_params[:nparams]
    
    # Filter the correlation matrix for top_x_params
    filtered_corr_matrix = correlation_matrix.loc[top_x_params, top_x_params]

    # Mask the diagonal
    np.fill_diagonal(filtered_corr_matrix.values, 0)  # Set diagonal entries to 0

    # Get the maximum value for each row, ignoring the diagonal
    max_values = filtered_corr_matrix.max(axis=1)

    # Sort top_x_params based on the maximum values
    top_x_params = max_values.sort_values(ascending=False).index.tolist()
    top_x_params[0], top_x_params[1] = top_x_params[1], top_x_params[0]

    # Define a function to add jitter to values based on the scale of the column
    def add_jitter_to_values(values, strenght=0.1):
        # Calculate jitter strength as a small percentage of the range of values
        jitter_strength = strenght * (values.max() - values.min())
        # Add random noise to the values
        return values + np.random.uniform(-jitter_strength, jitter_strength, size=len(values))

    # Calculate the 0.8 quantile for the maximum color scale value
    cmin = np.quantile(df['values'], 0.2)
    cmax = np.max(df['values'])

    # Conditional squaring of scaled values: square the normalized values (values/cmax)
    color_values = np.where(df['values'] > 0, np.square(np.square(df['values'] / cmax)), 0)

    # Create a parallel coordinates plot using graph_objects
    nticks = 5
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=color_values, #df['values'],
                colorscale='Viridis', # Adjust transparency of the lines  # Use 'Blues' colormap for the lines
                # showscale=True
            ),
            # Add additional linear-scale dimensions 'qv', 'x', and 'replacement_rate'
            dimensions = [
                dict(                    
                    label=param.replace('log_', '').replace('sq_', '') if param in log_par_cols or param in sq_par_cols else param,        
                    values=add_jitter_to_values(df[param]) if param in cat_par_cols else df[param],  # Apply jitter to parameter values
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
                ) for param in top_x_params
            ]
        )
    )

    # Adjust the layout for a more compact visualization
    fig.update_layout(
        width=600,  # Adjust the width of the entire plot
        height=400  # Adjust the height of the entire plot
    )

    # Save the figure as an SVG file for parallel coordinates
    svg_file_path = file_path + "/parallel_coordinates.svg"
    fig.write_image(svg_file_path, format='svg')
    print(f"Parallel coordinates plot saved to: {svg_file_path}")

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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from plotly.io import write_image, show

import argparse
import json
import os
import itertools

from SDOstreamclust import clustering

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import roc_auc_score

from scipy.io.arff import loadarff 
from scipy.interpolate import griddata

import optuna
from optuna.samplers import TPESampler, RandomSampler


def optimize_sdc(trial, data, y, t=None, fixed_pams=None, int_pams=None, float_pams=None, log_pams=None, eval_pams=None):
    if fixed_pams is None:
        fixed_pams = {}
    
    if int_pams is None:
        int_pams = {}
    
    if float_pams is None:
        float_pams = {}
    
    if log_pams is None:
        log_pams = {}
    
    # Extract fixed parameters with defaults
    pam = {**fixed_pams}
    
    # Integer parameters
    for param_name, param_range in int_pams.items():
        if len(param_range) == 3:
            min_value, max_value, step_value = param_range
            pam[param_name] = trial.suggest_int(param_name, min_value, max_value, step=step_value)
        elif len(param_range) == 2:
            min_value, max_value = param_range
            pam[param_name] = trial.suggest_int(param_name, min_value, max_value)

    # Float parameters
    for param_name, param_range in float_pams.items():
        if len(param_range) == 3:
            min_value, max_value, step_value = param_range
            pam[param_name] = trial.suggest_float(param_name, min_value, max_value, step=step_value)
        elif len(param_range) == 2:
            min_value, max_value = param_range
            pam[param_name] = trial.suggest_float(param_name, min_value, max_value)

    # Float log scale parameters
    for param_name, param_range in log_pams.items():        
        min_value, max_value = param_range
        pam[param_name] = trial.suggest_float(param_name, min_value, max_value, log=True)
    
    # Default values if not provided by pam
    k = pam.get('k', 300)
    T = pam.get('T', 500)
    qv = pam.get('qv', 0.3)
    x = pam.get('x', 5)
    chi_min = pam.get('chi_min', 8)
    chi_prop = pam.get('chi_prop', 0.05)
    zeta = pam.get('zeta', 0.6)
    e = pam.get('e', 2)
    outlier_threshold = pam.get('outlier_threshold', 5.0)
    outlier_handling = pam.get('outlier_handling', False)
    rel_outlier_score = pam.get('rel_outlier_score', True)
    perturb = pam.get('perturb', 0.0)
    random_sampling = pam.get('random_sampling', True)
    freq_bins = pam.get('freq_bins', 1)
    max_freq = pam.get('max_freq', 1.0)
    input_buffer = pam.get('input_buffer', 0)
    
    # If t is None, generate a default time array from 1 to len(y)
    if t is None:
        t = np.arange(0, len(y))

    # Add block size parameters
    first_block_size = eval_pams.get('first_block_size', 1)
    block_size = eval_pams.get('block_size', 1)

    # Evaluate the algorithm n_eval times and store the results
    scores = []
    
    n_eval = eval_pams.get('n_eval', 1)
    for _ in range(n_eval):        
        # Generate a random integer for the seed
        current_seed = np.random.randint(0, 2**31 - 1)

        # Create the clustering algorithm with parameters from fixed and trial settings
        alg = clustering.SDOstreamclust(
            k=k,
            T=T,
            qv=qv,
            x=x,
            chi_min=chi_min,
            chi_prop=chi_prop,
            zeta=zeta,
            e=e,
            rel_outlier_score=rel_outlier_score,
            outlier_handling=outlier_handling,
            outlier_threshold=outlier_threshold,
            perturb=perturb,
            random_sampling=random_sampling,
            freq_bins=freq_bins,
            max_freq=max_freq,
            input_buffer=input_buffer,
            seed=current_seed
        )

        # Process the first block separately with size k
        chunk = data[:first_block_size, :]
        chunk_time = t[:first_block_size]
        p_, s_ = alg.fit_predict(chunk, chunk_time)
        
        plist = []
        slist = []

        plist.append(p_)
        slist.append(s_)
        for i in range(first_block_size, data.shape[0], block_size):
            chunk = data[i:i + block_size, :]
            chunk_time = t[i:i + block_size]
            p_, s_ = alg.fit_predict(chunk, chunk_time)
            
            plist.append(p_)
            slist.append(s_)
        p = np.concatenate(plist)  # clustering labels
        s = np.concatenate(slist)  # outlierness scores
        s = -1/(s+1)  # norm. to avoid inf scores

        # Evaluate the performance for each run
        warm_up = eval_pams.get('warm_up', 0)
        metric = eval_pams.get('metric', 'ari')
        
        if metric=='ari':
            score = adjusted_rand_score(p[warm_up:], y[warm_up:])    
        elif metric == 'outlier':
            score = roc_auc_score(y[warm_up:] < 0, p[warm_up:] < 0)
        scores.append(score)

        del alg

    # Return the median score over all evaluations    
    return np.median(scores)

# retrieve dataset from file into x,y arrays
def load_data(filename):

    dataname = filename.split("/")[-1].split(".")[0]
    arffdata = loadarff(filename)
    df_data = pd.DataFrame(arffdata[0])

    if(df_data['class'].dtypes == 'object'):
        df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))
        y = df_data['class'].str.strip().astype(int).to_numpy() #df_data['class'].to_numpy()
    else:
        y = df_data['class'].astype(int).to_numpy()
    # t = np.arange(len(y))
    # Generate random values
    random_values = np.random.uniform(0, len(y), len(y))
    # Sort the values to ensure monotonic increase
    t = np.sort(random_values)

    df_data.drop(columns=['class'], inplace=True)
    x = df_data.to_numpy()
    del df_data

    clusters = len(np.unique(y)) # num clusters in the GT
    outliers = np.sum(y==-1)
    if outliers > 0:
        clusters = clusters -1
    
    [n,m] = x.shape

    # normalize dataset
    x = MinMaxScaler().fit_transform(x)

    return t,x,y,n,m,clusters,outliers,dataname


def plot_boxplot(trials, param_name, metric='ari', n_bins=5, filepath=None):
    """
    Plot a box plot showing the distribution of scores for different bins of a parameter.
    
    Args:
        trials (list): List of all trials with their parameters.
        param_name (str): The name of the parameter to analyze.
        metric (str): Metric used for evaluating the performance (e.g., 'ari' or 'roc_auc').
        n_bins (int): Number of bins to divide the parameter values into.
        filepath (str or None): Directory path to save the figure. If None, the figure is shown interactively.
    """
    # Collect parameter values and scores
    param_values = [trial.params.get(param_name) for trial in trials]
    scores = [trial.value for trial in trials]
    
    # Create a DataFrame
    df = pd.DataFrame({
        'param_value': param_values,
        'score': scores
    })
    
    # Ensure no NaN values for the plot
    df = df.dropna()
    
    # Bin the parameter values
    df['param_bin'] = pd.cut(df['param_value'], bins=n_bins)
    
    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x='param_bin', y='score', data=df)    
    
    # plt.title(f"Box Plot of {metric} Scores by {param_name}")
    plt.xlabel(f'{param_name}')
    plt.ylabel(f"{metric}")
    
    # Rotate x-ticks for readability
    plt.xticks(rotation=20)
    
    # Set y-axis range from 0 to 1
    plt.ylim(0, 1)
    
    plt.grid(True)
    
    if filepath:
        plt.savefig(f"{filepath}/box_{param_name}.svg", format='svg')
        print(f"Box plot saved to {filepath}/box_{param_name}.svg")
        plt.close()
    else:
        plt.show()


def plot_violinplot(trials, param_name, metric='ari', n_bins=5, filepath=None):
    """
    Plot a violin plot showing the distribution of scores for different bins of a parameter.
    
    Args:
        trials (list): List of all trials with their parameters.
        param_name (str): The name of the parameter to analyze.
        metric (str): Metric used for evaluating the performance (e.g., 'ari' or 'roc_auc').
        n_bins (int): Number of bins to divide the parameter values into.
    """
    # Collect parameter values and scores
    param_values = [trial.params.get(param_name) for trial in trials]
    scores = [trial.value for trial in trials]
    
    # Create a DataFrame
    df = pd.DataFrame({
        'param_value': param_values,
        'score': scores
    })
    
    # Ensure no NaN values for the plot
    df = df.dropna()
    
    # Bin the parameter values
    df['param_bin'] = pd.cut(df['param_value'], bins=n_bins)
    
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='param_bin', y='score', data=df)

    # plt.title(f"Violin Plot of {metric} Scores by {param_name}")
    plt.xlabel(f'{param_name}')
    plt.ylabel(f"{metric}")
    plt.xticks(rotation=20)
    
    # Set y-axis range from 0 to 1
    plt.ylim(0, 1)
    
    plt.grid(True)
    if filepath:
        plt.savefig(f"{filepath}/violin_{param_name}.svg", format='svg')
        print(f"Violin plot saved to {filepath}/violin_{param_name}.svg")     
        plt.close()
    else:
        plt.show()


def plot_bar(trials, param_name, metric='ari', n_bins=5, filepath=None):
    """
    Plot parameter values with mean scores and error bars representing min and max range.
    
    Args:
        trials (list): List of all trials with their parameters.
        param_name (str): The name of the parameter to analyze.
        metric (str): Metric used for evaluating the performance (e.g., 'ari' or 'roc_auc').
        n_bins (int): Number of bins to divide the parameter values into.
        filepath (str or None): Directory path to save the figure. If None, the figure is shown interactively.
    """
    # Collect parameter values and scores
    param_values = [trial.params.get(param_name) for trial in trials]
    scores = [trial.value for trial in trials]
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'param_value': param_values,
        'score': scores
    })
    
    # Ensure no NaN values
    df = df.dropna()
    
    # Bin the parameter values
    bins = pd.cut(df['param_value'], bins=n_bins)
    df['param_bin'] = bins
    
    # Calculate mean, min, and max of scores for each bin
    summary_stats = df.groupby('param_bin')['score'].agg(['mean', 'min', 'max']).reset_index()
    
    # Get the bin centers
    bin_edges = bins.cat.categories
    bin_centers = [(edge.left + edge.right) / 2 for edge in bin_edges]
    
    # Plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.errorbar(bin_centers, summary_stats['mean'], 
                yerr=[summary_stats['mean'] - summary_stats['min'], summary_stats['max'] - summary_stats['mean']], 
                fmt='o', capsize=5, capthick=2, ecolor='red')
    
    plt.title(f"{param_name} vs. {metric} with Error Bars (Min/Max Range)")
    plt.xlabel(param_name)
    plt.ylabel(f"{metric} Score")
    plt.xticks(rotation=30)
    
    # Set y-axis range from 0 to 1
    plt.ylim(0, 1)

    plt.grid(True)
    
    if filepath:
        plt.savefig(f"{filepath}/bar_{param_name}.svg", format='svg')
        print(f"Bar plot saved to {filepath}/bar_{param_name}.svg")     
        plt.close()
    else:
        plt.show()


def plot_heat(trials, param_x, param_y, metric='ari', n_bins=5, n_ticks=5, filepath=None):
    """Generate a heatmap of parameter combinations and their scores."""
    
    # Extract parameter values and scores from trials
    values_x = []
    values_y = []
    scores = []
    
    for trial in trials:
        values_x.append(trial.params[param_x])
        values_y.append(trial.params[param_y])
        scores.append(trial.value)

    # Create a 2D array to store the scores in each bin
    xedges = np.linspace(min(values_x), max(values_x), n_bins + 1)
    yedges = np.linspace(min(values_y), max(values_y), n_bins + 1)

    # Initialize 2D lists to store all the scores in each bin
    bin_scores = [[[] for _ in range(n_bins)] for _ in range(n_bins)]

    # Place each trial score into its respective bin
    for i in range(len(values_x)):
        x_idx = np.digitize(values_x[i], xedges) - 1
        y_idx = np.digitize(values_y[i], yedges) - 1
        if x_idx < n_bins and y_idx < n_bins:  # Ensure index is within bounds
            bin_scores[x_idx][y_idx].append(scores[i])
    
    # Calculate the median for each bin
    heatmap_median = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            if bin_scores[i][j]:  # Only calculate median if there are scores
                heatmap_median[i, j] = np.max(bin_scores[i][j])
            else:
                heatmap_median[i, j] = np.nan  # Assign NaN if no scores

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(heatmap_median, cmap="Blues", annot=True, vmin=0, vmax=1)

    # Custom tick positions with left, right, and evenly distributed middle ticks
    x_ticks = np.linspace(0, n_bins, n_ticks)  # Positions for n_ticks x-ticks
    y_ticks = np.linspace(0, n_bins, n_ticks)  # Positions for n_ticks y-ticks

    # Create tick labels corresponding to param_x and param_y values
    # Ensure that the first and last ticks correspond exactly to min and max values
    x_labels = [f'{np.round(val, 2)}' for val in np.linspace(min(values_x), max(values_x), n_ticks)]
    y_labels = [f'{np.round(val, 2)}' for val in np.linspace(min(values_y), max(values_y), n_ticks)]


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


def plot_contour(trials, param_x, param_y, metric='ari', filepath=None):
    """Generate a contour plot of parameter combinations and their scores."""
    
    # Extract parameter values and scores from trials
    values_x = []
    values_y = []
    scores = []
    
    for trial in trials:
        values_x.append(trial.params[param_x])
        values_y.append(trial.params[param_y])
        scores.append(trial.value)

    # Create a grid of points for contour plotting
    x_grid = np.linspace(min(values_x), max(values_x), 100)
    y_grid = np.linspace(min(values_y), max(values_y), 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Interpolate scores onto the grid
    Z_grid = griddata((values_x, values_y), scores, (X_grid, Y_grid), method='linear')

    # Plot the contour plot
    plt.figure(figsize=(12, 6))
    contour = plt.contourf(X_grid, Y_grid, Z_grid, levels=20, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(contour, label=f'{metric}')
    
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    # plt.title(f'Contour Plot of {param_x} vs {param_y} ({metric})')

    if filepath:
        plt.savefig(f"{filepath}/xcontour_{param_x}_{param_y}.svg", format='svg')
        print(f"Contour plot saved to {filepath}/xcontour_{param_x}_{param_y}.svg")
        plt.close()
    else:
        plt.show()


def plot_all(study, metric='ari', n_bins=5, filepath=None):
    """Generate heatmaps for all combinations of parameters."""
    param_names = list(study.best_params.keys())  # Extract the parameter names
    for i in range(len(param_names)):
        param_x = param_names[i]
        plot_bar(study.trials, param_x, metric=metric, n_bins=n_bins, filepath=filepath)
        plot_boxplot(study.trials, param_x, metric=metric, n_bins=n_bins, filepath=filepath)
        plot_violinplot(study.trials, param_x, metric=metric, n_bins=n_bins, filepath=filepath)
        for j in range(i + 1, len(param_names)):
            param_x = param_names[i]
            param_y = param_names[j]
            print(f'Generating heatmap for {param_x} vs {param_y}')
            plot_heat(study.trials, param_x, param_y, metric=metric, n_bins=n_bins, filepath=filepath)

            # # Generate contour plot
            # print(f'Generating contour maps for {param_x} vs {param_y}')
            # plot_contour(study.trials, param_x, param_y, metric=metric, filepath=filepath)

            # # Generate contour plot
            # fig = optuna.visualization.plot_contour(study, params=[param_x, param_y])
            
            # if filepath:                
            #     fig.write_image(f"{filepath}/contour_{param_x}_{param_y}.svg", format='svg')
            #     print(f"Contour saved to {filepath}/heat_{param_x}_{param_y}.svg")
            # else:
            #     show(fig)
            

def main(json_file, data_file):
    # Load parameters from JSON file
    with open(json_file, 'r') as f:
        params = json.load(f)
    
    fixed_pams = params.get('fixed_pams', {})
    int_pams = params.get('int_pams', {})
    float_pams = params.get('float_pams', {})
    log_pams = params.get('log_pams', {})
    eval_pams = params.get('eval_pams', {})
    grid_pams = params.get('grid_pams', {})

    # Load data
    t, x, y, n, m, clusters, outliers, dataname = load_data(data_file)

    result_path0 = eval_pams.get('result_path', None)

    # Extract parameter names and values
    grid_names = list(grid_pams.keys())
    grid_values = [grid_pams[name] for name in grid_names]
    # Iterate over all combinations of parameter values using itertools.product
    for values in itertools.product(*grid_values):
        result_path = result_path0 + "/"
        for name, value in zip(grid_names, values):
            fixed_pams[name] = value
            result_path = result_path + name + str(value).replace('.', '-')
            
        # Set up the study and optimizer
        # sampler = TPESampler(seed=10)
        sampler = RandomSampler(seed=10)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        
        # Optimize the objective function
        n_trials = eval_pams.get('n_trials', 50)    
        study.optimize(lambda trial: optimize_sdc(trial, x, y, t, fixed_pams, int_pams, float_pams, log_pams, eval_pams), n_trials=n_trials, n_jobs=-1)
        
        # Print the best parameters
        print('Best parameters:', study.best_params)
        print('Best score:', study.best_value)
        
        if result_path:
            # Ensure the directory exists
            os.makedirs(result_path, exist_ok=True)
            
            result_filename = os.path.join(result_path, 'result.json')
            with open(result_filename, 'w') as f:
                json.dump({
                    'fixed_pams': fixed_pams,
                    'best_params': study.best_params,
                    'best_value': study.best_value,
                    'trials': [t.params for t in study.trials],
                    'values': [t.value for t in study.trials]
                }, f, indent=4)
        
        # Plot the heatmap for two selected parameters
        metric = eval_pams.get('metric', 'ari')
        n_bins = eval_pams.get('n_bins', 5)
        # Plot heatmaps for all parameter combinations
        plot_all(study, metric=metric, n_bins=n_bins, filepath=result_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize clustering algorithm parameters.')
    parser.add_argument('--json', type=str, required=True, help='Path to JSON file with parameters.')
    parser.add_argument('--data', type=str, required=True, help='Path to data file.')
    
    args = parser.parse_args()
    main(args.json, args.data)

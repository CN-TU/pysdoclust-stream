#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from scipy.io.arff import loadarff 

import os

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
    """
    Projects x to a 3D array where the first two dimensions are the projections of d1 and d2,
    and the third dimension is the norm of the other dimensions.
    
    Parameters:
    - x: np.ndarray, input array of shape (n_samples, n_features)
    - d1: int, index of the first dimension to project
    - d2: int, index of the second dimension to project
    
    Returns:
    - projected: np.ndarray, 3D array of shape (n_samples, n_features - 2, 2)
    """
    # Ensure d1 and d2 are within the bounds of x's dimensions
    if d1 >= x.shape[1] or d2 >= x.shape[1]:
        raise ValueError("d1 and d2 must be valid indices of the dimensions of x.")
    
    # Get the projections
    projections = x[:, [d1, d2]]
    
    # Calculate the norm of the other dimensions
    other_dims = np.delete(x, [d1, d2], axis=1)
    norms = np.linalg.norm(other_dims, axis=1)/other_dims.shape[1]
    
    # Stack the projections and norms into a 3D array
    projected = np.empty((x.shape[0], 3))
    projected[:, 0:2] = projections
    projected[:, 2] = norms
    
    return projected

syn_types = ['base', 'moving', 'nonstat', 'sequential']

for syn_type in syn_types:
    filename = f'evaluation_tests/data/synthetic/{syn_type}_normal/{syn_type}_data_1.arff'
    t,x,y,n,m,clusters,outliers,dataname = load_data(filename)

    num_gt_labels = len(np.unique(y[y>-1]))
    cmap_gt = plt.get_cmap('Paired', num_gt_labels)
    norm_gt = plt.Normalize(vmin=0, vmax=num_gt_labels-1)

    # Create a figure with 5 subplots for each dimension of x
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))  # 5 rows and 1 column

    # Create 5 plots for each dimension of x (f0, f1, f2, f3, f4)
    for i in range(3):
        axs[i].scatter(t[y > -1], x[y > -1, i], c=y[y > -1], s=1, cmap=cmap_gt, norm=norm_gt, label='Class > -1')
        axs[i].scatter(t[y == -1], x[y == -1, i], c='black', s=1, label='Class = -1')

        axs[i].set_xlabel('t')
        #axs[i].set_ylabel(f'f{i}')  # Dynamic labeling based on dimension
        # axs[i].set_title(f'Plot of Time vs f{i}')  # Add title for clarity
        # axs[i].legend(loc='best')  # Optional: add legend

    axs[0].set_ylabel('f0')
    axs[1].set_ylabel('f1')
    axs[2].set_ylabel('f2')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Directory to save the frames
    frames_dir = 'frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Save the figure as an EPS file
    frame_file = os.path.join(frames_dir, f'frame_{syn_type}_gt.eps')
    plt.savefig(frame_file)

    plt.close()
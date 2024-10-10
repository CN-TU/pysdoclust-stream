#!/usr/bin/env python3

from SDOstreamclust import clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from scipy.io.arff import loadarff 

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import roc_auc_score

import moviepy.editor as mpy
import os

# Directory to save the frames
frames_dir = 'frames'
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# retrieve dataset from file into x,y arrays
def load_data(filename):

    dataname = filename.split("/")[-1].split(".")[0]
    arffdata = loadarff(filename)
    df_data = pd.DataFrame(arffdata[0])

    if(df_data['class'].dtypes == 'object'):
        df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))

    y = df_data['class'].str.strip().astype(int).to_numpy() #df_data['class'].to_numpy()
    t = np.arange(len(y))
    # # Generate random values
    # random_values = np.random.uniform(0, len(y), len(y))
    # # Sort the values to ensure monotonic increase
    # t = np.sort(random_values)

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

filename = 'evaluation_tests/data/example/concept_drift.arff'
t,x,y,n,m,clusters,outliers,dataname = load_data(filename)

# Define marker shapes, which will cycle if the number of labels exceeds the number of available shapes
marker_shapes = ['.', 'o', 's', 'd', '^', 'v', '<', '>', 'h', 'p', '*', '+', '1', '2', '3', '4', '8', 'P', 'D', 'H']
num_shapes = len(marker_shapes)

num_gt_labels = len(np.unique(y[y>-1]))
cmap_gt = plt.get_cmap('tab20', num_gt_labels)
norm_gt = plt.Normalize(vmin=0, vmax=num_gt_labels-1)

frame_files = []

# Initialize index
idx = 0

# Plot and save each frame
freq = 550
for obs_t in range(int(freq/2), 10000, int(freq/2)):
    print(obs_t)
    # Create a new figure for each frame
    fig, ax = plt.subplots(figsize=(12, 12))

    # Filter points for the current time slice
    time_min = obs_t - int(freq/2)
    time_max = obs_t + int(freq/2)
    mask = (t >= time_min) & (t <= time_max)   
    filtered_points = x[mask] 
    filtered_gt_labels = y[mask].astype(int)
    
    # Plot the filtered points
    for label in np.unique(filtered_gt_labels):
        if label != -1:
            shape = marker_shapes[label % num_shapes]  # Get shape for the current label
            ax.scatter(filtered_points[filtered_gt_labels != -1, 0], 
                       filtered_points[filtered_gt_labels != -1, 1], 
                       c=filtered_gt_labels[filtered_gt_labels != -1], 
                       cmap=cmap_gt, 
                       norm=norm_gt,
                       s=5, 
                       marker=shape)
        else:
            ax.scatter(filtered_points[filtered_gt_labels == -1, 0], 
                       filtered_points[filtered_gt_labels == -1, 1], 
                       c='black', s=5, marker='x', label='Outliers')
    
    ax.set_title(f'Ground Truth at Time: {obs_t} +/- {int(freq/2)}')
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Save the frame
    frame_file = os.path.join(frames_dir, f'frame_{idx:04d}.png')
    plt.savefig(frame_file)
    frame_files.append(frame_file)

    # Save the frame as SVG
    svg_frame_file = os.path.join(frames_dir, f'frame_{idx:04d}.svg')
    plt.savefig(svg_frame_file)

    # Close the figure to free memory
    plt.close(fig)

    # Increment the index
    idx += 1

# Create a video from the saved frames
clip = mpy.ImageSequenceClip(frame_files, fps=30)  # Increased fps for smoother video

# Optional: Resize the video if needed
# clip = clip.resize(height=720)  # Resize to height of 720 pixels, maintaining aspect ratio

video_file = 'conglomerate_drift_gt.mp4'
# Write the video file with higher bitrate
clip.write_videofile(video_file, codec='libx264', bitrate='2000k')  # Increase bitrate for better quality


# Clean up frames
for frame_file in frame_files:
    os.remove(frame_file)

print(f'Video saved as {video_file}')

fig = plt.figure(figsize=(12,4))
# cmap = plt.get_cmap('tab20', len(np.unique(p)))

# Plotting ground truth
for i in range(3):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    ax.scatter3D(t[y>-1], x[y>-1,0], x[y>-1,1], s=5, c=y[y>-1], cmap=cmap_gt, norm=norm_gt)
    ax.scatter3D(t[y==-1], x[y==-1,0], x[y==-1,1], s=5, c='black')
    ax.view_init(azim=280+i*30, elev=20)
    ax.set_xlabel('time')
    ax.set_ylabel('f0')
    ax.set_zlabel('f1')

plt.tight_layout()

# Save the frame
frame_file = os.path.join(frames_dir, f'frame_{idx+1:04d}.png')
plt.savefig(frame_file)
frame_files.append(frame_file)

# Save the frame as SVG
svg_frame_file = os.path.join(frames_dir, f'frame_{idx+1:04d}.svg')
plt.savefig(svg_frame_file)
plt.close(fig)


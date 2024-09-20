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

filename = 'evaluation_tests/data/real/occupancy.arff'
t,x,y,n,m,clusters,outliers,dataname = load_data(filename)

# Set the initial block to be of size k
first_block_size = 1000
block_size = 200  # Remaining blocks will have this size

# Controls the time window of ground truth / predictions points shown at each frame: obs_T +/- (T / f_T), 
# obs_T is time of model (observer) snapshot
f_T = 10

k = 400 # Model size
T = 1000 # Time Horizon
# ibuff = 10 # input buffer
chi_prop = 0.05
qv = 0.2
e = 3
outlier_threshold = 5
outlier_handling = True
x_ = 5
freq_bins= 1 #10
max_freq= 1# 1100
# chi_prop=0.05, e=2, outlier_threshold=5.0, outlier_handling=False 
classifier = clustering.SDOstreamclust(
    k=k, 
    T=T, 
    qv=qv,
    x=x_, 
    chi_prop=chi_prop, 
    e=e, 
    outlier_threshold=outlier_threshold, 
    outlier_handling=outlier_handling,
    freq_bins=freq_bins,
    max_freq=max_freq)

all_predic = []
all_scores = []

all_obs_points = []
all_obs_labels = []
all_obs_t = []

# Process the first block separately 
chunk = x[:first_block_size, :]
chunk_time = t[:first_block_size]
labels, outlier_scores = classifier.fit_predict(chunk, chunk_time)
obs_points, obs_labels, observations, av_observations = classifier.get_observers()

# Handle n_active calculation
n_active = int((1-qv)*(len(obs_points)-1)) + 1

# Store the first block's results
all_obs_points.append(obs_points[:n_active])
all_obs_labels.append(obs_labels[:n_active])
all_obs_t.append(chunk_time[-1])

all_predic.append(labels)
all_scores.append(outlier_scores)

# Process the remaining blocks with size block_size
for i in range(first_block_size, x.shape[0], block_size):
    chunk = x[i:i + block_size, :]
    chunk_time = t[i:i + block_size]
    labels, outlier_scores = classifier.fit_predict(chunk, chunk_time)
    obs_points, obs_labels, observations, av_observations = classifier.get_observers()
    
    n_active = int((1-qv)*(len(obs_points)-1)) + 1

    all_obs_points.append(obs_points[:n_active])
    all_obs_labels.append(obs_labels[:n_active])
    all_obs_t.append(chunk_time[-1])

    all_predic.append(labels)
    all_scores.append(outlier_scores)
p = np.concatenate(all_predic) # clustering labels
s = np.concatenate(all_scores) # outlierness scores
s = -1/(s+1) # norm. to avoid inf scores

# Thresholding top outliers based on Chebyshev's inequality (88.9%)
# th = np.mean(s)+3*np.std(s)
# p[s>th]=-1

# Evaluation metrics
print("Adjusted Rand Index (clustering):", adjusted_rand_score(y,p))
# print("ROC AUC score (outlier/anomaly detection):", roc_auc_score(y<0,s))
try:
    print("ROC AUC score (outlier/anomaly detection):", roc_auc_score(y<0,p<0))
except ValueError as e:
    print("Outlier rated (detected):", sum(p<0)/len(p))
 
unique_predic_labels = np.unique(p)  # Unique labels from clustering predictions
unique_obs_labels = np.unique(np.concatenate(all_obs_labels))  # Unique observed labels
# Combine both to get all unique labels
all_unique_labels = np.unique(np.concatenate([unique_predic_labels, unique_obs_labels]))
if -1 not in all_unique_labels:
    all_unique_labels = np.append(all_unique_labels, -1)

le = LabelEncoder().fit(all_unique_labels)
p = le.transform(p) -1

cmap = plt.get_cmap('tab20', len(le.classes_))
norm = plt.Normalize(vmin=min(np.unique(p)), vmax=max(np.unique(p)))

# Define marker shapes, which will cycle if the number of labels exceeds the number of available shapes
marker_shapes = ['.', 'o', 's', 'd', '^', 'v', '<', '>', 'h', 'p', '*', '+', '1', '2', '3', '4', '8', 'P', 'D', 'H']
num_shapes = len(marker_shapes)

cmap_gt = plt.get_cmap('Dark2', len(np.unique(y)))
frame_files = []

# Plot and save each frame
for idx, (obs_points, obs_labels, obs_t) in enumerate(zip(all_obs_points, all_obs_labels, all_obs_t)):
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 6))

    # Plot filtered points with corresponding shapes
    time_min = obs_t - T/f_T
    time_max = obs_t + T/f_T
    mask = (t >= time_min) & (t <= time_max)
    filtered_points = x[mask]
    filtered_labels = p[mask].astype(int)
    filtered_gt_labels = y[mask].astype(int)
    for i in range(3):
        ax = fig.add_subplot(3, 3, i+1, projection='3d')
        for label in all_unique_labels:            
            if label != -1:
                shape = marker_shapes[label % num_shapes]  # Use filtered_labels for marker shapes
                ax.scatter3D(
                                filtered_points[filtered_labels == label, 0], 
                                filtered_points[filtered_labels == label, 1],  
                                filtered_points[filtered_labels == label, 2], 
                                c=filtered_labels[filtered_labels == label], 
                                cmap=cmap, norm=norm, s=10, marker=shape)
            else:
                ax.scatter3D(
                                filtered_points[filtered_labels==-1, 0], 
                                filtered_points[filtered_labels==-1, 1], 
                                filtered_points[filtered_labels==-1, 2], 
                                c='black', s=10, marker='x', label='Outliers')
        ax.view_init(azim=280 + i * 30, elev=20)
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')
        ax.set_zlabel('Feature X')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Predictions at Time: {obs_t} +/- {T/f_T}')
    

    # Observers
    points = np.array(obs_points)
    labels = le.transform(np.array(obs_labels)) - 1

    for i in range(3):
        ax = fig.add_subplot(3, 3, i+4, projection='3d')
        for label in all_unique_labels:
            if label != -1:
                shape = marker_shapes[label % num_shapes]
                ax.scatter3D(
                    points[labels == label, 0], 
                    points[labels == label, 1], 
                    points[labels == label, 2], 
                    c=labels[labels == label], 
                    cmap=cmap, norm=norm, s=10, marker=shape)
            else:
                ax.scatter3D(
                    points[labels == -1, 0], 
                    points[labels == -1, 1], 
                    points[labels == -1, 2], 
                    c='black', s=10, marker='x', label='Outliers')

        ax.view_init(azim=280 + i * 30, elev=20)
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')
        ax.set_zlabel('Feature X')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Observers at Time: {obs_t}')

    # Ground Truth
    for i in range(3):
        ax = fig.add_subplot(3, 3, i+7, projection='3d')
        for label in all_unique_labels:
            if label != -1:
                shape = marker_shapes[label % num_shapes]
                ax.scatter3D(
                    filtered_points[filtered_gt_labels == label, 0], 
                    filtered_points[filtered_gt_labels == label, 1], 
                    filtered_points[filtered_gt_labels == label, 2], 
                    c=filtered_gt_labels[filtered_gt_labels == label], 
                    cmap=cmap_gt, s=10, marker=shape)
            else:
                ax.scatter3D(
                    filtered_points[filtered_gt_labels == -1, 0], 
                    filtered_points[filtered_gt_labels == -1, 1], 
                    filtered_points[filtered_gt_labels == -1, 2], 
                    c='black', s=10, marker='x', label='Outliers')
                
        ax.set_title(f'Ground Truth at Time: {obs_t} +/- {T / f_T}')
        ax.view_init(azim=280 + i * 30, elev=20)
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')
        ax.set_zlabel('Feature X')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Observers at Time: {obs_t}')

    plt.tight_layout()

    # Save the frame
    frame_file = os.path.join(frames_dir, f'frame_{idx:04d}.png')
    plt.savefig(frame_file)
    frame_files.append(frame_file)
    plt.close(fig)

fig = plt.figure(figsize=(18,6))
# cmap = plt.get_cmap('tab20', len(np.unique(p)))

for i in range(3):
    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    ax.scatter3D(t[p>-1], x[p>-1,0], x[p>-1,1], s=5, c=p[p>-1], cmap=cmap, norm=norm)
    ax.scatter3D(t[p==-1], x[p==-1,0], x[p==-1,1], s=5, c='black')
    ax.view_init(azim=280+i*30, elev=20)
    ax.set_xlabel('time')
    ax.set_ylabel('f0')
    ax.set_zlabel('f1')

# Plotting ground truth
for i in range(3):
    ax = fig.add_subplot(2, 3, i+4, projection='3d')
    ax.scatter3D(t[y>-1], x[y>-1,0], x[y>-1,1], s=5, c=y[y>-1], cmap=cmap_gt)
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
plt.close(fig)

# Create a video from the saved frames
clip = mpy.ImageSequenceClip(frame_files, fps=10)
video_file = 'occupancy.mp4'
clip.write_videofile(video_file, codec='libx264')

# Clean up frames
for frame_file in frame_files:
    os.remove(frame_file)

print(f'Video saved as {video_file}')
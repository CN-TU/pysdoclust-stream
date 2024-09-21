#!/usr/bin/env python3

from SDOstreamclust import clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

filename = 'evaluation_tests/data/real/retail.arff'
t,x,y,n,m,clusters,outliers,dataname = load_data(filename)

# Set the initial block to be of size k
first_block_size = 100
block_size = 5  # Remaining blocks will have this size

# Controls the time window of ground truth / predictions points shown at each frame: obs_T +/- (T / f_T), 
# obs_T is time of model (observer) snapshot
f_T = 10

k = 75 # Model size
T = 150 # Time Horizon
# ibuff = 10 # input buffer
chi_prop = 0.15
qv = 0.1
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
# print("ROC AUC score (outlier/anomaly detection):", roc_auc_score(y<0,p<0))
print("Outlier rated (detected):", sum(p<0)/len(p))
 

unique_predic_labels = np.unique(p)  # Unique labels from clustering predictions
unique_obs_labels = np.unique(np.concatenate(all_obs_labels))  # Unique observed labels
# Combine both to get all unique labels
all_unique_labels = np.unique(np.concatenate([unique_predic_labels, unique_obs_labels]))
if -1 not in all_unique_labels:
    all_unique_labels = np.append(all_unique_labels, -1)
le = LabelEncoder().fit(all_unique_labels)
p = le.transform(p)-1

num_labels = len(all_unique_labels) - 1  # Number of unique labels (minus outlier label)
cmap = plt.get_cmap('tab20', num_labels)
norm = plt.Normalize(vmin=0, vmax=num_labels-1)

frame_files = []

for idx, (obs_points, obs_labels, obs_t) in enumerate(zip(all_obs_points, all_obs_labels, all_obs_t)):

    fig, ax = plt.subplots(figsize=(8, 6))

    # Prepare the data for stripplot
    time_min = obs_t - T/f_T
    time_max = obs_t + T/f_T
    mask = (t >= time_min) & (t <= time_max)
    filtered_points = x[mask].ravel()
    filtered_labels = p[mask].astype(int)
    filtered_gt_labels = y[mask].astype(int)
    
    points = np.array(obs_points)
    labels = le.transform(np.array(obs_labels))-1    

    # Create DataFrames for filtered points, observers, and ground truth
    plot_data_filtered = pd.DataFrame({
        'Feature': filtered_points.ravel(),  # x values
        'Source': np.array(['Prediction'] * len(filtered_points)),  # y values as categories
        'Class': filtered_labels  # hue for class labels
    })

    plot_data_obs = pd.DataFrame({
        'Feature': points.ravel(),  # x values
        'Source': np.array(['Model'] * len(points)),  # y values as categories
        'Class': labels  # hue for class labels
    })

    plot_data_gt = pd.DataFrame({
        'Feature': filtered_points.ravel(),  # x values
        'Source': np.array(['Ground Truth'] * len(filtered_points)),  # y values as categories
        'Class': filtered_gt_labels  # hue for class labels
    })

    # Concatenate the DataFrames
    plot_data = pd.concat([plot_data_filtered, plot_data_obs, plot_data_gt], ignore_index=True)

    # Create the stripplot
    sns.stripplot(data=plot_data, x='Feature', y='Source', hue='Class', ax=ax, jitter=True, size=4, palette=cmap, legend=False, orient='h')

    ax.set_title(f'Stripplot at Time: {obs_t} +/- {T/f_T}')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Source')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()

    # Save the frame
    frame_file = os.path.join(frames_dir, f'frame_{idx:04d}.png')
    plt.savefig(frame_file)
    frame_files.append(frame_file)
    plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(8, 6))

# Plot predictions
axs[0].scatter(t, x, s=5, c=p, cmap=cmap, norm=norm)
axs[0].set_title('Predictions')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Feature')
axs[0].set_xlim(t.min(), t.max())
axs[0].set_ylim(0, 1)

# Plotting ground truth
axs[1].scatter(t, x, s=5, c=y, cmap=cmap, norm=norm)
axs[1].set_title('Ground Truth')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Feature')
axs[1].set_xlim(t.min(), t.max())
axs[1].set_ylim(0, 1)

plt.tight_layout()

# Save the frame
frame_file = os.path.join(frames_dir, f'frame_{idx+1:04d}.png')
plt.savefig(frame_file)
frame_files.append(frame_file)
plt.close(fig)

# Create a video from the saved frames
clip = mpy.ImageSequenceClip(frame_files, fps=10)
video_file = 'retail.mp4'
clip.write_videofile(video_file, codec='libx264')

# Clean up frames
for frame_file in frame_files:
    os.remove(frame_file)

print(f'Video saved as {video_file}')
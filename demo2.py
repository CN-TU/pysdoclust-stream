#!/usr/bin/env python3

from SDOstreamclust import clustering
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from scipy.io.arff import loadarff 

def plotdata(x,y,t,pltname):

    if x.shape[1]>1:
        feats = np.sort(np.random.choice(x.shape[1], 2, replace=False))
    else:
        feats = np.sort(np.random.choice(x.shape[1], 2, replace=True))

    x = x[:,feats]
    # t = np.arange(len(y))
    y = y.astype(int)
    f0 = 'f'+str(feats[0])
    f1 = 'f'+str(feats[1])

    gs_kw = dict(width_ratios=[1, 2], height_ratios=[1, 1])
    fig, axes = plt.subplot_mosaic([['left', 'upper right'],['left', 'lower right']], gridspec_kw=gs_kw, figsize=(10, 4), layout="constrained")
    #for k, ax in axd.items():
    #    annotate_axes(ax, f'axd[{k!r}]', fontsize=14)

    #fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.get_cmap('tab20', len(np.unique(y)))
    if np.sum(y==-1)==0:
        axes['left'].scatter(x[:,0], x[:,1], s=5, c=y, alpha=0.7, cmap=cmap) # ,edgecolors='w'
        axes['upper right'].scatter(t, x[:,0], s=5, c=y, alpha=0.7, cmap=cmap)
        axes['lower right'].scatter(t, x[:,1], s=5, c=y, alpha=0.7, cmap=cmap)
    else:
        axes['left'].scatter(x[y>-1,0], x[y>-1,1], s=5, c=y[y>-1], alpha=0.7, cmap=cmap) # ,edgecolors='w'
        axes['left'].scatter(x[y==-1,0], x[y==-1,1], s=5, c='black', alpha=0.7, cmap=cmap) # edgecolors='w'
        axes['upper right'].scatter(t[y>-1], x[y>-1,0], s=5, c=y[y>-1], alpha=0.7, cmap=cmap)
        axes['upper right'].scatter(t[y==-1], x[y==-1,0], s=5, c='black', alpha=0.7, cmap=cmap)
        axes['lower right'].scatter(t[y>-1], x[y>-1,1], s=5, c=y[y>-1], alpha=0.7, cmap=cmap)
        axes['lower right'].scatter(t[y==-1], x[y==-1,1], s=5, c='black', alpha=0.7, cmap=cmap)

    axes['left'].set_ylabel(f0)
    axes['left'].set_xlabel(f1)
    axes['upper right'].set_ylabel(f0)
    axes['lower right'].set_xlabel('time')
    axes['lower right'].set_ylabel(f1)
    axes['upper right'].yaxis.set_label_position("right")
    axes['lower right'].yaxis.set_label_position("right")

    # plt.savefig(pltname) 
    # plt.close()
    plt.show()

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
    random_values = random_values = np.random.uniform(0, len(y), len(y))
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

filename = 'evaluation_tests/data/example/concept_drift.arff'
t,x,y,n,m,clusters,outliers,dataname = load_data(filename)

k = 600 # Model size
T = 1000 # Time Horizon
# ibuff = 10 # input buffer
chi_prop = 0.025
e = 3
outlier_threshold = 3
outlier_handling = True
x_ = 7
# chi_prop=0.05, e=2, outlier_threshold=5.0, outlier_handling=False 
classifier = clustering.SDOstreamclust(k=k, T=T, x=x_, chi_prop=chi_prop, e=e, outlier_threshold=outlier_threshold, outlier_handling=outlier_handling)

all_predic = []
all_scores = []

all_obs_points = []
all_obs_labels = []
all_obs_t = []

block_size = 200 # 1 would be point per-point processing
for i in range(0, x.shape[0], block_size):
    chunk = x[i:i + block_size, :]
    chunk_time = t[i:i + block_size]
    labels, outlier_scores = classifier.fit_predict(chunk, chunk_time)
    obs_points, obs_labels, observations, av_observations = classifier.get_observers()
    
    n_active = int(0.7*(len(obs_points)-1)) + 1

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
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import roc_auc_score
print("Adjusted Rand Index (clustering):", adjusted_rand_score(y,p))
# print("ROC AUC score (outlier/anomaly detection):", roc_auc_score(y<0,s))
print("ROC AUC score (outlier/anomaly detection):", roc_auc_score(y<0,p<0))

 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

unique_predic_labels = np.unique(p)  # Unique labels from clustering predictions
unique_obs_labels = np.unique(np.concatenate(all_obs_labels))  # Unique observed labels
# Combine both to get all unique labels
all_unique_labels = np.unique(np.concatenate([unique_predic_labels, unique_obs_labels]))
le = LabelEncoder().fit(all_unique_labels)
p = le.transform(p) -1

cmap = plt.get_cmap('tab20', len(le.classes_))
norm = plt.Normalize(vmin=min(np.unique(p)), vmax=max(np.unique(p)))

cmap_gt = plt.get_cmap('Dark2', len(np.unique(y)))

# Plotting
# for obs_points, obs_labels, obs_t in zip(all_obs_points, all_obs_labels, all_obs_t):
#     points = np.array(obs_points)
#     labels = le.transform(np.array(obs_labels)) - 1

#     time_min = obs_t - T
#     time_max = obs_t + T
#     mask = (t >= time_min) & (t <= time_max)
#     filtered_points = x[mask]
#     filtered_labels = p[mask].astype(int)
#     filtered_gt_labels = y[mask].astype(int)

#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
#     # Plot filtered points from x (left plot)
#     scatter_filtered = axes[0].scatter(filtered_points[filtered_labels!=-1, 0], filtered_points[filtered_labels!=-1, 1], c=filtered_labels[filtered_labels!=-1], cmap=cmap, norm=norm, s=10, marker='.')
#     axes[0].scatter(filtered_points[filtered_labels==-1, 0], filtered_points[filtered_labels==-1, 1], 
#                 c='black', s=10, marker='d', label='Outliers')

#     axes[0].set_title(f'Predictions at Time: {obs_t} +/- {T}')
#     axes[0].set_xlabel('Feature 0')
#     axes[0].set_ylabel('Feature 1')
#     axes[0].set_xlim(0, 1)
#     axes[0].set_ylim(0, 1)

#     scatter_obs = axes[1].scatter(points[:, 0], points[:, 1], c=labels, cmap=cmap, norm=norm, s=10, marker='x')
#     axes[1].set_title(f'Observers at Time: {obs_t}')
#     axes[1].set_xlabel('Feature 0')
#     axes[1].set_ylabel('Feature 1')
#     axes[1].set_xlim(0, 1)
#     axes[1].set_ylim(0, 1)

#     scatter_gt = axes[2].scatter(filtered_points[filtered_gt_labels!=-1, 0], filtered_points[filtered_gt_labels!=-1, 1], c=filtered_gt_labels[filtered_gt_labels!=-1], cmap=cmap_gt, s=10, marker='.')
#     axes[2].scatter(filtered_points[filtered_gt_labels==-1, 0], filtered_points[filtered_gt_labels==-1, 1], 
#                 c='black', s=10, marker='d', label='Outliers')
#     axes[2].set_title(f'Ground Truth at Time: {obs_t} +/- {T}')
#     axes[2].set_xlabel('Feature 0')
#     axes[2].set_ylabel('Feature 1')
#     axes[2].set_xlim(0, 1)
#     axes[2].set_ylim(0, 1)

#     # axes[0].legend(*scatter_filtered.legend_elements(), title="Labels", loc="upper right")
#     # axes[1].legend(*scatter_obs.legend_elements(), title="Labels", loc="upper right")
#     # axes[2].legend(*scatter_gt.legend_elements(), title="Labels", loc="upper right")

#     plt.tight_layout()
#     plt.show()

# fig = plt.figure(figsize=(15,4))
# cmap = plt.get_cmap('tab20', len(np.unique(p)))

# for i in range(3):
#     ax = fig.add_subplot(1, 3, i+1, projection='3d')
#     ax.scatter3D(t[p>-1], x[p>-1,0], x[p>-1,1], s=5, c=p[p>-1], cmap=cmap)
#     ax.scatter3D(t[p==-1], x[p==-1,0], x[p==-1,1], s=5, c='black')
#     ax.view_init(azim=280+i*30, elev=20)
#     ax.set_xlabel('time')
#     ax.set_ylabel('f0')
#     ax.set_zlabel('f1')

fig = plt.figure(figsize=(15,8))
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

plt.show()


plotdata(x,y,t,'gt_plot')
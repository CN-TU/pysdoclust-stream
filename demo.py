from SDOclustream import clustering
import numpy as np
import pandas as pd

df = pd.read_csv('example/dataset.csv')
t = df['timestamp'].to_numpy()
x = df[['f0','f1']].to_numpy()
y = df['label'].to_numpy()

k = 200 # Model size
T = 400 # Time Horizon
ibuff = 10 # input buffer
classifier = clustering.SDOclustream(k=k, T=T, input_buffer=ibuff)

all_predic = []
all_scores = []

block_size = 1 # per-point processing
for i in range(0, x.shape[0], block_size):
    chunk = x[i:i + block_size, :]
    chunk_time = t[i:i + block_size]
    labels, outlier_scores = classifier.fit_predict(chunk, chunk_time)
    all_predic.append(labels)
    all_scores.append(outlier_scores)
p = np.concatenate(all_predic) # clustering labels
s = np.concatenate(all_scores) # outlierness scores

# Thresholding top outliers based on Chebyshev's inequality (88.9%)
th = np.mean(s)+3*np.std(s)
p[s>th]=-1

# Evaluation metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import roc_auc_score
print("Adjusted Rand Index (clustering):", adjusted_rand_score(y,p))
print("ROC AUC score (outlier/anomaly detection):", roc_auc_score(y<0,s))

 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
p = LabelEncoder().fit_transform(p) -1

fig = plt.figure(figsize=(15,4))
cmap = plt.get_cmap('tab20', len(np.unique(p)))

for i in range(3):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    ax.scatter3D(t[p>-1], x[p>-1,0], x[p>-1,1], s=5, c=p[p>-1], cmap=cmap)
    ax.scatter3D(t[p==-1], x[p==-1,0], x[p==-1,1], s=5, c='black')
    ax.view_init(azim=280+i*30, elev=20)
    ax.set_xlabel('time')
    ax.set_ylabel('f0')
    ax.set_zlabel('f1')

plt.show()

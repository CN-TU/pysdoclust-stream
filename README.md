# pysdoclust-stream

**SDOstreamclust** 

Incremental stream clustering (and outlier detection) algorithm based on Sparse Data Observers (SDO). 

SDOstreamclust is suitable for large, multi-dimensional datasets where clusters are statistically well represented.

<br>

## Dependencies

SDOstreamclust requires **numpy**.

<br>

## Installation

SDOstreamclust can be installed from the **main branch**:

        pip3 install git+https://github.com/CN-TU/pysdoclust-stream.git@main

or simply download the **main branch** and run:

        pip3 install pysdoclust-stream-main.zip  

<br>

## Folder Structure and Evaluation Experiments

The [cpp] folder contains the code for the C++ core algorithms, which might be used directly in C++ projects. 

When using SDOstreamclust from Python, the C++ algorithms are wrapped by the interfaces in the [swig] folder. These wrapper functions are translated to a Python interface and have the main purpose of providing an interface which can easily be parsed by SWIG.

The [python] folder contains the Python interface invoking the Python interface provided by SWIG.

Finally, complete experiments, datasets, scripts and results conducted for the paper **Stream Clustering Robust to Concept Drift** are provided in the **[evaluation_tests] folder**. They have been tested with **Python v3.8.14**. 

A Docker version is also available in: [https://hub.docker.com/r/fiv5/sdostreamclust](https://hub.docker.com/r/fiv5/sdostreamclust) 
 

<br>

## Example

SDOstreamclust is a straighforward algorithm and very easy to configure. The main parameters are the number of observers `k`, which determines the size of the model and the parameter `T`, which defines the memory of the algorithm. 

Setting the right `k` (default=300) depends on the variability of the data and the expected number of clusters, but this is quite a robust parameter that gives proper performances with values between [200,500] in most scenarios. On the other hand, `T` (default=500) sets the model dynamics and inertia. Intuitively, it is the number of points processed that results in a fully replaced model (on average). Low `T` is recommended when the data show very fast dynamics, while if data evolution is slow and retaining old clusters is dedired, `T` should be set with high values.

Additionally, `input_buffer` (default=0) establishes how many points are necessary for the observers to update the internal clustering. This fundamentally affects the processing speed. Most scenarios commonly tolerate high values in the `input_buffer` without significantly affecting the accuracy performance. Beyond the mentioned ones, other parameters are inherited from SDOclust and SDOstream and do not usually require adjustment. They are described in *python/clustering.py* file.

The following example code retrieves a data stream and initialize SDOstreamclust.

```python
from SDOstreamclust import clustering
import numpy as np
import pandas as pd

df = pd.read_csv('example/dataset.csv')
t = df['timestamp'].to_numpy()
x = df[['f0','f1']].to_numpy()
y = df['label'].to_numpy()

k = 200 # Model size
T = 400 # Time Horizon
ibuff = 10 # input buffer
classifier = clustering.SDOstreamclust(k=k, T=T, input_buffer=ibuff)
```

In the piece of code below the stream data is processed point by point. SDOstreamclust provides a clustering label and an outlierness score per point. It can also perform outlier thresholding internally by giving the label *-1* to outliers. To do this, ``outlier_handling=True`` must be set and the ``outlier_threshold`` (default=5) adjusted.


```python
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
s = -1/(s+1) # norm. to avoid inf scores

# Thresholding top outliers based on Chebyshev's inequality (88.9%)
th = np.mean(s)+3*np.std(s)
p[s>th]=-1

# Evaluation metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import roc_auc_score
print("Adjusted Rand Index (clustering):", adjusted_rand_score(y,p))
print("ROC AUC score (outlier/anomaly detection):", roc_auc_score(y<0,s))
```

Giving *ARI=0.97* and *ROC-AUC=0.99*. Note how SDOstreamclust assigns high outlierness scores to the first points of emerging clusters.

![](example/example_dataset.png)


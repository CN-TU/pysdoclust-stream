# pysdoclust-stream
incremental stream clustering algorithm based on SDO

## Usage in python

```python
from SDOclustream import clustering
import numpy as np

size = 500
X = np.random.rand(size, 3)

k = 50 # Model size
T = 150 # Time Horizon
classifier = clustering.SDOclustream(k=k, T=T)

# Initialize an array to store the labels
all_labels = []
block_size = 25
# Iterate through the data in blocks
for i in range(0, X.shape[0], block_size):
    chunk = X[i:i + block_size, :]
    labels = classifier.fit_predict(chunk)
    all_labels.append(labels)
final_labels = np.concatenate(all_labels)

# print model
obs = classifier.get_observers()
for o in obs:
      print(o)

# print labels
for l in all_labels:
      print(l)
```

## Rebuilding

When adding new algorithms or modifying the interface, the SWIG wrappers have to be rebuilt. To this end, SWIG has to be installed and a ``pip`` package can be created and installed  using

```make && pip3 install SDOclustream.tar.xz```

# pysdoclust-stream
incremental stream clustering algorithm based on SDO

## Installation

```
pip3 install git+https://github.com/CN-TU/pysdoclust-stream@clean
```

## Usage in python



```python
from SDOclustream import clustering
import numpy as np

size = 500
X = np.random.rand(size, 3)
times = np.sort(np.random.rand(size)) * size # random timestamp from 0 to size-1
# times = np.arange(0,size) # timestamp ordered from 0 to size-1

k = 50 # Model size
T = 200 # Time Horizon
classifier = clustering.SDOclustream(k=k, T=T)

# Initialize an array to store the labels
all_labels = []
all_scores = []
block_size = 20
# Iterate through the data in blocks
for i in range(0, X.shape[0], block_size):
    chunk = X[i:i + block_size, :]
    chunk_time = times[i:i + block_size]
    labels, outlier_scores = classifier.fit_predict(chunk, chunk_time)
    all_labels.append(labels)
    all_scores.append(outlier_scores)
final_labels = np.concatenate(all_labels)
final_scores = np.concatenate(all_scores)

# print model
obs = classifier.get_observers()
for o in obs:
      print(o)

# print labels
for l in all_labels:
      print(l)
```

## Architecture

The cpp folder contains the code for the C++ core algorithms, which might be used directly by C++ projects. 

When using SDOclustream from Python, the C++ algorithms are wrapped by the interfaces in the SWIG folder. These wrapper functions are translated to a Python interface and have the main purpose of providing an interface which can easily be parsed by SWIG.

Finally, the python folder contains the Python interface invoking the Python interface provided by SWIG.

## Rebuilding

When adding new algorithms or modifying the interface, the SWIG wrappers have to be rebuilt. To this end, SWIG has to be installed and a ``pip`` package can be created and installed  using

```make && pip3 install SDOclustream.tar.xz```

## Aknowledgments

I would like to thank the developers of the [dSalmon](https://github.com/CN-TU/dSalmon) project for providing the framework and algorithms, in particular the MTree implementation, that were instrumental in the development of this project.

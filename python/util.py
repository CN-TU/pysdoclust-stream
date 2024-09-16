import numpy as np
from SDOstreamclust import swig

def sanitizeData(data, float_type=np.float64):
    if not (isinstance(data, np.ndarray) and data.dtype==float_type and data.flags['C_CONTIGUOUS']):
        data = np.array(data, dtype=float_type, order='C')
    if len(data.shape) == 1:
        data = data[None,:]
    assert len(data.shape) == 2
    # Avoid occupying twice the memory due to element-wise comparison
    # when large chunks are passed.
    flattened = data.reshape(-1)
    for i in range(0,flattened.size,100000):
        assert not np.isnan(flattened[i:i+100000]).any(), 'NaN values are not allowed'
    return data

def sanitizeTimes(times, data_len, last_time, float_type=np.float64):
    if times is None:
        times = np.arange(last_time + 1, last_time + 1 + data_len, dtype=float_type)
    else:
        if not (isinstance(times, np.ndarray) and times.dtype==float_type and times.flags['C_CONTIGUOUS']):
            times = np.array(times, dtype=float_type, order='C')
        assert len(times.shape) <= 1
        if len(times.shape) == 0:
            times = np.repeat(times[None], data_len)
        else:
            assert times.shape[0] == data_len
    return times

def lookupDistance(name, float_type, **kwargs):
    wrappers = {
        'chebyshev': 'ChebyshevDist',
        'cityblock': 'ManhattanDist',
        'euclidean': 'EuclideanDist',
    }
    suffix = {np.float32: '32', np.float64: '64'}[float_type]

    if name in wrappers:
        return swig.__dict__[wrappers[name] + suffix]()
    elif name == 'minkowski':
        if not 'p' in kwargs:
            raise TypeError('p is required for Minkowski distance')
        return swig.__dict__['MinkowskiDist' + suffix](kwargs['p'])
    else:
        raise TypeError('Unknown metric')
    
    
class Buffer:
    """
    A class to buffer data streams in batches (NumPy arrays).
    """
    def __init__(self, input_buffer, nfeatures, dtype):
        self.buffer = {'data': np.empty((0, nfeatures), dtype=dtype),  # Empty array with room for buffer size
                       'times': np.empty((0, ), dtype=dtype)}  # Empty array with room for buffer size
        self.input_buffer = input_buffer
        self.nfeatures = nfeatures
        self.dtype = dtype

    def add_batch(self, data, times, last=False):
        """
        Adds a batch of data points to the buffer and returns remaining data.

        Args:
          data: A NumPy array of shape (batch_size, n_features).

        Returns:
          A NumPy array of remaining data points that couldn't fit in the buffer.
        """

        # Add data to the buffer
        self.buffer['data'] = np.vstack((self.buffer['data'],  data))
        self.buffer['times'] = np.hstack((self.buffer['times'], times))

        # Check if buffer needs flushing
        return self.flush(last=last)

    def flush(self, last=False):
        """
        Calls the provided program function with the buffered data and clears the buffer.

        Returns:
          data: A NumPy array of flushed data points.
          times: A NumPy array of corresponding times.
        """
        if self.buffer['data'].shape[0] >= self.input_buffer:
            data = self.buffer['data'][:self.input_buffer]  # Slice to get requested elements
            times = self.buffer['times'][:self.input_buffer]  # Slice to get requested elements
            self.buffer['data'] = self.buffer['data'][self.input_buffer:]  # Remove returned elements
            self.buffer['times'] = self.buffer['times'][self.input_buffer:]  # Remove returned elements
            return data, times
        
        if last and self.buffer['data'].shape[0] > 0:
            data = self.buffer['data'][:self.input_buffer]  # Slice to get requested elements
            times = self.buffer['times'][:self.input_buffer]  # Slice to get requested elements
            self.buffer = {'data': np.empty((0, self.nfeatures), dtype=self.dtype),  # Empty array with room for buffer size
                       'times': np.empty((0, ), dtype=self.dtype)}  # Empty array with room for buffer size
            return data, times
        
        return np.empty((0, self.nfeatures), dtype=self.dtype), np.empty((0, ), dtype=self.dtype)  # Empty buffer if no data
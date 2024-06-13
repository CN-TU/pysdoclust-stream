"""
Streaming clustering models.
"""

import numpy as np
from SDOstreamclust import swig as SDOstreamclust_cpp
# from dSalmon import projection
from SDOstreamclust.util import sanitizeData, sanitizeTimes, lookupDistance

class Clustering(object):
    """
    Base class for streaming clustering models.
    """

    def _init_model(self, p):
        pass

    def get_params(self, deep=True):
        """
        Return the used algorithm parameters as a dictionary.

        Parameters
        ----------
        deep: bool, default=True
            Ignored. Only for compatibility with scikit-learn.

        Returns
        -------
        params: dict
            Dictionary of parameters.
        """
        return self.params

    def set_params(self, **params):
        """
        Reset the model and set the parameters according to the
        supplied dictionary.

        Parameters
        ----------
        **params: dict
            Dictionary of parameters.
        """
        p = self.params.copy()
        for key in params:
            assert key in p, 'Unknown parameter: %s' % key
            p[key] = params[key]
        self._init_model(p)

    def fit_predict(self, X, times=None):
        """
        Process the next chunk of data and perform clustering.

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.

        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample.
        """
        # In most cases, fitting isn't any faster than additionally
        # performing outlier scoring. We override this method only
        # when it yields faster processing.
        self.fit_predict(X, times)

    def _process_data(self, data):
        data = sanitizeData(data, self.params['float_type'])
        assert self.dimension == -1 or data.shape[1] == self.dimension
        self.dimension = data.shape[1]
        return data

    def _process_times(self, data, times):
        times = sanitizeTimes(times, data.shape[0], self.last_time, self.params['float_type'])
        self.last_time = times[-1]
        return times

class SDOstreamclust(Clustering):
    """
    Streaming clustering based on Sparse Data Observers :cite:p:`Hartl2019`.

    Parameters
    ----------
    k: int (default=300)
        Number of observers to use.

    T: int (default=500)
        Characteristic time for the model.
        Increasing T makes the model adjust slower, decreasing T
        makes it adjust quicker.

    qv: float, optional (default=0.3)
        Ratio of unused observers due to model cleaning.

    x: int (default=5)
        Number of nearest observers to consider for clustering.

    metric: string (default='euclidean')
        Which distance metric to use. Currently supported metrics
        include 'chebyshev', 'cityblock', 'euclidean', and 'minkowsi'.

    metric_params: dict
        Parameters passed to the metric. Minkowsi distance requires
        setting an integer `p` parameter.

    float_type: np.float32 or np.float64 (default=np.float64)
        The floating point type to use for internal processing.

    zeta: float, optional (default=0.6)
        Determines ratio between local h and global h that determines h for each Observer.

    chi_min: int, optional (default=8)
        Minimum amount of Observers that determine h (closeness parameter) for each Observer.

    chi_prop: float, optional (default=0.05)
        Parameter to determine closeness parameter of an Observer. The chi_prop * Modelsize Observers of an Observers are "close".

    e: int, optional (default=2)
        Minimum size of a cluster (number of Observers spanning / representing it)

    freq_bins: int, optional (default=1)
        Number of bins when using temporal SDO model. If 1 "normal" SDO model is used.

    max_freq: float, optional (default=1.0)
        Temportal frequency when using temporal SDO model

    outlier_handling (default=False)
        Outlier handling activation flag.

    rel_outlier_score (default=True)
        Give outlier score either as absolute distance or relative distance with regard to h_bar. 
        Median value of x closest active observers.

    outlier_threshold: float, optional (default=5.0)
        Threshold for outlier handling. 
        If point has distance = outlier_threshold * h_bar to a (closest) observer probability of being an outlier wrt to this Observer is 0.5 
        If distance is <= h_bar then probability is 0. Calibrated on an activation function (tangens hyperbolicus).

    perturb: float, optional (default=0.0)
        Perturbation parameter to differentiate between equal points. 
        Recommended to use a value smaller than an expected small distance between two close points.

    random_sampling: float, optional (default=True)
        Flag to decide if Random Sampling is used.
        Recommneded to set to True.

    input_buffer: int, optional (default=0)
        Batch size that is actually processed. If smaller batch is given algorithm waits to process.
        If larger batch is given, batch is split into pieces of size input_buffer.

    seed: int (default=0)
        Random seed to use.
    """
    def __init__(self, k=300, T=500, qv=0.3, x=5, metric='euclidean', metric_params=None,
                 float_type=np.float64, seed=0, return_sampling=False, zeta=0.6, chi_min=8, 
                 chi_prop=0.05, e=2, outlier_threshold=5.0, outlier_handling=False, rel_outlier_score=True, 
                 perturb=0.0, random_sampling=True, freq_bins=1, max_freq=1.0, input_buffer=0):
        self.params = {k: v for k, v in locals().items() if k != 'self'}
        self._init_model(self.params)

    def _init_model(self, p):
        assert p['float_type'] in [np.float32, np.float64]
        assert 0 <= p['qv'] < 1, 'qv must be in [0,1)'
        assert p['x'] > 0, 'x must be > 0'
        assert p['k'] > 0, 'k must be > 0'
        assert p['T'] > 0, 'T must be > 0'
        assert 0 <= p['zeta'] < 1, 'zeta must be in [0,1)'
        assert p['chi_min'] > 0, 'chi_min must be > 0'
        assert 0 <= p['chi_prop'] < 1, 'chi_prop must be in [0,1)'
        assert p['e'] > 0, 'e must be > 0'
        assert 1 <= p['freq_bins'], 'freq_bins must be in (1,inf)'
        assert 0 < p['max_freq'], 'max_freq must be in (0, inf)'     
        assert p['outlier_handling'] in [True, False]    
        assert p['rel_outlier_score'] in [True, False]
        assert 1 < p['outlier_threshold'], 'outlier_threshold must be in (1,inf)'
        assert 0 <= p['perturb'], 'perturb shall be small, so 1e-7 or something'
        assert p['random_sampling'] in [True, False]
        assert p['input_buffer'] >= 0

        # Map the Python metric name to the C++ distance function
        distance_function = lookupDistance(p['metric'], p['float_type'], **(p['metric_params'] or {}))
        
        # Create an instance of the C++ SDOcluststream class
        if p['freq_bins']>1:
            cpp_obj = {
                np.float32: SDOstreamclust_cpp.tpSDOstreamclust32,
                np.float64: SDOstreamclust_cpp.tpSDOstreamclust64
            }[p['float_type']]
        else:
            cpp_obj = {
                np.float32: SDOstreamclust_cpp.SDOstreamclust32,
                np.float64: SDOstreamclust_cpp.SDOstreamclust64
            }[p['float_type']]
        self.model = cpp_obj(
            p['k'], 
            p['T'], 
            p['qv'], 
            p['x'], 
            p['chi_min'], 
            p['chi_prop'], 
            p['zeta'],
            p['e'], 
            p['freq_bins'], 
            p['max_freq'], 
            p['outlier_threshold'], 
            p['outlier_handling'],
            p['rel_outlier_score'],
            p['perturb'], 
            p['random_sampling'],
            p['input_buffer'],
            distance_function, 
            p['seed']
        )
        self.last_time = 0
        self.dimension = -1

    def fit_predict(self, X, times=None, last=False):
        """
        Process next chunk of data.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        
        Returns    
        -------
        y: ndarray, shape (n_samples,)
            Labels for provided input data.
        """
        X = self._process_data(X)
        times = self._process_times(X, times)        
        labels = np.empty(X.shape[0], dtype=np.int32)   
        scores = np.empty(X.shape[0], dtype=self.params['float_type'])
        # if self.params['return_sampling']:
        #     sampling = np.empty(X.shape[0], dtype=np.int32)
        #     self.model.fit_predict_with_sampling(X, labels, times, sampling)
        #     return labels, sampling
        # else:
        #     self.model.fit_predict(X, labels, times)
        #     # self.model.fit_predict_batch(X, labels, times)
        #     return labels
        
        self.model.fit_predict(X, labels, scores, times)
        return labels, scores

    def observer_count(self):
        """Return the current number of observers."""
        return self.model.observer_count()
        
    def get_observers(self, time=None):
        """
        Return observer data.
        
        Returns    
        -------
        data: ndarray, shape (n_observers, n_features)
            Sample used as observer.
            
        observations: ndarray, shape (n_observers,)
            Exponential moving average of observations.
            
        av_observations: ndarray, shape (n_observers,)
            Exponential moving average of observations
            normalized according to the theoretical maximum.

        labels: ndarray, shape (n_observers,)
            Labels / colors of observer.
        """
        if time is None:
            time = self.last_time
        observer_cnt = self.model.observer_count()
        if observer_cnt == 0:
            return np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=np.int32), np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=self.params['float_type'])
        data = np.empty([observer_cnt, self.dimension], dtype=self.params['float_type'])
        observations = np.empty(observer_cnt, dtype=self.params['float_type'])
        av_observations = np.empty(observer_cnt, dtype=self.params['float_type'])
        labels = np.empty(observer_cnt, dtype=np.int32)
        self.model.get_observers(data, labels, observations, av_observations, self.params['float_type'](time))
        return data, labels, observations, av_observations


class tpSDOstreamclust(Clustering):
    """
    Streaming clustering based on Sparse Data Observers :cite:p:`Hartl2019`.

    Parameters
    ----------
    k: int (default=300)
        Number of observers to use.

    T: int (default=500)
        Characteristic time for the model.
        Increasing T makes the model adjust slower, decreasing T
        makes it adjust quicker.

    qv: float, optional (default=0.3)
        Ratio of unused observers due to model cleaning.

    x: int (default=5)
        Number of nearest observers to consider for clustering.

    metric: string (default='euclidean')
        Which distance metric to use. Currently supported metrics
        include 'chebyshev', 'cityblock', 'euclidean', and 'minkowsi'.

    metric_params: dict
        Parameters passed to the metric. Minkowsi distance requires
        setting an integer `p` parameter.

    float_type: np.float32 or np.float64 (default=np.float64)
        The floating point type to use for internal processing.

    zeta: float, optional (default=0.6)
        Determines ratio between local h and global h that determines h for each Observer.

    chi_min: int, optional (default=8)
        Minimum amount of Observers that determine h (closeness parameter) for each Observer.

    chi_prop: float, optional (default=0.05)
        Parameter to determine closeness parameter of an Observer. The chi_prop * Modelsize Observers of an Observers are "close".

    e: int, optional (default=2)
        Minimum size of a cluster (number of Observers spanning / representing it)

    freq_bins: int, optional (default=1)
        Number of bins when using temporal SDO model. If 1 "normal" SDO model is used.

    max_freq: float, optional (default=1.0)
        Temportal frequency when using temporal SDO model

    outlier_handling (default=False)
        Outlier handling activation flag.

    rel_outlier_score (default=True)
        Give outlier score either as absolute distance or relative distance with regard to h_bar. 
        Median value of x closest active observers.

    outlier_threshold: float, optional (default=5.0)
        Threshold for outlier handling. 
        If point has distance = outlier_threshold * h_bar to a (closest) observer probability of being an outlier wrt to this Observer is 0.5 
        If distance is <= h_bar then probability is 0. Calibrated on an activation function (tangens hyperbolicus).

    perturb: float, optional (default=0.0)
        Perturbation parameter to differentiate between equal points. 
        Recommended to use a value smaller than an expected small distance between two close points.

    random_sampling: float, optional (default=True)
        Flag to decide if Random Sampling is used.
        Recommneded to set to True.

    input_buffer: int, optional (default=0)
        Batch size that is actually processed. If smaller batch is given algorithm waits to process.
        If larger batch is given, batch is split into pieces of size input_buffer.

    seed: int (default=0)
        Random seed to use.
    """
    def __init__(self, k=300, T=500, qv=0.3, x=5, metric='euclidean', metric_params=None,
                 float_type=np.float64, seed=0, return_sampling=False, zeta=0.6, chi_min=8, 
                 chi_prop=0.05, e=2, outlier_threshold=5.0, outlier_handling=False, rel_outlier_score=True, 
                 perturb=0.0, random_sampling=True, freq_bins=1, max_freq=1.0, input_buffer=0):
        self.params = {k: v for k, v in locals().items() if k != 'self'}
        self._init_model(self.params)

    def _init_model(self, p):
        assert p['float_type'] in [np.float32, np.float64]
        assert 0 <= p['qv'] < 1, 'qv must be in [0,1)'
        assert p['x'] > 0, 'x must be > 0'
        assert p['k'] > 0, 'k must be > 0'
        assert p['T'] > 0, 'T must be > 0'
        assert 0 <= p['zeta'] < 1, 'zeta must be in [0,1)'
        assert p['chi_min'] > 0, 'chi_min must be > 0'
        assert 0 <= p['chi_prop'] < 1, 'chi_prop must be in [0,1)'
        assert p['e'] > 0, 'e must be > 0'
        assert 1 <= p['freq_bins'], 'freq_bins must be in (1,inf)'
        assert 0 < p['max_freq'], 'max_freq must be in (0, inf)'        
        assert p['outlier_handling'] in [True, False]     
        assert p['rel_outlier_score'] in [True, False]
        assert 1 < p['outlier_threshold'], 'outlier_threshold must be in (1,inf)'
        assert 0 <= p['perturb'], 'perturb shall be small, so 1e-7 or something'
        assert p['random_sampling'] in [True, False]

        # Map the Python metric name to the C++ distance function
        distance_function = lookupDistance(p['metric'], p['float_type'], **(p['metric_params'] or {}))
        
        # Create an instance of the C++ SDOcluststream class
        cpp_obj = {
            np.float32: SDOstreamclust_cpp.tpSDOstreamclust32,
            np.float64: SDOstreamclust_cpp.tpSDOstreamclust64
        }[p['float_type']]
        
        self.model = cpp_obj(
            p['k'], 
            p['T'], 
            p['qv'], 
            p['x'], 
            p['chi_min'], 
            p['chi_prop'], 
            p['zeta'],
            p['e'], 
            p['freq_bins'], 
            p['max_freq'], 
            p['outlier_threshold'], 
            p['outlier_handling'],
            p['rel_outlier_score'],
            p['perturb'], 
            p['random_sampling'],
            distance_function, 
            p['seed']
        )
        
        self.last_time = 0
        self.dimension = -1

    def fit_predict(self, X, times=None):
        """
        Process next chunk of data.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        
        Returns    
        -------
        y: ndarray, shape (n_samples,)
            Labels for provided input data.
        """
        X = self._process_data(X)
        times = self._process_times(X, times)        
        labels = np.empty(X.shape[0], dtype=np.int32)   
        scores = np.empty(X.shape[0], dtype=self.params['float_type'])
        # if self.params['return_sampling']:
        #     sampling = np.empty(X.shape[0], dtype=np.int32)
        #     self.model.fit_predict_with_sampling(X, labels, times, sampling)
        #     return labels, sampling
        # else:
        #     self.model.fit_predict(X, labels, times)
        #     # self.model.fit_predict_batch(X, labels, times)
        #     return labels
        
        self.model.fit_predict(X, labels, scores, times)
        return labels, scores

    def observer_count(self):
        """Return the current number of observers."""
        return self.model.observer_count()
        
    def get_observers(self, time=None):
        """
        Return observer data.
        
        Returns    
        -------
        data: ndarray, shape (n_observers, n_features)
            Sample used as observer.
            
        observations: ndarray, shape (n_observers,)
            Exponential moving average of observations.
            
        av_observations: ndarray, shape (n_observers,)
            Exponential moving average of observations
            normalized according to the theoretical maximum.

        labels: ndarray, shape (n_observers,)
            Labels / colors of observer.
        """
        if time is None:
            time = self.last_time
        observer_cnt = self.model.observer_count()
        if observer_cnt == 0:
            return np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=np.int32), np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=self.params['float_type'])
        data = np.empty([observer_cnt, self.dimension], dtype=self.params['float_type'])
        observations = np.empty(observer_cnt, dtype=self.params['float_type'])
        av_observations = np.empty(observer_cnt, dtype=self.params['float_type'])
        labels = np.empty(observer_cnt, dtype=np.int32)
        self.model.get_observers(data, labels, observations, av_observations, self.params['float_type'](time))
        return data, labels, observations, av_observations
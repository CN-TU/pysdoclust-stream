#ifndef DSALMON_CLUSTERING_WRAPPER_H
#define DSALMON_CLUSTERING_WRAPPER_H

#include "SDOstreamclust.h"
#include "tpSDOstreamclust.h"
// #include "histogram.h"

#include "array_types.h"
#include "distance_wrappers.h"

template<typename FloatType>
class SDOstreamclust_wrapper {
    int dimension;
    // std::size_t freq_bins;
    SDOstreamclust<FloatType> sdoclust; // Use SDOstreamclust

  public:
    SDOstreamclust_wrapper(
      int observer_cnt, 
      FloatType T, 
      FloatType idle_observers, 
      int neighbour_cnt, 
      int chi_min, 
      FloatType chi_prop,
      FloatType zeta, 
      int e, 
      int freq_bins,
      FloatType max_freq, 
      FloatType outlier_threshold, 
      bool outlier_handling,  
      bool rel_outlier_score,
      FloatType perturb,
      bool random_sampling,
      int input_buffer,
      Distance_wrapper<FloatType>* distance, 
      int seed);

    void fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times);
    void fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<int>  labels, NumpyArray1<FloatType>  scores, const NumpyArray1<FloatType> times);
    // void fit_predict_with_sampling(const NumpyArray2<FloatType> data, NumpyArray1<int> labels, const NumpyArray1<FloatType> times, NumpyArray1<int> sampled);
    int observer_count();
    void get_observers(NumpyArray2<FloatType> data, NumpyArray1<int> labels, NumpyArray1<FloatType> observations, NumpyArray1<FloatType> av_observations, FloatType time);

};

// Instantiate the class for different floating-point types
DEFINE_FLOATINSTANTIATIONS(SDOstreamclust)

template<typename FloatType>
class tpSDOstreamclust_wrapper {
    int dimension;
    // std::size_t freq_bins;
    tpSDOstreamclust<FloatType> sdoclust; // Use SDOstreamclust

  public:
    tpSDOstreamclust_wrapper(
      int observer_cnt, 
      FloatType T, 
      FloatType idle_observers, 
      int neighbour_cnt, 
      int chi_min, 
      FloatType chi_prop,
      FloatType zeta, 
      int e,
      int freq_bins,
      FloatType max_freq,    
      FloatType outlier_threshold, 
      bool outlier_handling,  
      bool rel_outlier_score,
      FloatType perturb,
      bool random_sampling,
      int input_buffer,
      Distance_wrapper<FloatType>* distance, 
      int seed);

    void fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times);
    void fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<int>  labels, NumpyArray1<FloatType>  scores, const NumpyArray1<FloatType> times);
    // void fit_predict_with_sampling(const NumpyArray2<FloatType> data, NumpyArray1<int> labels, const NumpyArray1<FloatType> times, NumpyArray1<int> sampled);
    int observer_count();
    void get_observers(NumpyArray2<FloatType> data, NumpyArray1<int> labels, NumpyArray1<FloatType> observations, NumpyArray1<FloatType> av_observations, FloatType time);

};

// Instantiate the class for different floating-point types
DEFINE_FLOATINSTANTIATIONS(tpSDOstreamclust)

#endif

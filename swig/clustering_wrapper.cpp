#include <algorithm>
#include <vector>
// #include <assert.h>
// #include <random>
#include <atomic>
#include <thread>

#include "clustering_wrapper.h"

template<typename FloatType>
static void fit_predict_ensemble(unsigned ensemble_size, int n_jobs, std::function<void(int)> worker) {
    std::atomic<unsigned> global_i(0);
    auto thread_worker = [&]() {
        for (unsigned tree_index = global_i++; tree_index < ensemble_size; tree_index = global_i++) {
            worker(tree_index);
        }
    };
    if (n_jobs < 2) {
        thread_worker();
    }
    else {
        std::thread threads[n_jobs];
        for (int i = 0; i < n_jobs; i++)
            threads[i] = std::thread{thread_worker};
        for (int i = 0; i < n_jobs; i++)
            threads[i].join();
    }
}

template<typename FloatType>
SDOstreamclust_wrapper<FloatType>::SDOstreamclust_wrapper(
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
        int seed) :
    dimension(-1),
    sdoclust(
        observer_cnt, 
        T, 
        idle_observers, 
        neighbour_cnt, 
        chi_min, 
        chi_prop, 
        zeta, 
        e, 
        outlier_threshold, 
        outlier_handling,
        rel_outlier_score,
        perturb, 
        random_sampling,
        input_buffer,
        distance->getFunction(), 
        seed)
{
}

template<typename FloatType>
void SDOstreamclust_wrapper<FloatType>::fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times) {
    std::vector<FloatType> vec_times(&times.data[0], &times.data[0] + times.dim1);
    std::vector<Vector<FloatType>> vec_data(data.dim1, Vector<FloatType>(data.dim2));    
    for (int i = 0; i < data.dim1; i++) {
        vec_data[i] = Vector<FloatType>(&data.data[i * data.dim2], data.dim2);
    }
    sdoclust.fit(vec_data, vec_times);  
}

template<typename FloatType>
void SDOstreamclust_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<int> labels, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times) {
    std::vector<FloatType> vec_times(&times.data[0], &times.data[0] + times.dim1);
    std::vector<Vector<FloatType>> vec_data(data.dim1, Vector<FloatType>(data.dim2));    
    for (int i = 0; i < data.dim1; i++) {
        vec_data[i] = Vector<FloatType>(&data.data[i * data.dim2], data.dim2);
    }
    std::vector<int> vec_label(data.dim1, 0);
    std::vector<FloatType> vec_score(data.dim1, FloatType(0));
    sdoclust.fitPredict(vec_label, vec_score, vec_data, vec_times);  
    std::move(vec_label.begin(), vec_label.end(), &labels.data[0]);  
    std::move(vec_score.begin(), vec_score.end(), &scores.data[0]);  
}

template<typename FloatType>
int SDOstreamclust_wrapper<FloatType>::observer_count() {
    return sdoclust.observerCount();
}

template<typename FloatType>
void SDOstreamclust_wrapper<FloatType>::get_observers(NumpyArray2<FloatType> data, NumpyArray1<int> labels, NumpyArray1<FloatType> observations, NumpyArray1<FloatType> av_observations, FloatType time) {
    // TODO: check dimensions
    int i = 0;
    for (auto observer : sdoclust) {
        Vector<FloatType> vec_data = observer.getData();
        std::copy(vec_data.begin(), vec_data.end(), &data.data[i * data.dim2]);
        observations.data[i] = observer.getObservations(time);
        av_observations.data[i] = observer.getAvObservations(time);
        labels.data[i] = observer.getColor();
        i++;
    }
}

template class SDOstreamclust_wrapper<double>;
template class SDOstreamclust_wrapper<float>;

// tpSDO

template<typename FloatType>
tpSDOstreamclust_wrapper<FloatType>::tpSDOstreamclust_wrapper(
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
        int seed) :
    dimension(-1),
    sdoclust(
        observer_cnt, 
        T, 
        idle_observers, 
        neighbour_cnt, 
        chi_min, 
        chi_prop, 
        zeta, 
        e, 
        freq_bins, 
        max_freq, 
        outlier_threshold, 
        outlier_handling,
        rel_outlier_score,
        perturb, 
        random_sampling,
        input_buffer,
        distance->getFunction(), 
        seed)
{
}

template<typename FloatType>
void tpSDOstreamclust_wrapper<FloatType>::fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times) {
    std::vector<FloatType> vec_times(&times.data[0], &times.data[0] + times.dim1);
    std::vector<Vector<FloatType>> vec_data(data.dim1, Vector<FloatType>(data.dim2));    
    for (int i = 0; i < data.dim1; i++) {
        vec_data[i] = Vector<FloatType>(&data.data[i * data.dim2], data.dim2);
    }
    sdoclust.fit(vec_data, vec_times);  
}

template<typename FloatType>
void tpSDOstreamclust_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<int> labels, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times) {
    std::vector<FloatType> vec_times(&times.data[0], &times.data[0] + times.dim1);
    std::vector<Vector<FloatType>> vec_data(data.dim1, Vector<FloatType>(data.dim2));    
    for (int i = 0; i < data.dim1; i++) {
        vec_data[i] = Vector<FloatType>(&data.data[i * data.dim2], data.dim2);
    }
    std::vector<int> vec_label(data.dim1, 0);
    std::vector<FloatType> vec_score(data.dim1, FloatType(0));
    sdoclust.fitPredict(vec_label, vec_score, vec_data, vec_times);  
    std::copy(vec_label.begin(), vec_label.end(), &labels.data[0]);  
    std::copy(vec_score.begin(), vec_score.end(), &scores.data[0]);  
}

template<typename FloatType>
int tpSDOstreamclust_wrapper<FloatType>::observer_count() {
    return sdoclust.observerCount();
}

template<typename FloatType>
void tpSDOstreamclust_wrapper<FloatType>::get_observers(NumpyArray2<FloatType> data, NumpyArray1<int> labels, NumpyArray1<FloatType> observations, NumpyArray1<FloatType> av_observations, FloatType time) {
    // TODO: check dimensions
    int i = 0;
    for (auto observer : sdoclust) {
        Vector<FloatType> vec_data = observer.getData();
        std::copy(vec_data.begin(), vec_data.end(), &data.data[i * data.dim2]);
        observations.data[i] = observer.getObservations(time);
        av_observations.data[i] = observer.getAvObservations(time);
        labels.data[i] = observer.getColor();
        i++;
    }
}

template class tpSDOstreamclust_wrapper<double>;
template class tpSDOstreamclust_wrapper<float>;


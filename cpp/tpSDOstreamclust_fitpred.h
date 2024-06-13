#ifndef TPSDOSTREAMCLUST_FITPRED_H
#define TPSDOSTREAMCLUST_FITPRED_H

// #include "tpSDOstreamclust_print.h"
#include "tpSDOstreamclust_util.h"
#include "tpSDOstreamclust_sample.h"
#include "tpSDOstreamclust_fit.h"
#include "tpSDOstreamclust_predict.h"

template<typename FloatType>
void tpSDOstreamclust<FloatType>::fitPredict_impl(
        std::vector<int>& label,
        std::vector<FloatType>& score,
        const std::vector<Vector<FloatType>>& data, 
        const std::vector<FloatType>& epsilon,
        const std::vector<FloatType>& time_data) {
    // Check for equal lengths:
    if (data.size() != time_data.size()) {
        throw std::invalid_argument("data and now must have the same length");
    }
    int first_index = last_index;
    // sample data
    std::unordered_set<int> sampled;   
    sample(sampled, data, epsilon, time_data, first_index);
    // fit model 
    fit_impl(data, epsilon, time_data, first_index);    
    // update graph
    update(time_data, sampled);
    // predict
    predict_impl(label, score, data, epsilon, first_index);
}

template<typename FloatType>
void tpSDOstreamclust<FloatType>::fitOnly_impl(
        const std::vector<Vector<FloatType>>& data, 
        const std::vector<FloatType>& epsilon,
        const std::vector<FloatType>& time_data) {
    // Check for equal lengths:
    if (data.size() != time_data.size()) {
        throw std::invalid_argument("data and now must have the same length");
    }
    int first_index = last_index;
    // sample data
    std::unordered_set<int> sampled;   
    sample(sampled, data, epsilon, time_data, first_index);
    // fit model 
    fit_impl(data, epsilon, time_data, first_index);    
    // update graph
    update(time_data, sampled);
}

template<typename FloatType>
void tpSDOstreamclust<FloatType>::predictOnly_impl(
        std::vector<int>& label,
        std::vector<FloatType>& score,
        const std::vector<Vector<FloatType>>& data, 
        const std::vector<FloatType>& epsilon,
        const std::vector<FloatType>& time) {
    // Check for equal lengths:
    if (data.size() != time.size()) {
        throw std::invalid_argument("data and now must have the same length");
    }
    int first_index = last_predicted_index;
    predict_impl(label, score, data, epsilon, first_index);
    last_predicted_index += data.size();
}

#endif  // TPSDOSTREAMCLUST_FITPRED_H
#ifndef SDOCLUSTREAM_FITPRED_H
#define SDOCLUSTREAM_FITPRED_H

// #include "SDOclustream_print.h"
#include "SDOclustream_util.h"
#include "SDOclustream_sample.h"
#include "SDOclustream_fit.h"
#include "SDOclustream_predict.h"

template<typename FloatType>
void SDOclustream<FloatType>::fitPredict_impl(
        std::vector<int>& label,
        std::vector<FloatType>& score,
        const std::vector<Vector<FloatType>>& data, 
        const std::vector<FloatType>& epsilon,
        const std::vector<FloatType>& time) {
    // Check for equal lengths:
    if (data.size() != time.size()) {
        throw std::invalid_argument("data and now must have the same length");
    }
    int first_index = last_index;
    // sample data
    std::unordered_set<int> sampled;   
    sample(sampled, data, epsilon, time, first_index);
    // fit model 
    fit_impl(data, epsilon, time, first_index);    
    // update graph
    update(time, sampled);
    // predict
    predict_impl(label, score, data, epsilon, first_index);
    last_predicted_index = last_index;
}

template<typename FloatType>
void SDOclustream<FloatType>::fitOnly_impl(
        const std::vector<Vector<FloatType>>& data, 
        const std::vector<FloatType>& epsilon,
        const std::vector<FloatType>& time) {
    // Check for equal lengths:
    if (data.size() != time.size()) {
        throw std::invalid_argument("data and now must have the same length");
    }
    int first_index = last_index;
    // sample data
    std::unordered_set<int> sampled;   
    sample(sampled, data, epsilon, time, first_index);
    // fit model 
    fit_impl(data, epsilon, time, first_index);    
    // update graph
    update(time, sampled);
}

template<typename FloatType>
void SDOclustream<FloatType>::predictOnly_impl(
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

#endif  // SDOCLUSTREAM_FITPRED_H
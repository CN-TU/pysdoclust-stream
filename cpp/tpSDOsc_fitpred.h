#ifndef TPSDOSC_FITPRED_H
#define TPSDOSC_FITPRED_H

#include "tpSDOsc_print.h"
#include "tpSDOsc_graph.h"
#include "tpSDOsc_util.h"
#include "tpSDOsc_sample.h"
#include "tpSDOsc_fit.h"
#include "tpSDOsc_predict.h"

// template<typename FloatType>
// class tpSDOsc<FloatType>::TreeNodeUpdater {
//     Vector<FloatType> new_data;
//     int new_key;
//     public:
//     TreeNodeUpdater(Vector<FloatType> new_data, int new_key) : new_data(new_data), new_key(new_key) {}
//     void operator() (Vector<FloatType>& vector, int& key) {
//         int i = 0;
//         for (FloatType& element : vector) {
//             element = new_data[i];
//             i++;
//         }
//         key = new_key;
//     }
// };

template<typename FloatType>
void tpSDOsc<FloatType>::fitPredict_impl(
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
    fit_impl(data, epsilon, time_data, sampled, first_index);    
    // update graph
    update(time_data, sampled);
    // predict
    predict_impl(label, score, data, epsilon, sampled, first_index);
}

template<typename FloatType>
void tpSDOsc<FloatType>::fitOnly_impl(
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
    fit_impl(data, epsilon, time_data, sampled, first_index);    
    // update graph
    update(time_data, sampled);
}

#endif  // TPSDOSC_FITPRED_H
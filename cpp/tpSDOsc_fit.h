#ifndef TPSDOSC_FIT_H
#define TPSDOSC_FIT_H

#include "tpSDOsc_observer.h"

template<typename FloatType>
void tpSDOsc<FloatType>::fit_impl(
        std::unordered_map<int, std::pair<std::vector<std::complex<FloatType>>, FloatType>>& temporary_scores,
        const Vector<FloatType>& point,
        const FloatType& now,           
        const int& current_observer_cnt,
        const int& current_neighbor_cnt,
        const int& observer_index) {  
    TreeNeighbors nearestNeighbors = tree.knnSearch(point, current_neighbor_cnt + 1); 
    std::vector<std::complex<FloatType>> score_vector;
    initNowVector(now, score_vector, obs_scaler[current_observer_cnt]);
    for (const auto& neighbor : nearestNeighbors) {
        int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
        if (idx!=observer_index) {
            if (temporary_scores.count(idx) > 0) {                
                auto& value_pair = temporary_scores[idx];
                std::vector<std::complex<FloatType>>& observations = value_pair.first;
                FloatType fading_factor = std::pow<FloatType>(fading, now-value_pair.second);
                for (std::size_t freq_ind = 0; freq_ind < freq_bins; freq_ind++) {
                    observations[freq_ind] *= fading_factor;
                    observations[freq_ind] += score_vector[freq_ind];
                }
                value_pair.second = now;
            } else { 
                std::vector<std::complex<FloatType>> observations(freq_bins);
                for (std::size_t freq_ind = 0; freq_ind < freq_bins; freq_ind++) {
                    observations[freq_ind] = score_vector[freq_ind];
                }
                temporary_scores[idx] = std::make_pair(observations, now);
            }
        }            
    }  
};

template<typename FloatType>
void tpSDOsc<FloatType>::fit_impl(
        std::unordered_map<int, std::pair<std::vector<std::complex<FloatType>>, FloatType>>& temporary_scores,
        const Vector<FloatType>& point,
        const FloatType& now,           
        const int& current_observer_cnt,
        const int& current_neighbor_cnt) {   
    TreeNeighbors nearestNeighbors = tree.knnSearch(point, current_neighbor_cnt); // one more cause one point is Observer
    std::vector<std::complex<FloatType>> score_vector;
    initNowVector(now, score_vector, obs_scaler[current_observer_cnt]);
    for (const auto& neighbor : nearestNeighbors) {
        int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
        if (temporary_scores.count(idx) > 0) {                
            auto& value_pair = temporary_scores[idx];
            std::vector<std::complex<FloatType>>& observations = value_pair.first;
            FloatType fading_factor = std::pow<FloatType>(fading, now-value_pair.second);
            for (std::size_t freq_ind = 0; freq_ind < freq_bins; freq_ind++) {
                observations[freq_ind] *= fading_factor;
                observations[freq_ind] += score_vector[freq_ind];
            }
            value_pair.second = now;
        } else { 
            std::vector<std::complex<FloatType>> observations(freq_bins);
            for (std::size_t freq_ind = 0; freq_ind < freq_bins; freq_ind++) {
                observations[freq_ind] = score_vector[freq_ind];
            }
            temporary_scores[idx] = std::make_pair(observations, now);
        }
    }  
};

template<typename FloatType>
void tpSDOsc<FloatType>::update_model(
        const std::unordered_map<int, std::pair<std::vector<std::complex<FloatType>>, FloatType>>& temporary_scores) {
    for (auto& [key, value_pair] : temporary_scores) {
        const MapIterator& it = indexToIterator[key];
        // Access the value pair:
        const std::vector<std::complex<FloatType>>& score_vector = value_pair.first;
        FloatType time_touched = value_pair.second;
        auto node = observers.extract(it);  
        Observer& observer = node.value();        
        observer.updateObservations(freq_bins, std::pow(fading, time_touched - observer.time_touched), score_vector);
        observer.time_touched = time_touched;
        observers.insert(std::move(node));
    }
};

#endif  // TPSDOSC_FIT_H
#ifndef TPSDOSC_SAMPLE_H
#define TPSDOSC_SAMPLE_H

#include "tpSDOsc_observer.h"

template<typename FloatType>
bool tpSDOsc<FloatType>::sampleData( 
    std::unordered_set<int>& sampled,
    const FloatType& now,
    const int& batch_size, // actually batch size - 1
    const FloatType& batch_time,
    const int& current_index) {
    bool add_as_observer = 
        batch_size == 0 ||
        (rng() - rng.min()) * batch_size < sampling_first * (rng.max() - rng.min()) * batch_time;
    if (add_as_observer) {            
        sampled.insert(current_index);   
        last_added_index = current_index;
        last_added_time = now;
        return true;
    }
    return false;
};

template<typename FloatType>
void tpSDOsc<FloatType>::sampleData(
        std::unordered_set<int>& sampled,
        const Vector<FloatType>& point,
        const FloatType& now,
        FloatType observations_sum,
        const int& current_observer_cnt,
        const int& current_neighbor_cnt,
        const int& current_index) {        
    bool add_as_observer;
    if (!observers.empty()) {            
        auto nearestNeighbors = tree.knnSearch(point, current_neighbor_cnt);
        FloatType observations_nearest_sum(0);
        for (const auto& neighbor : nearestNeighbors) {
            int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered               
            const MapIterator& it = indexToIterator[idx];
            observations_nearest_sum += it->getObservations() * std::pow<FloatType>(fading, now-it->time_touched);
        }   
        add_as_observer = 
            (rng() - rng.min()) * current_neighbor_cnt * observations_sum * (current_index - last_added_index) < 
                sampling_first * (rng.max() - rng.min()) * current_observer_cnt * observations_nearest_sum * (now - last_added_time);
    } else {
        add_as_observer = 
        (rng() - rng.min()) * (current_index - last_added_index) < 
            sampling_first * (rng.max() - rng.min()) * (now - last_added_time);
    }        
    if (add_as_observer) {            
        sampled.insert(current_index);   
        last_added_index = current_index;
        last_added_time = now;
    }
};

template<typename FloatType>
void tpSDOsc<FloatType>::replaceObservers(
        Vector<FloatType> data,
        std::unordered_set<int>& dropped,
        std::priority_queue<MapIterator,std::vector<MapIterator>,IteratorAvCompare>& worst_observers,
        const FloatType& now,
        const int& current_observer_cnt,
        const int& current_index) {        
    MapIterator obsIt = observers.end();
    std::vector<std::complex<FloatType>> init_score_vector;
    FloatType init_score = obs_scaler[current_observer_cnt];
    initNowVector(now,  init_score_vector, init_score);
    if (observers.size() < observer_cnt) {
        obsIt = observers.insert(Observer(data, init_score_vector, now, init_score, current_index, &tree, &treeA)); // maybe init_score instead of 1
    } else {
        // find worst observer
        obsIt = worst_observers.top();  // Get iterator to the "worst" element         
        worst_observers.pop(); 
        int indexToRemove = obsIt->index;
        // do index handling
        dropped.insert(indexToRemove);            
        indexToIterator.erase(indexToRemove);
        // update Observer(s)
        auto node = observers.extract(obsIt);
        Observer& observer = node.value();
        observer.reset(data, init_score_vector, now, init_score, current_index, &tree, &treeA); // maybe init_score instead of 1
        observers.insert(std::move(node));    
    }
    indexToIterator[current_index] = obsIt;
};

#endif  // TPSDOSC_SAMPLE_H
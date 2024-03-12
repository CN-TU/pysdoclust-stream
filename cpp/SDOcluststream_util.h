#ifndef SDOCLUSTSTREAM_UTIL_H
#define SDOCLUSTSTREAM_UTIL_H

#include "SDOcluststream_observer.h"

template<typename FloatType>
bool SDOcluststream<FloatType>::hasEdge(
        FloatType distance, 
        const MapIterator& it) {
    return distance < (zeta * it->h + (1 - zeta) * h);
}

template<typename FloatType>
void SDOcluststream<FloatType>::setObsScaler() {
    FloatType prob0 = 1.0f;
    for (int i = neighbor_cnt; i > 0; --i) {
        prob0 *= static_cast<FloatType>(i) / (observer_cnt+1 - i);
    }

    obs_scaler[observer_cnt] = 1.0f;
    FloatType prob = prob0;

    int current_neighbor_cnt = neighbor_cnt;
    
    for (int i = observer_cnt - 1; i > 0; --i) {
        prob *= static_cast<FloatType>(i+1) / static_cast<FloatType>((i+1)-current_neighbor_cnt);

        int current_neighbor_cnt_target = (static_cast<FloatType>(i-1)) / static_cast<FloatType>((observer_cnt-1)) * neighbor_cnt + 1;   
        while (current_neighbor_cnt > current_neighbor_cnt_target) {      
            prob *= static_cast<FloatType>(i+1-current_neighbor_cnt) / static_cast<FloatType>(current_neighbor_cnt);

            current_neighbor_cnt--;
        }
        obs_scaler[i] = prob0 / prob;
    }
    obs_scaler[0] = prob0;
}

template<typename FloatType>
void SDOcluststream<FloatType>::setModelParameters(
        int& current_observer_cnt, int&current_observer_cnt2,
        int& active_threshold, int& active_threshold2,
        int& current_neighbor_cnt, int& current_neighbor_cnt2,
        std::size_t& current_e,
        std::size_t& chi,
        bool print) {

    current_observer_cnt = observers.size();
    current_observer_cnt2 = observers.size()-1;

    active_threshold = (current_observer_cnt - 1) * active_observers; // active_threshold+1 active observers
    active_threshold2 = (current_observer_cnt2 - 1) * active_observers; // active_threshold+1 active observers

    current_neighbor_cnt = (observers.size() == observer_cnt) ?
                        neighbor_cnt :
                        static_cast<int>((current_observer_cnt - 1) / static_cast<FloatType>(observer_cnt - 1) * neighbor_cnt + 1);
    current_neighbor_cnt2 = static_cast<int>((current_observer_cnt2 - 1) / static_cast<FloatType>(observer_cnt - 1) * neighbor_cnt + 1);
    
    current_e = (observers.size() == observer_cnt) ?
            e :
            static_cast<size_t>((current_observer_cnt - 1) / static_cast<FloatType>(observer_cnt - 1) * e + 1);

    int current_chi_min = (observers.size() == observer_cnt) ?
                    chi_min :
                    static_cast<int>((current_observer_cnt - 1) / static_cast<FloatType>(observer_cnt - 1) * chi_min + 1);
    chi = std::max(static_cast<std::size_t>(current_observer_cnt * chi_prop), static_cast<std::size_t>(current_chi_min));
    
    if (print) {
        std::cout << std::endl;
        std::cout << "Observers: " << current_observer_cnt << ", " << current_observer_cnt2;
        std::cout << ", Active Observers: " << active_threshold + 1 << ", " << active_threshold2 + 1;
        std::cout << ", Neighbors: " << current_neighbor_cnt << ", " << current_neighbor_cnt2;
        std::cout << ", e: " << current_e;
        std::cout << ", chi: " << chi;
        std::cout << std::endl;     
    }            
}

template<typename FloatType>
void SDOcluststream<FloatType>::updateModel(
        const std::unordered_map<int,std::pair<FloatType, FloatType>>& temporary_scores) {
    
    for (auto& [key, value_pair] : temporary_scores) {
        const MapIterator& it = indexToIterator[key];

        // Access the value pair:
        FloatType score = value_pair.first;
        FloatType time_touched = value_pair.second;

        auto node = observers.extract(it);    

        Observer& observer = node.value();
        observer.observations *= std::pow<FloatType>(fading, time_touched-observer.time_touched);
        observer.observations += score;
        observer.time_touched = time_touched;
        observers.insert(std::move(node));
    }
}

template<typename FloatType>
bool SDOcluststream<FloatType>::sampleData( 
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
}      

template<typename FloatType>
void SDOcluststream<FloatType>::replaceObservers(
        Vector<FloatType> data,
        std::unordered_set<int>& dropped,
        std::priority_queue<MapIterator,std::vector<MapIterator>,IteratorAvCompare>& worst_observers,
        const FloatType& now,
        const int& current_observer_cnt,
        const int& current_index) {        
    MapIterator obsIt = observers.end();
    if (observers.size() < observer_cnt) {
        obsIt = observers.insert(Observer(data, obs_scaler[current_observer_cnt], now, now, current_index)); // to add to the distance matrix
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
        observer.reset(data, obs_scaler[current_observer_cnt], now, now, current_index);
        observers.insert(std::move(node));    
    }
    indexToIterator[current_index] = obsIt;
}

#endif  // SDOCLUSTSTREAM_UTIL_H
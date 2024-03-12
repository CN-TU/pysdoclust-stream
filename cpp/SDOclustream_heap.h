#ifndef SDOCLUSTSTREAM_HEAP_H
#define SDOCLUSTSTREAM_HEAP_H

#include "SDOcluststream_util.h"
#include <array>

template<typename FloatType>
bool SDOcluststream<FloatType>::sampleData(
        std::unordered_set<int>& sampled,
        HeapType& heap,
        const Vector<FloatType>& point,
        const FloatType& now,
        FloatType observations_sum,
        const int& current_observer_cnt,
        const int& current_neighbor_cnt,
        const int& current_index) {        
    bool add_as_observer;
    if (!observers.empty()) {            
        constexpr size_t size = observers.size();
        std::array<DistancePairType> pairs;        
        int i = 0;
        for (auto it = observers.begin(); it != observers.end(); ++it) {
            FloatType distance = distance_function(it->getData(), point); 
            pairs[i++] = DistancePairType(it->index, distance);
            heap.insert(it->index, distance);
            if (i = active_threshold)
        }
        for (auto hIt = heap.begin(); hIt != heap.end(); ++hIt) { // not ordered, but not necessary here
            int idx = hIt->first;              
            const MapIterator& it = indexToIterator[idx];
            observations_nearest_sum += it->observations * std::pow<FloatType>(fading, now-it->time_touched);
        }  
        add_as_observer = 
            (rng() - rng.min()) * current_neighbor_cnt * observations_sum * (current_index - last_added_index) < 
                sampling_first * (rng.max() - rng.min()) * current_observer_cnt * observations_nearest_sum * (now - last_added_time);
        // if observer prepare heap with all distances
        if (add_as_observer) {
            heap.setBufferSize(observer_cnt - neighbor_cnt);
            for (const auto& pair : pairs) {
                if (!heap.isin(pair.first)) {
                    heap.insert(pair);
                }
            }
        }
    } else {
        add_as_observer = 
        (rng() - rng.min()) * (current_index - last_added_index) < 
            sampling_first * (rng.max() - rng.min()) * (now - last_added_time);
    }        
    if (add_as_observer) {            
        sampled.insert(current_index);   
        last_added_index = current_index;
        last_added_time = now;
        return true;
    }
    return false;
};

template<typename FloatType>
void SDOcluststream<FloatType>::fit_impl(
        std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
        HeapType& heap,
        const FloatType& now,          
        const int& current_observer_cnt,
        const int& current_neighbor_cnt) {        
    heap.setK(current_neighbor_cnt);
    for (auto hIt = heap.begin(); hIt != heap.end(); ++hIt) { // not ordered, but not necessary here
        int idx = hIt.first;
        if (temporary_scores.count(idx) > 0) {                
            auto& value_pair = temporary_scores[idx];
            value_pair.first *= std::pow<FloatType>(fading, now-value_pair.second);
            value_pair.second = now;
            value_pair.first += obs_scaler[current_observer_cnt];
        } else {
            temporary_scores[idx] = std::make_pair(obs_scaler[current_observer_cnt], now);
        }    
    }  
};

template<typename FloatType>
void SDOcluststream<FloatType>::predict_impl(
        int& label,
        FloatType& score,
        Heap heap,
        const Vector<FloatType>& point, // could be accessed as with observer_index
        const int& active_threshold,
        const int& current_neighbor_cnt) {
    std::unordered_map<int, FloatType> label_vector;
    heap.setK(current_neighbor_cnt);
    MapIterator it = observers.begin();
    std::advance(it, active_threshold + 1); //first inactive Obsever
    while (it !=observers.end()) {
        heap.erase(it->getIndex());
        ++it;
    }    
    if (heap.size() < current_neighbor_cnt) {
        std::cerr << "Warning: Heap size " << heap.size() << " is smaller than the threshold!" << std::endl;
    }

    score = heap.median(); // if buffer was too small heap can have less than current_neighbor_cnt elements
    for (auto hIt = heap.begin(); hIt != heap.end(); ++hIt) { // not ordered, but not necessary here
        int idx = hIt.first;
        const MapIterator& it = indexToIterator[idx];
        const auto& color_distribution = it->color_distribution;
        for (const auto& pair : color_distribution) {
            label_vector[pair.first] += pair.second;
        }
    }  
    // set label
    FloatType maxColorScore(0);
    for (const auto& pair : label_vector) {
        if ( pair.second > maxColorScore || (pair.second == maxColorScore && pair.first < label) ) {
            label = pair.first;
        }
    }
};

template<typename FloatType>
void SDOcluststream<FloatType>::updateH_single(
        MapIterator it, 
        size_t n) { 
    const Heap& heap = heap_matrix[it->index];
    heap.setK(n);
    if (!heap.empty()) {
        it->h = heap.top();
    } else {
        it->h = 0;
    }
}

template<typename FloatType>
void SDOcluststream<FloatType>::DFS(
        IndexSetType& cluster, 
        IndexSetType& processed, 
        const MapIterator& it) {
    // insert to sets
    processed.insert(it->index);   
    cluster.insert(it->index);
    const Heap& heap = heap_matrix[it->index];
    for (auto hIt = heap.begin(); hIt != heap.end(); ++it) { // not ordered 
        FloatType distance = hIt->second;
        int idx = hIt->first;
        if (!(processed.count(idx)>0) && hasEdge(distance, it)) {            
            const MapIterator& it1 = indexToIterator[dIt->index];
            if (hasEdge(distance, it1)) {
                DFS(cluster, processed, it1);
            }
        }
    }
}

template<typename FloatType>
void SDOcluststream<FloatType>::updateDistanceHeap(
        const HeapMatrix& sampled, // map of heaps // const?
        const std::unordered_set<int>& dropped,
        const std::unordered_set<int>& active,
        const std::unordered_set<int>& inactive,
        const std::unordered_set<int>& activated,
        const std::unordered_set<int>& deactivated) {

    // drop heaps from dropped observers
    for (int idx : dropped) {
        auto it = heap_matrix.find(idx);
        if (it != heap_matrix.end()) {
            heap_matrix.erase(it);
        }
    }
    // drop for all heaps the dropped entries
    for (auto& pair : distance_matrix) {
        int idx = pair.first;
        HeapType& heap = pair.second;
        for (int idy : dropped) {
            heap.erase(idy);
        }
    }
    // add distances between old Observers and newly sampled Observers
    // drop for the dropped observers
    // add distances between newly sampled Observers
    for (auto mIt = sampled.begin(); mIt != sampled.end(); ++mIt) {
        int idx = mIt->first;
        HeapType& heap = mIt->second;
        for (auto hIt = kd.begin(); hIt != kdend(); ++hIt) {
            int idy = hIt->first;
            if (dropped.count(idy)>0) {
                heap.erase(idy);
            } else {
                FloatType distance = hIt->second;
                heap_matrix[idy].insert(idx, distance);
            }
        }
        const MapIterator& it = indexToIterator[idx];
        Vector<FloatType> point = it->getData();
        for (auto mIt1 = sampled.begin(); mIt1 != mIt; ++mIt) {
            int idy = mIt1->second;
            HeapType& heap1 = mIt1->second;
            const MapIterator& it1 = indexToIterator[idy];
            FloatType distance = distance_function(point, it1->getData());
            heap.insert(idy, distance, true); 
            heap1.insert(idx, distance, true);
        }
    }
    // (de)activate 
    for (auto& pair : distance_matrix) {
        HeapType& heap = pair.second;
        for (int idy : deactivated) {
            heap.deactivate(idy);
        }
        for (int idy : activate) {
            heap.activate(idy);
        }
    }
    for (auto& pair : sampled) {
        HeapType& heap = pair.second;
        for (int idy : inactive) {
            if (heap.active(idy)) { heap.deactivate(idy); }
        }
        for (int idy : active) {
            if (!heap.active(idy)) { heap.activate(idy); }
        }
    }

    // add heaps from sampled observers
    heap_matrix.reserve(heap_matrix.size() + sampled.size()); // Optional: reserve memory for efficiency    
    heap_matrix.insert(std::move(sampled.begin()), std::move(sampled.end()));

    

}

#endif  // SDOCLUSTSTREAM_HEAP_H
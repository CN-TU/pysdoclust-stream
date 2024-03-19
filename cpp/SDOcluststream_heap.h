#ifndef SDOCLUSTSTREAM_HEAP_H
#define SDOCLUSTSTREAM_HEAP_H

#include "SDOcluststream_util.h"
#include "SDOcluststream_print.h"
#include "SDOcluststream_graph.h"

template<typename FloatType>
bool SDOcluststream<FloatType>::sampleData(
        std::unordered_set<int>& sampled,
        FibHeapType& heap,
        const Vector<FloatType>& point,
        const FloatType& now,
        FloatType observations_sum,
        const int& current_observer_cnt,
        const int& current_neighbor_cnt,
        const int& current_index) {        
    bool add_as_observer;
    if (!observers.empty()) {   
        int i = 0;
        for (auto it = observers.begin(); it != observers.end(); ++it) {
            FloatType distance = distance_function(it->getData(), point); 
            heap.push(std::make_pair(distance, it->index));
        }  
        FloatType observations_nearest_sum(0);      
        for (auto hIt = heap.begin(); hIt != heap.end(); ++hIt) { // not ordered, but not necessary here
            int idx = hIt->second;              
            const MapIterator& it = indexToIterator[idx];
            observations_nearest_sum += it->observations * std::pow<FloatType>(fading, now-it->time_touched);
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
        return true;
    }
    return false;
};

template<typename FloatType>
void SDOcluststream<FloatType>::fit_impl(
        std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
        FibHeapType& heap,
        std::unordered_set<int> dropped,
        const FloatType& now,          
        const int& current_observer_cnt,
        const int& current_neighbor_cnt) { 
    FibHeapType tempHeap;
    int i = 0;
    while (i<current_neighbor_cnt) {
        auto pair = heap.top();
        heap.pop();
        int idx = pair.second;
        if (!(dropped.count(idx)>0)) {
            if (temporary_scores.count(idx) > 0) {                
                auto& value_pair = temporary_scores[idx];
                value_pair.first *= std::pow<FloatType>(fading, now-value_pair.second);
                value_pair.second = now;
                value_pair.first += obs_scaler[current_observer_cnt];
            } else {
                temporary_scores[idx] = std::make_pair(obs_scaler[current_observer_cnt], now);
            }   
            tempHeap.push(pair);
            i++;
        }         
    }
    heap.merge(tempHeap); 
};

template<typename FloatType>
void SDOcluststream<FloatType>::predict_impl(
        int& label,
        FloatType& score,
        FibHeapType& heap,
        const int& current_neighbor_cnt) {
    std::unordered_map<int, FloatType> label_vector;
    int i = 0;
    while (i<current_neighbor_cnt) {
        auto pair = heap.top();
        heap.pop();
        int idx = pair.second;
        // label / color
        const MapIterator& it = indexToIterator[idx];
        if (it->active) {
            const auto& color_distribution = it->color_distribution;
            for (const auto& cpair : color_distribution) {
                label_vector[cpair.first] += cpair.second;
            }
            // outlier score / median
            if (current_neighbor_cnt%2>0) {
                if (i==current_neighbor_cnt/2) { score = pair.first; }
            } else {
                if ((i==current_neighbor_cnt/2) || (i==current_neighbor_cnt/2-1)) { score += 0.5 * pair.first; }
            }
            i++;     
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
    HeapType& heap = heap_matrix[it->index];
    heap.balanceK(n);
    if (!heap.empty()) {
        it->h = heap.topK();
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
    const HeapType& heap = heap_matrix[it->index];
    for (auto hIt = heap.begin(); hIt != heap.end(); ++hIt) { // not ordered 
        FloatType distance = hIt->first;
        int idx = hIt->second;
        if (!(processed.count(idx)>0) && hasEdge(distance, it)) {            
            const MapIterator& it1 = indexToIterator[idx];
            if (hasEdge(distance, it1)) {
                DFS(cluster, processed, it1);
            }
        }
    }
};

template<typename FloatType>
void SDOcluststream<FloatType>::updateHeap(
        FibHeapType& heap, // map of heaps // const?
        const Vector<FloatType>& point,  
        const std::unordered_set<int>& sampled,
        int observer_index) {
    for (int idy : sampled) {
        if (idy != observer_index) {
            const MapIterator& it1 = indexToIterator[idy];
            FloatType distance = distance_function(point, it1->getData());
            heap.push(std::make_pair(distance, idy));           
        }
    }
};

template<typename FloatType>
void SDOcluststream<FloatType>::updateHeapMatrix(
        HeapMatrix& sampled, // map of heaps // const?
        const std::unordered_set<int>& dropped,
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
    // drop for all heaps the drop entries
    for (auto& pair : heap_matrix) {
        int idx = pair.first;
        HeapType& heap = pair.second;
        for (int idy : dropped) {
            heap.erase(idy);
        }
    }    
    // drop for the sampled heap drop enties
    for (auto& pair : sampled) {
        int idx = pair.first;
        HeapType& heap = pair.second;
        for (int idy : dropped) {
            heap.erase(idy);
        }
    }
    // distance between old obs and newly sampled obs
    for (auto it = observers.begin(); it != observers.end(); ++it) {
        int idx = it->index;
        if (!(sampled.count(idx)>0)) { // with new Observers distances exist
            for (auto& pair : sampled) {
                int idy = pair.first;
                const HeapType& heap = pair.second;
                FloatType distance = distance_function(it->getData(), indexToIterator[idy]->getData());
                // heap_matrix[idx].insert(idy, distance);
                heap_matrix[idx].insert(idy, heap[idx], inactive.count(idy)>0); // condition if inactive or active
            }
        }
    }
    // (de)activate
    for (auto& pair : heap_matrix) {
        int idx = pair.first;
        if (!(sampled.count(idx)>0)) {
            HeapType& heap = pair.second;
            for (int idy : deactivated) {
                heap.deactivate(idy);
            }
            for (int idy : activated) {
                heap.activate(idy);
            }    
        }        
    }
    // add heaps from sampled observers
    heap_matrix.reserve(heap_matrix.size() + sampled.size()); // Optional: reserve memory for efficiency    
    std::transform(std::make_move_iterator(sampled.begin()), std::make_move_iterator(sampled.end()),
                std::inserter(heap_matrix, heap_matrix.end()),
                [](const auto& pair) { return std::make_pair(std::move(pair.first), std::move(pair.second)); });

    sampled.clear();
};



template<typename FloatType>
std::vector<int> SDOcluststream<FloatType>::fitPredict_impl(
        const std::vector<Vector<FloatType>>& data, 
        const std::vector<FloatType>& time_data, 
        bool fit_only) {
    // Check for equal lengths:
    if (data.size() != time_data.size()) {
        throw std::invalid_argument("data and now must have the same length");
    }
    FloatType now = time_data.front();
    std::vector<int> labels(data.size(), 0);    
    std::unordered_set<int> sampled;
    int active_threshold(0), active_threshold2(0);
    int current_neighbor_cnt(0), current_neighbor_cnt2(0);
    int current_observer_cnt(0), current_observer_cnt2(0);
    size_t current_e(0); // unused 
    size_t chi(0);    
    const int first_index(last_index);
    FloatType observations_sum(0);     
    for (auto it = observers.begin(); it != observers.end(); ++it) {
        observations_sum += it->observations * std::pow<FloatType>(fading, now-it->time_touched);
    }
    FibHeapMatrix heaps;
    if (observers.empty()) {
        bool firstPointSampled(false);
        for (size_t i = 0; i < data.size(); ++i) {  
            heaps[i] = FibHeapType();  
            int current_index = last_index;         
            if (firstPointSampled) {
                sampleData(
                    sampled,
                    heaps[current_index],
                    data[i],                    
                    time_data[i],
                    observations_sum * std::pow<FloatType>(fading, time_data[i]-now), // 0
                    current_observer_cnt,
                    current_neighbor_cnt,
                    current_index);        
            } else {
                if (sampleData(
                        sampled,
                        time_data[i],
                        data.size() - 1,
                        time_data.back() - now,
                        current_index)
                    ) {firstPointSampled = true;}
            }
            last_index++;
        }

        if (!firstPointSampled) {
            std::uniform_int_distribution<int> dist(0, data.size() - 1);
            int indexToSample = dist(rng);
            sampled.insert(indexToSample);
            last_added_index = indexToSample;
            last_added_time = time_data[indexToSample];                
        }
    } else {
        setModelParameters(
            current_observer_cnt, current_observer_cnt2,
            active_threshold, active_threshold2,
            current_neighbor_cnt, current_neighbor_cnt2,
            current_e,
            chi,
            false); // true for print
        for (size_t i = 0; i < data.size(); ++i) { 
            int current_index = last_index;   
            heaps[current_index] = FibHeapType();               
            sampleData(
                sampled,
                heaps[current_index],
                data[i],                    
                time_data[i],
                observations_sum * std::pow<FloatType>(fading, time_data[i]-now),
                current_observer_cnt,
                current_neighbor_cnt,
                current_index); 
            last_index++;
        }
    }
    printFibHeapMatrix(heaps);
    // Can not replace more observers than max size of model
    if (sampled.size()>observer_cnt) {
        // 1. Transfer elements to a vector:
        std::vector<typename std::unordered_set<int>::value_type> shuffled_elements(sampled.begin(), sampled.end());

        // 2. Shuffle the elements in the vector:
        std::shuffle(shuffled_elements.begin(), shuffled_elements.end(), rng);

        // 3. Clear the original set:
        sampled.clear();

        // 4. Reinsert only the desired number of elements:
        sampled.insert(shuffled_elements.begin(), shuffled_elements.begin() + observer_cnt);
    }



    // only push number of sampled into this queue, double side queue necessary then
    IteratorAvCompare iterator_av_compare(fading, now);
    std::priority_queue<MapIterator,std::vector<MapIterator>,IteratorAvCompare> worst_observers(iterator_av_compare);
    for (auto it = observers.begin(); it != observers.end(); ++it) {
        worst_observers.push(it);            
    }
    std::unordered_set<int> dropped;
    for (size_t i = 0; i < data.size(); ++i) {
        if (sampled.count(first_index + i) > 0) {
            replaceObservers(
                data[i],
                dropped,
                worst_observers,
                now,
                current_observer_cnt,
                first_index + i
            );
            last_added_index = first_index + i;
            last_added_time = time_data[i];
        }
    }    
    setModelParameters(
        current_observer_cnt, current_observer_cnt2,
        active_threshold, active_threshold2,
        current_neighbor_cnt, current_neighbor_cnt2,
        current_e,
        chi,
        false); // true for print
    for (size_t i = 0; i < data.size(); ++i) { 
        int current_index = first_index + i;
        bool is_observer = sampled.count(current_index) > 0; 
        updateHeap(
            heaps[current_index],
            data[i],
            sampled,
            is_observer ? current_index : -1);
    }  
    printFibHeapMatrix(heaps);  
    // fit model
    std::unordered_map<int, std::pair<FloatType, FloatType>> temporary_scores; // index, (score, time_touched)
    for (size_t i = 0; i < data.size(); ++i) { 
        int current_index = first_index + i;
        bool is_observer = sampled.count(current_index) > 0; 
        fit_impl(
            temporary_scores,
            heaps[current_index],
            dropped,
            time_data[i],
            is_observer ? current_observer_cnt2 : current_observer_cnt, // if sampled or not
            is_observer ? current_neighbor_cnt2 : current_neighbor_cnt);
    }
    updateModel(temporary_scores);
    // update active tree
    std::unordered_set<int> activated;
    std::unordered_set<int> active;
    std::unordered_set<int> deactivated;
    std::unordered_set<int> inactive;
    int i = 0;
    for (MapIterator it = observers.begin(); it != observers.end(); ++it) {            
        if (i > active_threshold) {
            inactive.insert(it->index);
            if (it->active) {
                deactivated.insert(it->index);
                it->active = false;
            }
        } else {
            active.insert(it->index);
            if (!it->active) {
                activated.insert(it->index);
                it->active = true;
            }
        }
        ++i;
    }
    HeapMatrix sampled_heaps;
    for (size_t i = 0; i < data.size(); ++i) { 
        int idx = first_index + i;
        if (sampled.count(idx)>0) {
            std::cout << idx << " ";
            FibHeapType& heap = heaps[idx];
            sampled_heaps[idx] = HeapType(current_neighbor_cnt);
            HeapType& sampled_heap = sampled_heaps[idx];
            for (auto it = heap.begin(); it != heap.end(); ++it) {
                if (active.count(it->second)>0) {
                    sampled_heap.insert(*it);
                    continue;
                }
                if (inactive.count(it->second)>0) {
                    sampled_heap.insert(*it, true);
                }
            }
        }
    }
    printHeapMatrix(sampled_heaps);
    updateHeapMatrix(
        sampled_heaps, // map of heaps // const?
        dropped,
        inactive,
        activated,
        deactivated);  
    printHeapMatrix();  
    // update graph
    now = time_data.back(); // last timestamp of batch    
    updateGraph(
        now,
        active_threshold,
        e, // current_e,
        chi);
    std::vector<FloatType> scores(data.size(), 0);  
    if (!fit_only) {
        for (size_t i = 0; i < data.size(); ++i) {
            int current_index = first_index + i;
            bool is_observer =  (sampled.count(current_index) > 0);            
            int label(0);
            predict_impl(
                labels[i],
                scores[i],
                heaps[current_index],
                is_observer ? current_neighbor_cnt2 : current_neighbor_cnt);
            // gamma_dist.update(score);
            gamma_dist.update(scores[i], fading, time_data[i]);
        }        
        gamma_dist.update();
        for (size_t i = 0; i < scores.size(); ++i) {
            if (gamma_dist.isOutlier(scores[i], p_outlier)) {labels[i] = 0;}
        }    
    } 
    return labels;
};

#endif  // SDOCLUSTSTREAM_HEAP_H
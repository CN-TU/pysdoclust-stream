#ifndef SDOCLUSTSTREAM_HEAP_H
#define SDOCLUSTSTREAM_HEAP_H

#include "SDOcluststream_util.h"
#include "SDOcluststream_print.h"
#include "SDOcluststream_graph.h"

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
        std::vector<std::pair<int, FloatType>> pairs(observers.size());  // Pre-allocate space    
        int i = 0;
        for (auto it = observers.begin(); it != observers.end(); ++it) {
            FloatType distance = distance_function(it->getData(), point); 
            pairs[i++] = std::make_pair(it->index, distance);
            heap.insert(it->index, distance);
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
        // if observer prepare heap with all distances
        if (add_as_observer) {
            heap.setMaxBufferSize(observer_cnt - heap.getK());
            for (auto pair : pairs) {
                if (!heap.in(pair.first)) {
                    heap.insert(pair.first, pair.second);
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
    heap.balanceK(current_neighbor_cnt);
    for (auto hIt = heap.begin(); hIt != heap.end(); ++hIt) { // not ordered, but not necessary here
        int idx = hIt->second;
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
        HeapType heap,
        const int& current_neighbor_cnt) {
    std::unordered_map<int, FloatType> label_vector;
    heap.balanceK(current_neighbor_cnt);
    MapIterator it = observers.begin();    
    if (heap.size() < current_neighbor_cnt) {
        std::cerr << "Warning: Heap size " << heap.size() << " is smaller than the threshold!" << std::endl;
    }

    score = heap.median(); // if buffer was too small heap can have less than current_neighbor_cnt elements
    for (auto hIt = heap.begin(); hIt != heap.end(); ++hIt) { // not ordered, but not necessary here
        int idx = hIt->second;
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
        HeapType& heap, // map of heaps // const?
        const Vector<FloatType>& point,        
        const std::unordered_set<int>& dropped,
        const std::unordered_set<int>& sampled,
        int observer_index,
        int current_neighbor_cnt) {
    heap.balanceK(current_neighbor_cnt);
    for (int idy : dropped) {
        heap.erase(idy);
    }
    for (int idy : sampled) {
        if (idy != observer_index) {
            const MapIterator& it1 = indexToIterator[idy];
            FloatType distance = distance_function(point, it1->getData());
            heap.insert(idy, distance);
        }
    }
};

template<typename FloatType>
void SDOcluststream<FloatType>::updateHeapMatrix(
        HeapMatrix& sampled, // map of heaps // const?
        const std::unordered_set<int>& dropped,
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
    for (auto& pair : heap_matrix) {
        int idx = pair.first;
        HeapType& heap = pair.second;
        for (int idy : dropped) {
            heap.erase(idy);
        }
    }
    // add distances between old Observers and newly sampled Observers
    // drop for the dropped observers
    for (auto& pair : sampled) {
        int idx = pair.first;
        HeapType& heap = pair.second;
        for (auto hIt = heap.begin(); hIt != heap.end(); ++hIt) {
            int idy = hIt->second;
            if (dropped.count(idy)>0) {
                heap.erase(idy);
            } else if (sampled.count(idy)==0) { // with new Observers distances exist
                FloatType distance = hIt->first;
                heap_matrix[idy].insert(idx, distance);
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
    heap_matrix.insert(std::move(sampled.begin()), std::move(sampled.end()));
};



template<typename FloatType>
std::vector<int> SDOcluststream<FloatType>::fitPredict_impl(
        const std::vector<Vector<FloatType>>& data, 
        const std::vector<FloatType>& time_data, 
        bool fit_only) {

    std::cout << ">>> Entering fitPredict_impl" << std::endl;

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

    int buffer_size = 10;

    HeapMatrix heaps;
    std::cout << ">>> Entering Sampling " << std::endl;
    if (observers.empty()) {
        // std::cout << std::endl << "init obs: ";
        bool firstPointSampled(false);
        for (size_t i = 0; i < data.size(); ++i) {  
            heaps[i] = HeapType(1, buffer_size);           
            if (firstPointSampled) {
                sampleData(
                    sampled,
                    heaps[i],
                    data[i],                    
                    time_data[i],
                    observations_sum * std::pow<FloatType>(fading, time_data[i]-now), // 0
                    current_observer_cnt,
                    current_neighbor_cnt,
                    last_index++);        
            } else {
                if (sampleData(
                        sampled,
                        time_data[i],
                        data.size() - 1,
                        time_data.back() - now,
                        last_index++)
                    ) {firstPointSampled = true;}
            }
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
            true); // true for print

        for (size_t i = 0; i < data.size(); ++i) {    
            heaps[last_index] = HeapType(current_neighbor_cnt, buffer_size);             
            sampleData(
                sampled,
                heaps[first_index + i],
                data[i],                    
                time_data[i],
                observations_sum * std::pow<FloatType>(fading, time_data[i]-now),
                current_observer_cnt,
                current_neighbor_cnt,
                last_index++);                
        }
    }

    std::cout << "HEAP AFTER SAMPLING" << std::endl;
    printHeapMatrix(heaps);

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

    // std::cout << "Before Replace " << std::endl;
    std::cout << ">>> Entering Replace " << std::endl;
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
            std::cout << indexToIterator[first_index+i]->index << " ";
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

    std::cout << ">>> Entering Update Heaps " << std::endl;
    for (size_t i = 0; i < data.size(); ++i) {        
        updateHeap(
            heaps[first_index + i],
            data[i],
            dropped, 
            sampled,
            (sampled.count(first_index + i) > 0) ? first_index + i : -1,
            (sampled.count(first_index + i) > 0) ? current_neighbor_cnt2 : current_neighbor_cnt
        );
    }
    
 
    // fit model
    std::cout << ">>> Entering Fit " << std::endl;
    std::unordered_map<int, std::pair<FloatType, FloatType>> temporary_scores; // index, (score, time_touched)
    for (size_t i = 0; i < data.size(); ++i) { 
        fit_impl(
            temporary_scores,
            heaps[first_index + i],
            time_data[i],
            (sampled.count(first_index + i) > 0) ? current_observer_cnt2 : current_observer_cnt, // if sampled or not
            (sampled.count(first_index + i) > 0) ? current_neighbor_cnt2 : current_neighbor_cnt);
    }
    std::cout << ">>> Entering Update Model " << std::endl;
    updateModel(temporary_scores);
    std::cout << ">>> Left Update Model " << std::endl;

    // update active tree
    std::unordered_set<int> activated;
    std::unordered_set<int> active;
    std::unordered_set<int> deactivated;
    std::unordered_set<int> inactive;
    int i = 0;
    for (MapIterator it = observers.begin(); it != observers.end(); ++it) {            
        if (i > current_observer_cnt) {
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

    std::cout << ">>> Entering Set inactive in data heaps " << std::endl;
    for (size_t i = 0; i < data.size(); ++i) { 
        for (int idx : inactive) {
            heaps[first_index + i].deactivate(idx);
        }
    }
    std::cout << ">>> ALL HEAPS " << std::endl;
    printHeapMatrix(heaps);

    
    HeapMatrix sampled_heaps;
    std::copy_if(heaps.begin(), heaps.end(),
                std::inserter(sampled_heaps, sampled_heaps.end()),
                [&sampled](const auto& pair) { return sampled.count(pair.first)>0; });
    // update distance heap

    std::cout << ">>> SAMPLED HEAPS " << std::endl;
    printHeapMatrix(sampled_heaps);

    std::cout << ">>>  HEAP MATRIX " << std::endl;
    printHeapMatrix();

    updateHeapMatrix(
        sampled_heaps, // map of heaps // const?
        dropped,
        activated,
        deactivated);
    // printDistanceMatrix();
    std::cout << ">>>  HEAP MATRIX AFTER " << std::endl;
    printHeapMatrix();
    
    // update graph
    std::cout << ">>> Entering Update Graph " << std::endl;
    now = time_data.back(); // last timestamp of batch    
    updateGraph(
        now,
        active_threshold,
        e, // current_e,
        chi);
    std::cout << ">>> Entering Predict " << std::endl;
    std::vector<FloatType> scores(data.size(), 0);    
    if (!fit_only) {
        for (size_t i = 0; i < data.size(); ++i) {
            int label(0);
            FloatType score(0);
            predict_impl(
                labels[i],
                scores[i],
                heaps[first_index + i],
                (sampled.count(first_index + i) > 0) ? current_neighbor_cnt2 : current_neighbor_cnt);
            // gamma_dist.update(score);
            gamma_dist.update(score, fading, time_data[i]);
        }
        
        gamma_dist.update();
        for (size_t i = 0; i < scores.size(); ++i) {
            if (gamma_dist.isOutlier(scores[i], p_outlier)) {labels[i] = 0;}
        }
    }     

    return labels;
};

#endif  // SDOCLUSTSTREAM_HEAP_H
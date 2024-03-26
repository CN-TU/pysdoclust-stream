// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.
#ifndef SDOCLUSTSTREAM_TREE_H
#define SDOCLUSTSTREAM_TREE_H

#include "SDOcluststream_print.h"
#include "SDOcluststream_graph.h"
#include "SDOcluststream_util.h"

// template<typename FloatType>
// class SDOcluststream<FloatType>::TreeNodeUpdater {
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
void SDOcluststream<FloatType>::DFS(
        IndexSetType& cluster, 
        IndexSetType& processed, 
        const MapIterator& it) {
    // insert to sets
    processed.insert(it->index);   
    cluster.insert(it->index);
    std::vector<std::pair<TreeIterator,FloatType>>& nearestNeighbors = it->nearestNeighbors;
    for (const auto& neighbor : nearestNeighbors) {       
        FloatType distance = neighbor.second;        
        if (!hasEdge(distance, it)) { break; }
        int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
        if (!(processed.count(idx)>0)) {
            const MapIterator& it1 = indexToIterator[idx];
            if (hasEdge(distance, it1)) {
                DFS(cluster, processed, it1);
            }
        }
    }
    if ((h > it->h) && (zeta < 1.0f)) {
        // Query search(const KeyType& needle, DistanceType min_radius = 0, DistanceType max_radius = std::numeric_limits<DistanceType>::infinity(), bool reverse = false, BoundEstimator estimator = NopBoundEstimator()) {
        auto additionalNeighbors = treeA.search(it->getData(), it->h , (zeta * it->h + (1 - zeta) * h));
        while (!additionalNeighbors.atEnd()) {
            // Dereference the iterator to get the current element
            auto neighbor = *additionalNeighbors;
            FloatType distance = neighbor.second;        
            if (!hasEdge(distance, it)) { break; }
            int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
            if (!(processed.count(idx)>0)) {
                const MapIterator& it1 = indexToIterator[idx];
                if (hasEdge(distance, it1)) {
                    DFS(cluster, processed, it1);
                }
            }
            ++additionalNeighbors;
        }
    }
}

template<typename FloatType>
void SDOcluststream<FloatType>::fit_impl(
        std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
        const Vector<FloatType>& point,
        const FloatType& now,           
        const int& current_observer_cnt,
        const int& current_neighbor_cnt,
        const int& observer_index) {  
    TreeNeighbors nearestNeighbors = tree.knnSearch(point, current_neighbor_cnt + 1); 
    for (const auto& neighbor : nearestNeighbors) {
        int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
        if (idx!=observer_index) {
            if (temporary_scores.count(idx) > 0) {                
                auto& value_pair = temporary_scores[idx];
                value_pair.first *= std::pow<FloatType>(fading, now-value_pair.second);
                value_pair.second = now;
                value_pair.first += obs_scaler[current_observer_cnt];
            } else { temporary_scores[idx] = std::make_pair(obs_scaler[current_observer_cnt], now); }
        }            
    }  
};

template<typename FloatType>
void SDOcluststream<FloatType>::fit_impl(
        std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
        const Vector<FloatType>& point,
        const FloatType& now,           
        const int& current_observer_cnt,
        const int& current_neighbor_cnt) {   
    TreeNeighbors nearestNeighbors = tree.knnSearch(point, current_neighbor_cnt); // one more cause one point is Observer
    for (const auto& neighbor : nearestNeighbors) {
        int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
        if (temporary_scores.count(idx) > 0) {                
            auto& value_pair = temporary_scores[idx];
            value_pair.first *= std::pow<FloatType>(fading, now-value_pair.second);
            value_pair.second = now;
            value_pair.first += obs_scaler[current_observer_cnt];
        } else { temporary_scores[idx] = std::make_pair(obs_scaler[current_observer_cnt], now); }
    }  
};

template<typename FloatType>
void SDOcluststream<FloatType>::determineLabelVector(
        std::unordered_map<int, FloatType>& label_vector,
        const std::pair<TreeIterator, FloatType>& neighbor) {
    int idx = neighbor.first->second; // second is distance, first->first Vector, Output is ordered
    const MapIterator& it = indexToIterator[idx];
    const auto& color_distribution = it->color_distribution;
    FloatType distance = neighbor.second;
    FloatType outlier_factor = FloatType(0);
    if (!hasEdge(distance / outlier_threshold, it)) {   
        FloatType h_bar = (zeta * it->h + (1 - zeta) * h);   
        // FloatType x = (distance - outlier_threshold * h_bar);
        // std::cout << x << " ";
        FloatType x = 1.0f/5.0f * (distance - outlier_threshold * h_bar) / h_bar; //
        outlier_factor = x / (1 + x);
        // 0.5 * ( 1 + kx) = kx >> 0,5 = 0,5*kx 
    }
    for (const auto& pair : color_distribution) {
        label_vector[pair.first] += (1-outlier_factor) * pair.second;
    }
    label_vector[-1] += outlier_factor; // outlier weight    
}

template<typename FloatType>
void SDOcluststream<FloatType>::predict_impl(
        int& label,
        FloatType& score,
        const Vector<FloatType>& point, // could be accessed as with observer_index
        const int& current_neighbor_cnt,
        const int& observer_index) {
    std::unordered_map<int, FloatType> label_vector;
    const MapIterator& it0 = indexToIterator[observer_index];
    TreeNeighbors& nearestNeighbors = it0->nearestNeighbors;
    int i = 0;
    for (const auto& neighbor : nearestNeighbors) {        
        if (observer_index!= neighbor.first->second) {            
            determineLabelVector(label_vector, neighbor);          
            ++i;
            if (i > current_neighbor_cnt) { break; }
        }
    }  
    // set label
    FloatType maxColorScore(0);
    if ( label_vector[-1]>(current_neighbor_cnt*0.5) ) {
        label = -1;
    } else {
        for (const auto& pair : label_vector) {            
            if (pair.first<0) { continue; }
            if (pair.second > maxColorScore || (pair.second == maxColorScore && pair.first < label) ) {
                label = pair.first;
                maxColorScore = pair.second;
            }
        }
    }
}

template<typename FloatType>
void SDOcluststream<FloatType>::predict_impl(
        int& label,
        FloatType& score,
        const Vector<FloatType>& point,
        const int& current_neighbor_cnt) {
    std::unordered_map<int, FloatType> label_vector;
    TreeNeighbors nearestNeighbors = treeA.knnSearch(point, current_neighbor_cnt, true, 0, std::numeric_limits<FloatType>::infinity(), false, false);
    int i = 0;
    for (const auto& neighbor : nearestNeighbors) {
        determineLabelVector(label_vector, neighbor);  
        ++i;
        if (i > current_neighbor_cnt) { break; }
    }  
    //set label
    FloatType maxColorScore(0);
    if ( label_vector[-1]>(current_neighbor_cnt*0.5) ) {
        label = -1;
    } else {
        for (const auto& pair : label_vector) {
            if (pair.first<0) { continue; }
            if (pair.second > maxColorScore || (pair.second == maxColorScore && pair.first < label) ) {
                label = pair.first;
                maxColorScore = pair.second;
            }
        }
    }
};

template<typename FloatType>
void SDOcluststream<FloatType>::sampleData(
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
    }
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
    std::vector<int> labels(data.size());
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
    if (observers.empty()) {
        // std::cout << std::endl << "init obs: ";
        bool firstPointSampled(false);
        for (size_t i = 0; i < data.size(); ++i) { 
            if (firstPointSampled) {
                sampleData(
                    sampled,
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
            false); // true for print
        for (size_t i = 0; i < data.size(); ++i) {                
            sampleData(
                sampled,
                data[i],                    
                time_data[i],
                observations_sum * std::pow<FloatType>(fading, time_data[i]-now),
                current_observer_cnt,
                current_neighbor_cnt,
                last_index++);                
        }
    }
    // Can not replace more observers than max size of model
    if (sampled.size()>observer_cnt) {
        std::vector<typename std::unordered_set<int>::value_type> shuffled_elements(sampled.begin(), sampled.end());
        std::shuffle(shuffled_elements.begin(), shuffled_elements.end(), rng);
        sampled.clear();
        sampled.insert(shuffled_elements.begin(), shuffled_elements.begin() + observer_cnt);
    }

    // std::cout << "Replacement rate: " << static_cast<FloatType>(sampled.size())/data.size() << std::endl;
    // Queue worst observers
    IteratorAvCompare iterator_av_compare(fading, now);
    std::priority_queue<MapIterator,std::vector<MapIterator>,IteratorAvCompare> worst_observers(iterator_av_compare);
    for (auto it = observers.begin(); it != observers.end(); ++it) {
        worst_observers.push(it);            
    }
    std::unordered_set<int> dropped;
    for (size_t i = 0; i < data.size(); ++i) {
        int current_index = first_index + i;
        bool is_observer = (sampled.count(current_index) > 0);     
        if (is_observer) {            
            replaceObservers(
                data[i],
                dropped,
                worst_observers,
                now,
                current_observer_cnt,
                current_index
            );
            last_added_index = current_index;
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

    // fit model
    std::unordered_map<int, std::pair<FloatType, FloatType>> temporary_scores; // index, (score, time_touched)
    for (size_t i = 0; i < data.size(); ++i) {   
        int current_index = first_index + 1;
        bool is_observer = (sampled.count(current_index) > 0);
        if (is_observer) {
            fit_impl(
                temporary_scores,
                data[i],
                time_data[i],
                current_observer_cnt2,
                current_neighbor_cnt2,
                current_index); 
        } else {
            fit_impl(
                temporary_scores,
                data[i],
                time_data[i],
                current_observer_cnt,
                current_neighbor_cnt); 
        }
    }
    updateModel(temporary_scores);
    // update active tree
    int i = 0;
    for (MapIterator it = observers.begin(); it != observers.end(); ++it) {            
        if (i > current_observer_cnt) {
            it->deactivate(&treeA);
        } else {
            it->activate(&treeA);
        }
        ++i;
    }
    i = 0;
    for (MapIterator it = observers.begin(); it != observers.end(); ++it) {   
        it->setH(&treeA, chi, (chi < current_neighbor_cnt2) ? current_neighbor_cnt2 : chi );
        ++i;
        if (i > current_observer_cnt) { break; }
    }
    updateH_all();
    // update graph
    now = time_data.back(); // last timestamp of batch    
    updateGraph(
        now,
        active_threshold,
        e, // current_e,
        chi);
    std::vector<FloatType> scores(data.size());
    if (!fit_only) {
        for (size_t i = 0; i < data.size(); ++i) {
            int label(0);
            FloatType score(0);
            int current_index = first_index + i;
            bool is_observer = sampled.count(first_index + i) > 0;
            if (is_observer) {
                predict_impl(
                    label,
                    score,
                    data[i],
                    current_neighbor_cnt2,
                    current_index);
            } else {
                predict_impl(
                    label,
                    score,
                    data[i],
                    current_neighbor_cnt);
            }
            labels[i] = label;
            scores[i] = score;   
        }
    }     

    return labels;
};

#endif  // SDOCLUSTSTREAM_TREE_H
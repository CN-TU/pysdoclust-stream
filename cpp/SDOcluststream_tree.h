// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.
#ifndef SDOCLUSTSTREAM_TREE_H
#define SDOCLUSTSTREAM_TREE_H

#include "SDOcluststream_print.h"
#include "SDOcluststream_graph.h"
#include "SDOcluststream_sorted.h"

template<typename FloatType>
class SDOcluststream<FloatType>::TreeNodeUpdater {
    Vector<FloatType> new_data;
    int new_key;
    public:
    TreeNodeUpdater(Vector<FloatType> new_data, int new_key) : new_data(new_data), new_key(new_key) {}
    void operator() (Vector<FloatType>& vector, int& key) {
        int i = 0;
        for (FloatType& element : vector) {
            element = new_data[i];
            i++;
        }
        key = new_key;
    }
};

template<typename FloatType>
struct SDOcluststream<FloatType>::MyTieBreaker {
    bool operator() (const typename Tree::ValueType& a, const typename Tree::ValueType& b) { return a.second > b.second; }
};  

template<typename FloatType>
void SDOcluststream<FloatType>::fit_impl(
        std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
        const Vector<FloatType>& point,
        const FloatType& now,
        const int& observer_index,            
        const int& current_observer_cnt,
        const int& current_neighbor_cnt) {        
    auto nearestNeighbors = tree.knnSearch(point, current_neighbor_cnt + 1); // one more cause one point is Observer
    for (const auto& neighbor : nearestNeighbors) {
        int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
        if (idx!=observer_index) {
            if (temporary_scores.count(idx) > 0) {                
                auto& value_pair = temporary_scores[idx];

                value_pair.first *= std::pow<FloatType>(fading, now-value_pair.second);
                value_pair.second = now;
                value_pair.first += obs_scaler[current_observer_cnt];
            } else {
                temporary_scores[idx] = std::make_pair(obs_scaler[current_observer_cnt], now);
            }
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
    auto nearestNeighbors = tree.knnSearch(point, current_neighbor_cnt);
    for (const auto& neighbor : nearestNeighbors) {
        int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
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
        const Vector<FloatType>& point, // could be accessed as with observer_index
        const int& observer_index,
        const int& current_neighbor_cnt) {
    
    // FloatType score = 0; // returned for first seen sample
    std::unordered_map<int, FloatType> label_vector;
    // int label (0);

    TopKDistanceHeap nearest(current_neighbor_cnt, distance_compare); // for getting median

    auto nearestNeighbors = tree_active.knnSearch(point, current_neighbor_cnt + 1); // one more cause point itself is found
    for (const auto& neighbor : nearestNeighbors) {
        int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
        if (idx!=observer_index) {
            const MapIterator& it = indexToIterator[idx];
            const auto& color_distribution = it->color_distribution;
            for (const auto& pair : color_distribution) {
                label_vector[pair.first] += pair.second;
            }
            FloatType distance = neighbor.second;
            nearest.insert(IndexDistancePair(it, distance));
        }            
    }  

    score = nearest.median();

    // set label
    FloatType maxColorScore(0);
    for (const auto& pair : label_vector) {
        if ( pair.second > maxColorScore || (pair.second == maxColorScore && pair.first < label) ) {
            label = pair.first;
        }
    }
};

template<typename FloatType>
void SDOcluststream<FloatType>::predict_impl(
        int& label,
        FloatType& score,
        const Vector<FloatType>& point,
        const int& current_neighbor_cnt) {
    std::unordered_map<int, FloatType> label_vector;
    TopKDistanceHeap nearest(current_neighbor_cnt, distance_compare); // for getting median
    auto nearestNeighbors = tree_active.knnSearch(point, current_neighbor_cnt); // one more cause point itself is found
    for (const auto& neighbor : nearestNeighbors) {
        int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
        const MapIterator& it = indexToIterator[idx];
        const auto& color_distribution = it->color_distribution;
        for (const auto& pair : color_distribution) {
            label_vector[pair.first] += pair.second;
        }
        FloatType distance = neighbor.second;
        nearest.insert(IndexDistancePair(it, distance));
    }  
    // set score
    score = nearest.median();
    // set label
    FloatType maxColorScore(0);
    for (const auto& pair : label_vector) {
        if ( pair.second > maxColorScore || (pair.second == maxColorScore && pair.first < label) ) {
            label = pair.first;
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
        // 1. Transfer elements to a vector:
        std::vector<typename std::unordered_set<int>::value_type> shuffled_elements(sampled.begin(), sampled.end());

        // 2. Shuffle the elements in the vector:
        std::shuffle(shuffled_elements.begin(), shuffled_elements.end(), rng);

        // 3. Clear the original set:
        sampled.clear();

        // 4. Reinsert only the desired number of elements:
        sampled.insert(shuffled_elements.begin(), shuffled_elements.begin() + observer_cnt);
    }

    IteratorAvCompare iterator_av_compare(fading, now);
    std::priority_queue<MapIterator,std::vector<MapIterator>,IteratorAvCompare> worst_observers(iterator_av_compare);
    for (auto it = observers.begin(); it != observers.end(); ++it) {
        worst_observers.push(it);            
    }

    // std::cout << "Before Replace " << std::endl;

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
    
    // update full tree
    TreeIterator it = tree.begin();
    while (it != tree.end()) {
        int idx = it->second;
        if (dropped.count(idx)>0) {
            TreeIterator it2 = it;
            ++it;   
            tree.erase(it2);
            continue;
        }
        ++it;
    }
    for (int idx : sampled) {
        MapIterator it = indexToIterator[idx];
        Vector<FloatType> point = it->data;
        tree.insert(tree.begin(), std::make_pair(point, idx));
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
        if (sampled.count(first_index + i) > 0) {                
            fit_impl(
                temporary_scores,
                data[i],
                time_data[i],
                first_index + i,
                current_observer_cnt2,
                current_neighbor_cnt2 + 1); 
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
    std::unordered_set<int> activated;
    std::unordered_set<int> active;
    std::unordered_set<int> deactivated(dropped);
    std::unordered_set<int> inactive(dropped);
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
    it = tree_active.begin();
    while (it != tree_active.end()) {
        int idx = it->second;
        if (deactivated.count(idx)>0) {
            TreeIterator it2 = it;
            ++it;   
            tree_active.erase(it2);
            continue;
        }    
        ++it;
    }
    for (int idx : activated) {
        MapIterator it = indexToIterator[idx];
        Vector<FloatType> point = it->data;
        tree_active.insert(tree_active.begin(), std::make_pair(point, idx));
    }
    // update distance matrix
    updateDistanceMatrix(
        active, 
        activated, 
        deactivated);
    // printDistanceMatrix();
    
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
            if (sampled.count(first_index + i) > 0) {
                predict_impl(
                    label,
                    score,
                    data[i],
                    first_index + i,
                    current_neighbor_cnt + 1);
            } else {
                predict_impl(
                    label,
                    score,
                    data[i],
                    current_neighbor_cnt);
            }
            labels[i] = label;
            scores[i] = score;   
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

#endif  // SDOCLUSTSTREAM_TREE_H
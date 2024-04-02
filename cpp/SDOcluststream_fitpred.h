#ifndef SDOCLUSTSTREAM_FITPRED_H
#define SDOCLUSTSTREAM_FITPRED_H

#include "SDOcluststream_print.h"
#include "SDOcluststream_graph.h"
#include "SDOcluststream_util.h"
#include "SDOcluststream_sample.h"
#include "SDOcluststream_fit.h"
#include "SDOcluststream_predict.h"

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
    if (!fit_only) {
        for (size_t i = 0; i < data.size(); ++i) {
            int current_index = first_index + i;
            bool is_observer = sampled.count(current_index) > 0;
            if (is_observer) {
                if (indexToIterator[current_index]->active) {
                    predict_impl(
                        labels[i],
                        data[i],
                        current_neighbor_cnt2,
                        current_index);
                } else {
                    predict_impl(
                    labels[i],
                    data[i],
                    current_neighbor_cnt2);
                }
            } else {
                predict_impl(
                    labels[i],
                    data[i],
                    current_neighbor_cnt);
            }
        }
    } 
    return labels;
};

#endif  // SDOCLUSTSTREAM_FITPRED_H
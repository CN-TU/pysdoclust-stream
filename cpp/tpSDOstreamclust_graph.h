#ifndef TPSDOSTREAMCLUST_GRAPH_H
#define TPSDOSTREAMCLUST_GRAPH_H

#include "tpSDOstreamclust_cluster.h"

template<typename FloatType>
void tpSDOstreamclust<FloatType>::update(
        const std::vector<FloatType>& time_data,
        const std::unordered_set<int>& sampled) {
    std::size_t active_threshold(0), active_threshold2(0);
    std::size_t current_neighbor_cnt(0), current_neighbor_cnt2(0);
    std::size_t current_observer_cnt(0), current_observer_cnt2(0);
    std::size_t current_e(0);
    std::size_t chi(0);    
    setModelParameters(
        current_observer_cnt, current_observer_cnt2,
        active_threshold, active_threshold2,
        current_neighbor_cnt, current_neighbor_cnt2,
        current_e,
        chi,
        false); // true for print
    // now is average time of batch
    FloatType now = std::accumulate(time_data.begin(), time_data.end(), 0.0) / time_data.size();
    FloatType active_observations_thresh = getActiveObservationsThreshold(active_threshold, now);
    std::vector<std::complex<FloatType>> now_vector;
    initNowVector(now, now_vector);
    // update active tree
    for (MapIterator it = observers.begin(); it != observers.end(); ++it) {   
        FloatType fading_factor = std::pow<FloatType>(fading, now-it->time_touched);  
        FloatType proj_observations = it->getProjObservations(now_vector, fading_factor);      
        if (proj_observations < active_observations_thresh) {
            it->deactivate(&treeA); 
        } else {
            it->activate(&treeA);
        }
    }
    for (MapIterator it = observers.begin(); it != observers.end(); ++it) {  
        if (it->active) { it->setH(&treeA, chi, (chi < current_neighbor_cnt2) ? current_neighbor_cnt2 : chi ); } 
    }
    updateH_all();
    // update graph
    FloatType batch_age = calcBatchAge(time_data);
    FloatType age_factor = std::pow(fading, time_data.back() - last_time);
    for (MapIterator it = observers.begin(); it != observers.end(); ++it) {           
        FloatType score(0);
        if (sampled.count(it->index) > 0) {
            score = binomial.calc(current_observer_cnt2, current_neighbor_cnt2) / binomial.calc(observer_cnt, neighbor_cnt);
            it->updateAge(age_factor, score * batch_age);
        } else {
            score = (observer_cnt==current_observer_cnt) ? FloatType(1) : binomial.calc(current_observer_cnt, current_neighbor_cnt) / binomial.calc(observer_cnt, neighbor_cnt);
            it->updateAge(age_factor, score * batch_age);
        }        
    }
    updateGraph(
        current_e, // current_e or e
        age_factor,
        batch_age); // score
    last_time = time_data.back(); 
}


template<typename FloatType>
void tpSDOstreamclust<FloatType>::DFS(
        IndexSetType& cluster, 
        IndexSetType& processed, 
        const MapIterator& it) {
    // insert to sets
    if (!(processed.count(it->index)>0)) {
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
            auto additionalNeighbors = treeA.search(it->data, it->h , (zeta * it->h + (1 - zeta) * h));
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
}

template<typename FloatType>
void tpSDOstreamclust<FloatType>::updateH_all(bool use_median) { 
    if (use_median) {       
        std::priority_queue<FloatType, std::vector<FloatType>, std::less<FloatType>> maxHeap; 
        std::priority_queue<FloatType, std::vector<FloatType>, std::greater<FloatType>> minHeap;         
        for (auto it = observers.begin(); it != observers.end(); ++it) {  
            if (it->active) {     
                // add h to heaps 
                if (maxHeap.empty() || it->h <= maxHeap.top()) {
                    maxHeap.push(it->h);
                } else {
                    minHeap.push(it->h);
                }
                // Balance the heaps if their sizes differ by more than 1
                if (maxHeap.size() > (minHeap.size() + 1)) {
                    minHeap.push(maxHeap.top());
                    maxHeap.pop();
                } else if (minHeap.size() > (maxHeap.size() + 1)) {
                    maxHeap.push(minHeap.top());
                    minHeap.pop();
                } 
            }
        }        
        // Calculate the median based on the heap sizes and top elements    
        if (maxHeap.size() == minHeap.size()) {
            h = (maxHeap.top() + minHeap.top()) / 2.0f;
        } else if (maxHeap.size() > minHeap.size()) {
            h = maxHeap.top();
        } else {
            h = minHeap.top();
        }
    } else {
        FloatType s(0);
        int i(0);
        for (auto it = observers.begin(); it != observers.end(); ++it) {    
            if (it->active) {      
                s += it->h; 
                i++;
            }
        }
        h = s/i;
    }
}


template<typename FloatType>
void tpSDOstreamclust<FloatType>::DetermineColor(
        ClusterModelMap& clusters,
        FloatType age_factor, 
        FloatType score) {
    std::unordered_set<int> takenColors;
    auto it = clusters.begin();
    while (it != clusters.end()) {
        int color;
        if (it->color_score > 0) {
            color = it->color;        
            takenColors.insert(color);
            auto nextIt = it;
            ++nextIt;  // Create a temporary iterator pointing to the next element
            while (nextIt != clusters.end() && takenColors.find(nextIt->color) != takenColors.end()) {                
                auto node = clusters.extract(nextIt);            
                ClusterModel& cluster = node.value();
                cluster.setColor(takenColors);
                clusters.insert(std::move(node));
                nextIt = it;
                ++nextIt;
            }
        } else {
            color = ++last_color;
            it->setColor(color);
        }        
        const IndexSetType& cluster_observers = it->cluster_observers;
        for (const int& id : cluster_observers) {
            const MapIterator& it1 = indexToIterator[id];
            it1->updateColorObservations(color, age_factor, score);
        }
        ++it; // Increment the iterator to move to the next cluster
    }
}


template<typename FloatType>
void tpSDOstreamclust<FloatType>::updateGraph(
    std::size_t current_e,
    FloatType age_factor,
    FloatType score) {
        clusters.clear();
        IndexSetType processed;
        for (auto it = observers.begin(); it != observers.end(); ++it) {
            if (it->active) {                
                IndexSetType cluster;                              
                DFS(cluster, processed, it);    
                if (!(cluster.size() < current_e)) {  
                    ClusterModel clusterM(cluster, indexToIterator);
                    clusters.insert(clusterM); 
                }
            }
        }
        DetermineColor(clusters, age_factor, score);
}

template<typename FloatType>
void tpSDOstreamclust<FloatType>::Observer::updateColorDistribution() {
    // Calculate the sum of all color observations
    FloatType sum = std::accumulate(color_observations.begin(), color_observations.end(), FloatType(0),
        [](FloatType sum, const std::pair<int, FloatType>& entry) {
            return sum + entry.second;
        });
    // Update color distribution
    for (auto& entry : color_observations) {
        color_distribution[entry.first] = entry.second / sum;
    }
}

template<typename FloatType>
void tpSDOstreamclust<FloatType>::Observer::updateColorObservations(
        int colorObs,
        FloatType age_factor,
        FloatType score) {
    // Apply fading to all entries in color_observations
    for (auto& entry : color_observations) {
        entry.second *= age_factor; // fade of whole batch
    }
    color_observations[colorObs] += score; // age of batch
    // update dominant color
    if (color > 0) {
        if (color_observations[colorObs] > color_observations[color]) {
            color = colorObs;
        }
    } else {
        color = colorObs; // first Observation
    }
    // time_cluster_touched = now;
    updateColorDistribution();
};

#endif  // TPSDOSTREAMCLUST_GRAPH_H
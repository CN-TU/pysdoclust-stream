#ifndef SDOCLUSTSTREAM_GRAPH_H
#define SDOCLUSTSTREAM_GRAPH_H

#include "SDOcluststream_cluster.h"

template<typename FloatType>
void SDOcluststream<FloatType>::updateH_all(
    const size_t& chi) {        
    std::priority_queue<FloatType, std::vector<FloatType>, std::less<FloatType>> maxHeap; 
    std::priority_queue<FloatType, std::vector<FloatType>, std::greater<FloatType>> minHeap;         
    for (auto it = observers.begin(); it != observers.end(); ++it) {    
        if (!(it->active)) { break; }      
        updateH_single(it, chi); 
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
    // Calculate the median based on the heap sizes and top elements    
    if (maxHeap.size() == minHeap.size()) {
        h = (maxHeap.top() + minHeap.top()) / 2.0f;
    } else if (maxHeap.size() > minHeap.size()) {
        h = maxHeap.top();
    } else {
        h = minHeap.top();
    }
}


template<typename FloatType>
void SDOcluststream<FloatType>::DetermineColor(
        ClusterModelMap& clusters, 
        std::unordered_map<int, FloatType>& modelColorDistribution,
        FloatType now) {

    std::unordered_set<int> takenColors;

    auto it = clusters.begin();
    while (it != clusters.end()) {
        auto& color_distribution = it->color_distribution;
        
        for (const auto& pair: color_distribution) {
            modelColorDistribution[pair.first] += pair.second;
        }

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
            it1->updateColorObservations(color, now, fading_cluster);
            // it1->printColorDistribution();
            // it1->printColorObservations(now, fading_cluster);
        }

        ++it; // Increment the iterator to move to the next cluster
        }
}


template<typename FloatType>
void SDOcluststream<FloatType>::updateGraph(
    const FloatType& now,
    const int& active_threshold,
    const std::size_t current_e,
    const std::size_t& chi) {
        updateH_all(chi);
        // std::cout << std::endl << "global h: " << h << std::endl;
        clusters.clear();
        IndexSetType processed;
        for (auto it = observers.begin(); it != observers.end(); ++it) {
            if (!(it->active)) {
                break;
            }
            IndexSetType cluster;                              
            DFS(cluster, processed, it);    
            if (!(cluster.size() < current_e)) {  
                ClusterModel clusterM(cluster, indexToIterator);
                clusters.insert(clusterM); 
                // clusterM.printObserverIndices();
                // clusterM.printDistribution();
                // clusterM.printColor();
            }
        }
        modelColorDistribution.clear();
        DetermineColor(clusters, modelColorDistribution, now);
}

template<typename FloatType>
void SDOcluststream<FloatType>::Observer::updateColorDistribution() {
    // Calculate the sum of all color observations
    FloatType sum = std::accumulate(color_observations.begin(), color_observations.end(), FloatType(0),
        [](FloatType acc, const std::pair<int, FloatType>& entry) {
            return acc + entry.second;
        });
    // Update color distribution
    for (auto& entry : color_observations) {
        color_distribution[entry.first] = entry.second / sum;
    }
}

template<typename FloatType>
void SDOcluststream<FloatType>::Observer::updateColorObservations(
        int colorObs, 
        FloatType now, 
        FloatType fading_cluster) {
    // Apply fading to all entries in color_observations
    for (auto& entry : color_observations) {
        entry.second *= std::pow(fading_cluster, now - time_cluster_touched);
    }
    color_observations[colorObs] += 1;
    // update dominant color
    if (color > 0) {
        if (color_observations[colorObs] > color_observations[color]) {
            color = colorObs;
        }
    } else {
        color = colorObs; // first Observation
    }
    time_cluster_touched = now;
    updateColorDistribution();
};

#endif  // SDOCLUSTSTREAM_GRAPH_H
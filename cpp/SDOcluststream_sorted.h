#ifndef SDOCLUSTSTREAM_SORTED_H
#define SDOCLUSTSTREAM_SORTED_H

#include "SDOcluststream_util.h"


// Distance Matrix Structures
template<typename FloatType>
struct SDOcluststream<FloatType>::IndexDistancePair {
    // MapIterator it;
    int index; // Derived from it->index
    FloatType distance;

    IndexDistancePair() {} // Default constructor
    IndexDistancePair(int index, FloatType distance) : index(index), distance(distance) {}
    IndexDistancePair(MapIterator it, FloatType distance) : index(it->index), distance(distance) {}
};

template<typename FloatType>
struct SDOcluststream<FloatType>::DistanceCompare{
    bool operator()(const IndexDistancePair& a, const IndexDistancePair& b) const {
        if (a.distance == b.distance) {
            return a.index > b.index;
        }
        return a.distance < b.distance;
    }
};

template<typename FloatType>
struct SDOcluststream<FloatType>::TopKDistanceHeap {
    int k;
    DistanceCompare distance_compare;
    std::priority_queue<IndexDistancePair, std::vector<IndexDistancePair>, DistanceCompare> pq;

    // Constructor with initialization
    TopKDistanceHeap(int k, DistanceCompare distance_compare) : k(k), distance_compare(distance_compare), pq(distance_compare) {}

    void insert(const IndexDistancePair& element) {
        if (pq.size() < k) {
            pq.push(element);
        } else if (distance_compare(element, pq.top())) {
            pq.pop();
            pq.push(element);
        }
    }

    // Calculate median
    FloatType median() const {
        int size = pq.size();
        if (size == 0) {
            return 0; // No elements, then say 0
        } else {
            // Copy the priority queue to another temporary one
            std::priority_queue<IndexDistancePair, std::vector<IndexDistancePair>, DistanceCompare> pqCopy = pq;

            // Move the iterator to the middle element(s)
            for (int i = 0; i < size / 2; ++i) {
                pqCopy.pop();
            }

            // Calculate median
            if (size % 2 == 0) { // Even number of elements
                FloatType median1 = pqCopy.top().distance;
                pqCopy.pop();
                FloatType median2 = pqCopy.top().distance;
                return (median1 + median2) / 2.0;
            } else { // Odd number of elements
                return pqCopy.top().distance;
            }
        }
    }

    // Function to get all indices as an unordered_set
    std::unordered_set<int> getIndices() const {
        std::unordered_set<int> indices;
        std::priority_queue<IndexDistancePair, std::vector<IndexDistancePair>, DistanceCompare> pqCopy = pq;

        while (!pqCopy.empty()) {
            indices.insert(pqCopy.top().index);
            pqCopy.pop();
        }

        return indices;
    }

    void print() const {
        std::priority_queue<IndexDistancePair, std::vector<IndexDistancePair>, DistanceCompare> pqCopy = pq;
        while (!pqCopy.empty()) {
            auto topElement = pqCopy.top();
            std::cout << "(Index: " << topElement.index << ", Distance: " << topElement.distance << ") ";
            pqCopy.pop();
        }
        std::cout << std::endl;
    }
};

template<typename FloatType>
void SDOcluststream<FloatType>::updateH_single(
        MapIterator it, 
        size_t n) { 
    const DistanceMapType& distance_map = distance_matrix[it->index];
    if (distance_map.template get<1>().size() > n) {
        auto dIt = distance_map.template get<1>().begin();
        std::advance(dIt, n-1);
        it->h = dIt->distance;
    } else if (!distance_map.empty()) {
        auto dIt = distance_map.template get<1>().rbegin();
        it->h = dIt->distance;
    }
}

// template<typename FloatType>
// void SDOcluststream<FloatType>::DFS(
//         IndexSetType& cluster, 
//         IndexSetType& processed, 
//         const MapIterator& it) {
//     // insert to sets
//     processed.insert(it->index);   
//     cluster.insert(it->index);
//     const DistanceMapType& distance_map = distance_matrix[it->index];
//     for (auto dIt = distance_map.template get<1>().begin(); dIt != distance_map.template get<1>().end(); ++dIt) {            
//         FloatType distance = dIt->distance;
//         if (!hasEdge(distance, it)) { break; }
//         int idx = dIt->index;
//         if (!(processed.count(idx)>0)) {
//             const MapIterator& it1 = indexToIterator[dIt->index];
//             if (hasEdge(distance, it1)) {
//                 DFS(cluster, processed, it1);
//             }
//         }
//     }
// }

template<typename FloatType>
void SDOcluststream<FloatType>::updateDistanceMatrix(
        const std::unordered_set<int>& active,
        const std::unordered_set<int>& activated,
        const std::unordered_set<int>& deactivated) {
    // for (int idx : activated) {
    //     MapIterator it = indexToIterator[idx];
    //     Vector<FloatType> point = it->data;
    //     tree_active.insert(tree_active.begin(), std::make_pair(point, idx));
    // }
    // update distance matrix
    for (int idx : deactivated) {
        auto it = distance_matrix.find(idx);
        if (it != distance_matrix.end()) { // Not necessary that it exists, dropped Observer must not have been active
            distance_matrix.erase(it);
        }
    }
    for (auto& pair : distance_matrix) { // const??
        int idx = pair.first;
        DistanceMapType& distance_map = pair.second;
        for (int idy : deactivated) {
            auto it = distance_map.template get<0>().find(idy);
            if (it != distance_map.template get<0>().end()) {
                distance_map.template get<0>().erase(it);
            }
        }
        const MapIterator& it = indexToIterator[idx];
        Vector point = it->getData();
        for (int idy : activated) {
            if (idx != idy) {
                const MapIterator& it1 = indexToIterator[idy];
                distance_map.insert(IndexDistancePair(idy, distance_function(point, it1->getData())));
            }                    
        }
    }
    for (int idx: activated) {
        DistanceMapType distance_map;
        const MapIterator& it = indexToIterator[idx];
        Vector point = it->getData();
        for (int idy: active) {
            if (idx != idy) {
                const MapIterator& it1 = indexToIterator[idy];
                distance_map.insert(IndexDistancePair(idy, distance_function(point, it1->getData())));
            }                    
        }
        distance_matrix[idx] = distance_map;
    }
}

#endif  // SDOCLUSTSTREAM_SORTED_H
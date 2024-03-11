#ifndef SDOCLUSTSTREAM_H
#define SDOCLUSTSTREAM_H

#include <algorithm>
#include <boost/container/set.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/identity.hpp>
#include <queue>

#include "Vector.h"
#include "Gamma.h"
#include "MTree.h"

template<typename FloatType=double>
class SDOcluststream {
  public:
    typedef std::function<FloatType(const Vector<FloatType>&, const Vector<FloatType>&)> DistanceFunction;

  private:
  
    // number of observers we want
    std::size_t observer_cnt;
    // fraction of observers to consider active
    FloatType active_observers;
    // factor for deciding if a sample should be sampled as observer
    FloatType sampling_prefactor;
    // factor for deciding if a sample should be sampled as observer if model is empty
    FloatType sampling_first;
    // factor for exponential moving average
    FloatType fading;
    // factor for exponential moving average for cluster observations
    FloatType fading_cluster; // for now set to 1
    // number of nearest observers to consider
    std::size_t neighbor_cnt;
    // number of nearest observer relative to active_observers
    FloatType p_outlier;
    // quantile of scores that is targeted to be identified as an outlier
    
    std::vector<FloatType> obs_scaler;

    // counter of processed samples
    int last_index;
    // counter index when we sampled the last time
    int last_added_index;
    // time when we last sampled
    FloatType last_added_time;

    DistanceFunction distance_function;
    std::mt19937 rng;

    std::size_t chi_min;
    FloatType chi_prop;
    FloatType zeta;
    FloatType h; // global h (median of all h)
    std::size_t e; // unused by now

    int last_color;

    typedef MTree< Vector<FloatType>, int, FloatType, MTreeDescendantCounter<Vector<FloatType>,int> > Tree;
    typedef typename Tree::iterator TreeIterator;
    typedef std::vector<std::pair<TreeIterator, FloatType>> TreeNeighbors;

    class TreeNodeUpdater {
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

    // Observer Structures
    struct Observer {
        Vector<FloatType> data;
        FloatType observations;
        FloatType time_touched;        
        FloatType time_added;
        int index;
        bool active;
        // TreeIterator treeIt;
        // TreeIterator treeIt_active;
        

        FloatType time_cluster_touched;
        int color;
        std::unordered_map<int, FloatType> color_observations; // color, score
        std::unordered_map<int, FloatType> color_distribution; // color, score normalized

        FloatType h;

        // Constructor for Observer
        Observer(
            Vector<FloatType> data,
            FloatType observations,
            FloatType time_touched,
            FloatType time_added,
            int index
            // Tree* tree // should contain index and data soon
        ) : data(data),
            observations(observations),
            time_touched(time_touched),
            time_added(time_added),
            index(index),
            active(false),
            // treeIt(),
            time_cluster_touched(time_touched),
            color(0),
            color_observations(),
            color_distribution(),
            h(0) {
                // treeIt = tree->insert(tree->begin(), std::make_pair(data, index)); 
            }
        
        int getIndex() {
            return index;
        }

        Vector<FloatType> getData() {
            return data;
        }

        // TreeIterator deactivate() {
        //     active = false;
        //     return treeIt_active;
        // }
        // void activate(TreeIterator _treeIt_active, Tree* tree) {
        //     TreeNodeUpdater updater(data, index);
        //     if (_treeIt_active != tree->end()) {
        //         tree->modify(_treeIt_active, updater);
        //     } else {
        //         std::cout << std::endl << " SOME MESSAGE !!!!!" << std::endl;
        //     }
        //     active = true;
        // }
        // void activate(Tree* tree) {
        //     treeIt_active = tree->insert(tree->begin(), std::make_pair(data, index)); 
        //     active = true;
        // }

        void reset(
            Vector<FloatType> _data,
            FloatType _observations,
            FloatType _time_touched,
            FloatType _time_added,
            int _index
            // Tree* tree
        ) {
            data = _data;
            observations = _observations;
            time_touched = _time_touched;
            time_added = _time_added;
            time_cluster_touched = _time_touched;
            index = _index;
            active = false;
            color_observations.clear();
            color_distribution.clear();
            color = 0;
            h = 0;

            // TreeNodeUpdater updater(_data, _index);
            // tree->modify(treeIt, updater);
        }

        void updateColorDistribution() {
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

        void printColorDistribution() {
            std::cout << std::endl << "Color Distribution Observer: " << index << ": ";
            for (auto& entry : color_distribution) {
                std::cout << "(" << entry.first << "," << entry.second << ") ";
            }
            std::cout << std::endl;
        }

        void updateColorObservations(
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
        }

        void printColorObservations(
                FloatType now, 
                FloatType fading_cluster) {

            std::cout << std::endl << "Color Observations Observer: " << index << ": ";
            for (auto& entry : color_observations) {
                std::cout << "(" << entry.first << "," << entry.second * std::pow(fading_cluster, now - time_cluster_touched) << ") ";
            }
            std::cout << std::endl;
        }

        void printData() const {
            std::cout << "[ ";
            for (const auto& value : data) {
                std::cout << value << " ";
            }
            std::cout << "]";
        }
    };

    struct ObserverCompare{
        FloatType fading;

        // ObserverCompare() : fading(1.0) {}
        ObserverCompare(FloatType fading) : fading(fading) {}

        bool operator()(const Observer& a, const Observer& b) const {
            FloatType common_touched = std::max(a.time_touched, b.time_touched);
            
            FloatType observations_a = a.observations
                * std::pow(fading, common_touched - a.time_touched);
            
            FloatType observations_b = b.observations
                * std::pow(fading, common_touched - b.time_touched);
            
            // tie breaker for reproducibility
            if (observations_a == observations_b)
                return a.index > b.index;
            return observations_a > observations_b;
        }
    } observer_compare;
    
    struct ObserverAvCompare{
        FloatType fading;
        ObserverAvCompare(FloatType fading) : fading(fading) {}
        bool operator()(FloatType now, const Observer& a, const Observer& b) {
            FloatType common_touched = std::max(a.time_touched, b.time_touched);
            
            FloatType observations_a = a.observations * std::pow(fading, common_touched - a.time_touched);
            FloatType age_a = 1-std::pow(fading, now-a.time_added);
            
            FloatType observations_b = b.observations * std::pow(fading, common_touched - b.time_touched);
            FloatType age_b = 1-std::pow(fading, now-b.time_added);
            
            // do not necessarily need a tie breaker here
            return observations_a * age_b > observations_b * age_a;
        }
    } observer_av_compare;

    typedef boost::container::multiset< Observer, ObserverCompare > MapType;
    typedef typename MapType::iterator MapIterator;
    MapType observers;

    struct IteratorAvCompare{
        FloatType fading;
        FloatType now;
        IteratorAvCompare(FloatType fading, FloatType now) : fading(fading), now(now) {}
        bool operator()(const MapIterator& it_a, const MapIterator& it_b) {
            const Observer& a = *it_a;
            const Observer& b = *it_b;
            FloatType common_touched = std::max(a.time_touched, b.time_touched);
            
            FloatType observations_a = a.observations * std::pow(fading, common_touched - a.time_touched);
            FloatType age_a = 1-std::pow(fading, now-a.time_added);
            
            FloatType observations_b = b.observations * std::pow(fading, common_touched - b.time_touched);
            FloatType age_b = 1-std::pow(fading, now-b.time_added);
            
            // do not necessarily need a tie breaker here
            return observations_a * age_b > observations_b * age_a;
        }
    };

    // Index Iterator Map Structure
    typedef std::unordered_map<int,MapIterator> IteratorMapType;
    IteratorMapType indexToIterator;
    typedef std::unordered_set<int> IndexSetType;

    struct ClusterModel {
        int color;
        FloatType color_score; // score of set color
        IndexSetType cluster_observers;
        std::unordered_map<int, FloatType> color_distribution;

        ClusterModel() {} // Default constructor

        ClusterModel(const IndexSetType& cluster_observers, const IteratorMapType& indexToIterator) : color(0), color_score(FloatType(0)), cluster_observers(cluster_observers), color_distribution() {
            calcColorDistribution(indexToIterator);
            setColor();
        }

        // Member function to calculate color distribution
        void calcColorDistribution(const IteratorMapType& indexToIterator) {
            // Clear existing color distribution
            color_distribution.clear();

            // Iterate over observers and accumulate color distributions
            for (const int& id: cluster_observers) {
                auto iIt = indexToIterator.find(id);
                if (iIt != indexToIterator.end()) {
                    const MapIterator& it = iIt->second;   
                    const Observer& observer = *it; // Dereference the iterator to get the Observer

                    // Add color_distribution of the observer to colorDistribution
                    for (const auto& entry : observer.color_distribution) {
                        color_distribution[entry.first] += entry.second;
                    }
                } else {
                    std::cerr << "Error (calcColorDistribution): id " << id << " not found in indexToIterator" << std::endl;
                }                
            }
        }

        void printDistribution() const {
            std::cout << "Cluster Distribution: " << std::endl;
            for (const auto& entry : color_distribution) {
                std::cout << "(" << entry.first << ", " << entry.second << ") ";
            }
            std::cout << std::endl;
        }

        void printColor() const {
            // Print color and color_score
            std::cout << std::endl << "Color: " << color << ", Score: " << color_score << std::endl;
        }

        void printObserverIndices() const {
            std::cout << "Cluster Indices: " << std::endl;
            for (const int& id : cluster_observers) {
                std::cout << id << " ";
            }
            std::cout << std::endl;
        }

        void setColor() {
            if (!color_distribution.empty()) {
                // Find the iterator with the maximum value in color_distribution
                auto maxIt = std::max_element(color_distribution.begin(), color_distribution.end(),
                    [](const auto& a, const auto& b) {
                        if (a.second == b.second) {
                            return a.first > b.first;
                        }
                        return a.second < b.second;
                    });

                // Set the color to the key with the maximum value
                color = maxIt->first;
                color_score = color_distribution[color];
            } 
        }

        void setColor(
                const std::unordered_set<int>& takenColors) {

            color = 0;
            color_score = FloatType(0);

            if (!color_distribution.empty()) {
                // Find the keys in color_distribution that are not in takenColors
                std::unordered_map<int, FloatType> difference;
                std::copy_if(
                    color_distribution.begin(), color_distribution.end(),
                    std::inserter(difference, difference.begin()),
                    [&takenColors](const auto& entry) {
                        return takenColors.find(entry.first) == takenColors.end();
                    }
                );

                // Check if the difference is non-empty
                if (!difference.empty()) {
                    // Find the iterator with the maximum value in colorDistribution,
                    // excluding colors in the difference set
                    auto maxIt = std::max_element(difference.begin(), difference.end(),
                        [](const auto& a, const auto& b) {
                            return (a.second == b.second) ? (a.first > b.first) : (a.second < b.second);
                        });

                    // Set the color to the key with the maximum value
                    color = maxIt->first;
                    color_score = color_distribution[color];
                }
            }
        }

        void setColor(
                int c) {

            color = c;
            color_score = FloatType(0);
        }
    };

    struct ClusterModelCompare {
        bool operator()(const ClusterModel& CM_a, const ClusterModel& CM_b) const {
            return (CM_a.color_score == CM_b.color_score) ? 
                CM_a.cluster_observers.size() > CM_b.cluster_observers.size() :
                CM_a.color_score > CM_b.color_score;
        }
    };

    typedef boost::container::multiset<ClusterModel,ClusterModelCompare> ClusterModelMap;
    
    ClusterModelMap clusters;
    std::unordered_map<int, FloatType>  modelColorDistribution;
   

    // Distance Matrix Structures
    struct IndexDistancePair {
        // MapIterator it;
        int index; // Derived from it->index
        FloatType distance;

        IndexDistancePair() {} // Default constructor
        IndexDistancePair(int index, FloatType distance) : index(index), distance(distance) {}
        IndexDistancePair(MapIterator it, FloatType distance) : index(it->index), distance(distance) {}
    };

    struct DistanceCompare{
        bool operator()(const IndexDistancePair& a, const IndexDistancePair& b) const {
            if (a.distance == b.distance) {
                return a.index > b.index;
            }
            return a.distance < b.distance;
        }
    } distance_compare;

    typedef boost::multi_index::multi_index_container<
        IndexDistancePair,
        boost::multi_index::indexed_by<
            // boost::multi_index::hashed_unique< // probably not ideal for small datasize
            boost::multi_index::ordered_unique<
                boost::multi_index::member<IndexDistancePair, int, &IndexDistancePair::index>
            >,
            boost::multi_index::ordered_unique<
                boost::multi_index::identity<IndexDistancePair>,
                DistanceCompare
            >
        >
    > DistanceMapType;  
    
    typedef std::unordered_map<int, DistanceMapType> DistanceMatrix;
    DistanceMatrix distance_matrix;
    
    //TODO: make it two heaps for better median
    struct TopKDistanceHeap {
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

    Gamma<FloatType> gamma_dist;

    Tree tree;
    Tree tree_active; 
    // typedef std::pair<const Key, T> ValueType;
    struct MyTieBreaker {
        bool operator() (const typename Tree::ValueType& a, const typename Tree::ValueType& b) { return a.second > b.second; }
    };
    

    void printClusters() {
        for (const auto& cluster : clusters) {
            cluster.printColor();
            cluster.printObserverIndices();            
            cluster.printDistribution();
        }
    }

    void printDistanceMatrix() {
        std::cout << std::endl << "Distance Matrix" << std::endl;
        for (const auto& entry : distance_matrix) {
            std::cout << "[" << entry.first << "]: ";
            const DistanceMapType& distance_map = entry.second;

            for (const auto& item : distance_map.template get<1>()) {
                std::cout << "(" << item.index << ", " << item.distance << ") ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for (const auto& entry : distance_matrix) {
            std::cout << "[" << entry.first << "]: ";
            const DistanceMapType& distance_map = entry.second;

            for (const auto& item : distance_map.template get<0>()) {
                std::cout << "" << item.index << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void printObservers(
        FloatType now) {

        std::cout << std::endl << "Observers" << std::endl;
        for (const auto& observer : observers) {
            FloatType pow_fading = std::pow(fading, now - observer.time_touched);
            FloatType age = 1-std::pow(fading, now - observer.time_added);
            std::cout << "(" << observer.index 
                    << ", " << observer.observations 
                    << ", " << observer.observations * pow_fading 
                    << ", " << observer.observations * pow_fading / age
                    << ", " << observer.time_added
                    << ", " << observer.time_touched << ") ";
            // observer.printData();
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void setModelParameters(
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

    void DetermineColor(
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

    void setObsScaler() {

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

    void updateH_single(
            MapIterator it, 
            size_t n) {          
        // Vector<FloatType> point(it->getData());        
        // n+1 because closest is point itself
        // auto nearestNeighbors = tree_active.knnSearch(point, n+1, true, 0, std::numeric_limits<FloatType>::infinity(), false, false);
        // it->h = nearestNeighbors[n].second; // ordered and reversed, in first is the TreeIterator 
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

    void updateH_all(
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

    bool hasEdge(
            FloatType distance, 
            const MapIterator& it) {
        return distance < (zeta * it->h + (1 - zeta) * h);
    }
    
    void DFS(
            IndexSetType& cluster, 
            IndexSetType& processed, 
            const MapIterator& it) {
        // insert to sets
        processed.insert(it->index);   
        cluster.insert(it->index);
        const DistanceMapType& distance_map = distance_matrix[it->index];
        for (auto dIt = distance_map.template get<1>().begin(); dIt != distance_map.template get<1>().end(); ++dIt) {            
            FloatType distance = dIt->distance;
            if (!hasEdge(distance, it)) { break; }
            int idx = dIt->index;
            if (!(processed.count(idx)>0)) {
                const MapIterator& it1 = indexToIterator[dIt->index];
                if (hasEdge(distance, it1)) {
                    DFS(cluster, processed, it1);
                }
            }
        }
        // look for neighbors
        // Vector<FloatType> point = it->getData();
        // FloatType radius = zeta * it->h + (1 - zeta) * h;
        // typename Tree::RangeQuery rangeNeighbors = tree_active.rangeSearch(point, radius);
        // while (!rangeNeighbors.atEnd()) {
        //     // Dereference the iterator to get the current element
        //     auto neighbor = *rangeNeighbors;
        //     int idx = neighbor.first->second;
        //     if (!(processed.count(idx)>0)) {
        //         FloatType distance = neighbor.second;
        //         MapIterator it1 = indexToIterator[idx];
        //         if (hasEdge(distance, it1)) {
        //             DFS(cluster, processed, it1);
        //         }
        //     }
        //     ++rangeNeighbors;
        // }
    }

    void fit_impl(
            std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
            const Vector<FloatType>& point,
            const FloatType& now,
            const int& observer_index,            
            const int& current_observer_cnt,
            const int& current_neighbor_cnt) {        
        auto nearestNeighbors = tree.knnSearch(point, current_neighbor_cnt + 1); // one more cause point is Observer
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
    }

    void fit_impl(
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
    }

    void predict_impl(
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
    }

    void predict_impl(
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
    }


    void updateModel(
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

    bool sampleData( 
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
        

    void sampleData(
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
    }

    void replaceObservers(
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

    void updateGraph(
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

    void updateDistanceMatrix(
        const std::unordered_set<int>& active,
        const std::unordered_set<int>& activated,
        const std::unordered_set<int>& deactivated) {
            for (int idx : activated) {
                MapIterator it = indexToIterator[idx];
                Vector<FloatType> point = it->data;
                tree_active.insert(tree_active.begin(), std::make_pair(point, idx));
            }
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

    std::vector<int> fitPredict_impl(
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
    }

public:
    SDOcluststream(
        std::size_t observer_cnt, 
        FloatType T, 
        FloatType idle_observers, 
        std::size_t neighbor_cnt,
        std::size_t chi_min,
        FloatType chi_prop,
        FloatType zeta,
        std::size_t e,
        FloatType p_outlier,
        SDOcluststream<FloatType>::DistanceFunction distance_function = Vector<FloatType>::euclidean, 
        int seed = 0
    ) : observer_cnt(observer_cnt), 
        active_observers(1-idle_observers), 
        sampling_prefactor(observer_cnt * observer_cnt / neighbor_cnt / T),
        sampling_first(observer_cnt / T),
        fading(std::exp(-1/T)),
        fading_cluster(FloatType(1)),
        neighbor_cnt(neighbor_cnt),
        p_outlier(p_outlier),
        obs_scaler(observer_cnt+1),
        last_index(0),
        last_added_index(0),
        distance_function(distance_function),
        rng(seed),
        chi_min(chi_min),
        chi_prop(chi_prop),
        zeta(zeta),
        h(FloatType(0)),
        e(e),
        last_color(0),
        observer_compare(fading),        
        observer_av_compare(fading),
        observers(observer_compare),  // Initialize observers container with initial capacity and comparison function
        clusters(),
        modelColorDistribution(),
        distance_compare(),
        distance_matrix(),
        gamma_dist(),
        tree(distance_function),
        tree_active(distance_function)
    {
        setObsScaler();
    }

    // TO DO
    void fit(const std::vector<Vector<FloatType>>& data, const std::vector<FloatType>& time_data) {
        fitPredict_impl(data, time_data, true);
    }

    std::vector<int> fitPredict(
            const std::vector<Vector<FloatType>>& data, 
            const std::vector<FloatType>& time_data) {
        if (true) {
            std::normal_distribution<FloatType> distribution(0.0, 1e-9);  // Adjust sigma as needed

            // Add noise efficiently
            std::vector<Vector<FloatType>> noisy_data = data;
            const std::size_t data_size = data.size();
            const std::size_t vec_size = data[0].size();

            // std::cout << std::endl << data_size << " " << vec_size << std::endl;
            for (std::size_t i = 0; i < data_size; ++i) {
                for (std::size_t j = 0; j < vec_size; ++j) {
                    noisy_data[i][j] += distribution(rng);  // Generate noise for each dimension
                }
            }
            return fitPredict_impl(noisy_data, time_data, false);
        }
        return fitPredict_impl(data, time_data, false);
    }
    
    int observerCount() { return observers.size(); }
    
    bool lastWasSampled() { return last_added_index == last_index - 1; }

    class ObserverView{
        FloatType fading;
        MapIterator it;
    public:
        ObserverView(FloatType fading, MapIterator it) :
            fading(fading),
            it(it)
        { }
        // int getIndex() {return it->index};
        Vector<FloatType> getData() { return it->data; }
        int getColor() { return it->color; }
        FloatType getObservations(FloatType now) {
            return it->observations * std::pow(fading, now - it->time_touched);
        }
        FloatType getAvObservations(FloatType now) {
            return (1-fading) * it->observations * std::pow(fading, now - it->time_touched) /
                (1-std::pow(fading, now - it->time_added));
        }
    };

    class iterator : public MapIterator {
        FloatType fading;
      public:
        ObserverView operator*() { return ObserverView(fading, MapIterator(*this)); };
        iterator() {}
        iterator(FloatType fading, MapIterator it) : 
            MapIterator(it),
            fading(fading)
        { }
    };

    iterator begin() { return iterator(fading, observers.begin()); }
    iterator end() { return iterator(fading, observers.end()); }
};                              

#endif  // SDOCLUSTSTREAM_H

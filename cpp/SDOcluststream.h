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
// #include "Gamma.h"
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
    FloatType k_tanh;
    // tanh(( k_tanh * (outlier_threshold-1)) = 0.5 where outlier_threshold is a factor of h_bar(Observer)
    
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
    class TreeNodeUpdater; // tree
    struct MyTieBreaker; // tree

    // Observer Structures
    struct Observer;
    struct ObserverCompare;
    ObserverCompare observer_compare;    
    struct ObserverAvCompare;
    ObserverAvCompare observer_av_compare;

    typedef boost::container::multiset< Observer, ObserverCompare > MapType;
    typedef typename MapType::iterator MapIterator;
    MapType observers;
    struct IteratorAvCompare;

    // Index Iterator Map Structure
    typedef std::unordered_map<int,MapIterator> IteratorMapType;
    IteratorMapType indexToIterator;
    typedef std::unordered_set<int> IndexSetType;

    // Cluster Declarations
    struct ClusterModel;
    struct ClusterModelCompare;
    typedef boost::container::multiset<ClusterModel,ClusterModelCompare> ClusterModelMap;    
    ClusterModelMap clusters;
    std::unordered_map<int, FloatType>  modelColorDistribution;

    Tree tree;
    Tree treeA; 
    
    void printClusters(); // print
    void printDistanceMatrix(); // print
    void printObservers(FloatType now); // print

    void setModelParameters(
            int& current_observer_cnt, int&current_observer_cnt2,
            int& active_threshold, int& active_threshold2,
            int& current_neighbor_cnt, int& current_neighbor_cnt2,
            std::size_t& current_e,
            std::size_t& chi,
            bool print); // util
    void DetermineColor(
            ClusterModelMap& clusters, 
            std::unordered_map<int, FloatType>& modelColorDistribution,
            FloatType now); // graph

    void setObsScaler(); // util

    void updateH_single(MapIterator it, size_t n);
    void updateH_all(const size_t& chi);
    void updateH_all();
    
    bool hasEdge(FloatType distance, const MapIterator& it); // util
    
    void DFS(IndexSetType& cluster, IndexSetType& processed, const MapIterator& it); // sorted or tree

    void fit_impl(
            std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
            const Vector<FloatType>& point,
            const FloatType& now,           
            const int& current_observer_cnt,
            const int& current_neighbor_cnt,
            const int& observer_index); // tree
    void fit_impl(
            std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
            const Vector<FloatType>& point,
            const FloatType& now,           
            const int& current_observer_cnt,
            const int& current_neighbor_cnt); // tree

    void determineLabelVector(
            std::unordered_map<int, FloatType>& label_vector,
            const std::pair<TreeIterator, FloatType>& neighbor);
    void predict_impl(
            int& label,
            const Vector<FloatType>& point, // could be accessed as with observer_index
            const int& current_neighbor_cnt,
            const int& observer_index); // tree
    void predict_impl(
            int& label,
            const Vector<FloatType>& point,
            const int& current_neighbor_cnt); // tree
    void updateModel(
            const std::unordered_map<int,std::pair<FloatType, FloatType>>& temporary_scores); // util

    bool sampleData( 
        std::unordered_set<int>& sampled,
        const FloatType& now,
        const int& batch_size, // actually batch size - 1
        const FloatType& batch_time,
        const int& current_index); // util

    void sampleData(
            std::unordered_set<int>& sampled,
            const Vector<FloatType>& point,
            const FloatType& now,
            FloatType observations_sum,
            const int& current_observer_cnt,
            const int& current_neighbor_cnt,
            const int& current_index); // util

    void replaceObservers(
            Vector<FloatType> data,
            std::unordered_set<int>& dropped,
            std::priority_queue<MapIterator,std::vector<MapIterator>,IteratorAvCompare>& worst_observers,
            const FloatType& now,
            const int& current_observer_cnt,
            const int& current_index); // util

    void updateGraph(
        const FloatType& now,
        const int& active_threshold,
        const std::size_t current_e,
        const std::size_t& chi); // util

    std::vector<int> fitPredict_impl(
        const std::vector<Vector<FloatType>>& data, 
        const std::vector<FloatType>& time_data, 
        bool fit_only); //tree

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
        FloatType outlier_threshold,
        SDOcluststream<FloatType>::DistanceFunction distance_function = Vector<FloatType>::euclidean, 
        int seed = 0
    ) : observer_cnt(observer_cnt), 
        active_observers(1-idle_observers), 
        sampling_prefactor(observer_cnt * observer_cnt / neighbor_cnt / T),
        sampling_first(observer_cnt / T),
        fading(std::exp(-1/T)),
        fading_cluster(FloatType(1)),
        neighbor_cnt(neighbor_cnt),
        k_tanh( atanh(0.5f) / (outlier_threshold-1) ),
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
        // distance_compare(),
        // distance_matrix(),
        // gamma_dist(),
        tree(distance_function),
        treeA(distance_function)
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

#include "SDOcluststream_tree.h"

#endif  // SDOCLUSTSTREAM_H

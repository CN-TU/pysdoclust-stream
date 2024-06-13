#ifndef SDOSTREAMCLUST_H
#define SDOSTREAMCLUST_H

#include <algorithm>
#include <boost/container/set.hpp>
#include <cmath>
#include <functional>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <queue>

#include "Vector.h"
#include "MTree.h"

template<typename FloatType=double>
class SDOstreamclust {
  private:
    typedef std::pair<Vector<FloatType>, FloatType> Point; // data, epsilon
  public:
    typedef std::function<FloatType(const Point&, const Point&)> DistanceFunction;
  private:
  
    // number of observers we want
    std::size_t observer_cnt;
    // fraction of observers to consider active
    FloatType active_observers;
    // factor for deciding if a sample should be sampled as observer
    FloatType sampling_prefactor;
    // factor for exponential moving average
    FloatType fading;
    // number of nearest observers to consider
    std::size_t neighbor_cnt;

    // tanh(( k_tanh * (outlier_threshold-1)) = 0.5 where outlier_threshold is a factor of h_bar(Observer)
    FloatType k_tanh;
    // flag for outlier handling
    bool outlier_handling;
    // relative or absolute distance for outlier score
    bool rel_outlier_score;

    // scaling of perturb factor for epsilon
    FloatType perturb;

    // flag for random Sampling 
    bool random_sampling;
    // input buffer
    std::size_t input_buffer;
    class DataBuffer;
    DataBuffer buffer;

    // for calculating observations scores before model is full
    // in util defined
    class BinomialCalculator;
    BinomialCalculator binomial;

    // counter of processed samples
    int last_index;    
    // counter index when we sampled the last time
    int last_added_index;
    // time when last processed
    FloatType last_time;
    // time when we last sampled
    FloatType last_added_time;
    // counter of predicted samples
    int last_predicted_index;

    DistanceFunction distance_function;
    std::mt19937 rng;

    // chi defines the threshold for cutting edges in the graph
    std::size_t chi_min;
    FloatType chi_prop;
    // weight between global and local density value h per observer
    FloatType zeta;
    // global density value h
    FloatType h;
    // min cluster size in observer model
    std::size_t e;

    // counter of used labels 
    int last_color;

    // MTrees for efficient knn
    typedef MTree< Point, int, FloatType, MTreeDescendantCounter<Point,int> > Tree;
    typedef typename Tree::iterator TreeIterator;
    typedef std::vector<std::pair<TreeIterator, FloatType>> TreeNeighbors;

    // observer
    struct Observer;
    struct ObserverCompare;
    ObserverCompare observer_compare;    

    typedef boost::container::multiset< Observer, ObserverCompare > MapType;
    typedef typename MapType::iterator MapIterator;
    MapType observers;
    struct IteratorAvCompare;

    // Index Iterator Map Structure
    typedef std::unordered_map<int,MapIterator> IteratorMapType;
    IteratorMapType indexToIterator;
    typedef std::unordered_set<int> IndexSetType;

    // cluster
    struct ClusterModel;
    struct ClusterModelCompare;
    typedef boost::container::multiset<ClusterModel,ClusterModelCompare> ClusterModelMap;    
    ClusterModelMap clusters;

    // Tree of all Observers
    Tree tree;
    // Tree of active Observers
    Tree treeA; 
    
    // print
    // void printClusters(); 
    // void printDistanceMatrix(); 
    // void printObservers(FloatType now);

    // util
    bool hasEdge(FloatType distance, const MapIterator& it);
    FloatType calcBatchAge(const std::vector<FloatType>& time, FloatType score = 1);
    // void setObsScaler();
    void setModelParameters(
            std::size_t& current_observer_cnt, std::size_t&current_observer_cnt2,
            std::size_t& active_threshold, std::size_t& active_threshold2,
            std::size_t& current_neighbor_cnt, std::size_t& current_neighbor_cnt2,
            std::size_t& current_e,
            std::size_t& chi,
            bool print); 

    // fit
    void fit_impl(
            const std::vector<Vector<FloatType>>& data,
            const std::vector<FloatType>& epsilon,
            const std::vector<FloatType>& time,
            int first_index);
    void fit_point(
            std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
            const Point& point,
            FloatType now,           
            std::size_t current_observer_cnt,
            std::size_t current_neighbor_cnt,
            int observer_index); // point was sampled
    void fit_point(
            std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
            const Point& point,
            FloatType now,           
            std::size_t current_observer_cnt,
            std::size_t current_neighbor_cnt); // point was not sampled
    void update_model(
            const std::unordered_map<int,std::pair<FloatType, FloatType>>& temporary_scores);

    // predict
    void predict_impl(
            std::vector<int>& labels,
            std::vector<FloatType>& score,
            const std::vector<Vector<FloatType>>& data,
            const std::vector<FloatType>& epsilon,
            int first_index);    
    void predict_impl(
            std::vector<int>& labels,
            std::vector<FloatType>& score,
            const std::vector<Vector<FloatType>>& data,
            const std::vector<FloatType>& epsilon);
    void determineLabelVector(
            std::unordered_map<int, FloatType>& label_vector,        
            std::vector<FloatType>& score_vector,
            const std::pair<TreeIterator, FloatType>& neighbor);
    void setLabel(
            int& label,
            const std::unordered_map<int, FloatType>& label_vector,
            std::size_t current_neighbor_cnt);
    void predict_point(
            int& label,
            FloatType& score,
            std::size_t current_neighbor_cnt,
            int observer_index);  // point was sampled
    void predict_point(
            int& label,
            FloatType& score,
            const Point& point,
            std::size_t current_neighbor_cnt); // point was not sampled
    
    // sample
    void sample(
            std::unordered_set<int>& sampled,
            const std::vector<Vector<FloatType>>& data,
            const std::vector<FloatType>& epsilon,
            const std::vector<FloatType>& time,
            int first_index);
    bool sample_point( 
            std::unordered_set<int>& sampled,
            FloatType now,
            std::size_t batch_size,
            FloatType batch_time,
            int current_index); // random sampling
    void sample_point(
            std::unordered_set<int>& sampled,
            const Point& point,
            FloatType now,
            FloatType observations_sum,
            std::size_t current_observer_cnt,
            std::size_t current_neighbor_cnt,
            int current_index); // importance sampling
    void replaceObservers(
            Point data,
            std::priority_queue<MapIterator,std::vector<MapIterator>,IteratorAvCompare>& worst_observers,
            FloatType now,
            std::size_t current_observer_cnt,
            std::size_t current_neighbor_cnt,
            int current_index);

    // graph
    void update(
            const std::vector<FloatType>& time,
            const std::unordered_set<int>& sampled);
    void updateGraph(
            std::size_t current_e,
            FloatType age_factor,
            FloatType score);
    void DFS(IndexSetType& cluster, IndexSetType& processed, const MapIterator& it);
    void updateH_all(bool use_median = false);
    void DetermineColor(
            ClusterModelMap& clusters,
            FloatType age_factor, 
            FloatType score);

    // fitpredict
    void fitPredict_impl(
            std::vector<int>& label,
            std::vector<FloatType>& score,
            const std::vector<Vector<FloatType>>& data,
            const std::vector<FloatType>& epsilon,
            const std::vector<FloatType>& time); 
    void fitOnly_impl(
            const std::vector<Vector<FloatType>>& data,
            const std::vector<FloatType>& epsilon,
            const std::vector<FloatType>& time); 
    void predictOnly_impl(
            std::vector<int>& label,
            std::vector<FloatType>& score,
            const std::vector<Vector<FloatType>>& data,
            const std::vector<FloatType>& epsilon,
            const std::vector<FloatType>& time); 

public:
    SDOstreamclust(
        std::size_t observer_cnt, 
        FloatType T, 
        FloatType idle_observers, 
        std::size_t neighbor_cnt,
        std::size_t chi_min,
        FloatType chi_prop,
        FloatType zeta,
        std::size_t e,
        FloatType outlier_threshold,
        bool outlier_handling = false,
        bool rel_outlier_score = true,
        FloatType perturb = 0,
        bool random_sampling = true,
        std::size_t input_buffer = 0,
        SDOstreamclust<FloatType>::DistanceFunction distance_function = Vector<FloatType>::euclideanE, 
        int seed = 0
    ) : observer_cnt(observer_cnt), 
        active_observers(1-idle_observers), 
        sampling_prefactor(observer_cnt / T),
        fading(std::exp(-1/T)),
        neighbor_cnt(neighbor_cnt),
        k_tanh( (!outlier_handling) ? 0 : atanh(0.5f) / (outlier_threshold-1) ),
        outlier_handling(outlier_handling),
        perturb(perturb),
        random_sampling(random_sampling),
        input_buffer(input_buffer),    
        buffer(input_buffer),   
        binomial(observer_cnt,neighbor_cnt),
        last_index(0),
        last_added_index(0),
        last_time(0),
        last_added_time(0),
        last_predicted_index(0),
        distance_function(distance_function),
        rng(seed),
        chi_min(chi_min),
        chi_prop(chi_prop),
        zeta(zeta),
        h(FloatType(0)),
        e(e),
        last_color(0),
        observer_compare(fading),   
        observers(observer_compare),  // Initialize observers container with initial capacity and comparison function
        clusters(),
        tree(distance_function),
        treeA(distance_function)
    {}

    void fit(
            const std::vector<Vector<FloatType>>& data, 
            const std::vector<FloatType>& time) {        
        std::vector<FloatType> epsilon(data.size(), 0.0);
        if (perturb > 0) {
            std::uniform_real_distribution<FloatType> distribution(-perturb, perturb);
            std::generate(epsilon.begin(), epsilon.end(), [&distribution, this] () {
                return distribution(rng);
            });
        }
        fitOnly_impl(data, epsilon, time);
    }

    void fitPredict(
            std::vector<int>& label, 
            std::vector<FloatType>& score,
            const std::vector<Vector<FloatType>>& data, 
            const std::vector<FloatType>& time) {
        std::vector<FloatType> epsilon(data.size(), 0.0);
        if (perturb > 0) {
            std::uniform_real_distribution<FloatType> distribution(-perturb, perturb);
            std::generate(epsilon.begin(), epsilon.end(), [&distribution, this] () {
                return distribution(rng);
            });
        }
        if (input_buffer>0 && !(observers.empty() && (data.size()<input_buffer))) {
            std::size_t i(0);
            if ((buffer.size > 0) && ((buffer.size + data.size()) > input_buffer)) {
                std::vector<Vector<FloatType>> chunk_data(input_buffer);
                std::vector<FloatType> chunk_time(input_buffer);
                std::vector<FloatType> chunk_epsilon(input_buffer);
                i = input_buffer - buffer.size;
                buffer.flush(chunk_data, chunk_epsilon, chunk_time);

                std::copy(data.begin(), data.begin() + i, chunk_data.begin() + input_buffer - i);
                std::copy(epsilon.begin(), epsilon.begin() + i, chunk_epsilon.begin() + input_buffer - i);                
                std::copy(time.begin(), time.begin() + i, chunk_time.begin() + input_buffer - i);

                fitOnly_impl(chunk_data, chunk_epsilon, chunk_time);

                chunk_data.erase(chunk_data.begin(), chunk_data.begin() + input_buffer - i);
                chunk_epsilon.erase(chunk_epsilon.begin(), chunk_epsilon.begin() + input_buffer - i);
                chunk_time.erase(chunk_time.begin(), chunk_time.begin() + input_buffer - i);

                std::vector<int> chunk_label(i, 0);
                std::vector<FloatType> chunk_score(i, 0.0f);
                predictOnly_impl(chunk_label, chunk_score, chunk_data, chunk_epsilon, chunk_time);
                std::move(chunk_label.begin(), chunk_label.begin() + i, label.begin());
                std::move(chunk_score.begin(), chunk_score.begin() + i, score.begin());
            }
            
            while ((i + input_buffer) <= data.size()) {                  
                std::vector<Vector<FloatType>> chunk_data(input_buffer);
                std::vector<FloatType> chunk_time(input_buffer);
                std::vector<FloatType> chunk_epsilon(input_buffer);
                std::copy(data.begin() + i, data.begin() + i + input_buffer, chunk_data.begin());
                std::copy(epsilon.begin() + i, epsilon.begin() + i + input_buffer, chunk_epsilon.begin());                
                std::copy(time.begin() + i, time.begin() + i + input_buffer, chunk_time.begin());
                std::vector<int> chunk_label(input_buffer, 0);
                std::vector<FloatType> chunk_score(input_buffer, 0.0f);
                fitPredict_impl(chunk_label, chunk_score, chunk_data, chunk_epsilon, chunk_time);
                std::move(chunk_label.begin(), chunk_label.end(), label.begin() + i);
                std::move(chunk_score.begin(), chunk_score.end(), score.begin() + i);                
                i += input_buffer;
            }

            if (i < data.size()) {
                int j = data.size() - i;
                std::vector<Vector<FloatType>> chunk_data(j);
                std::vector<FloatType> chunk_time(j);
                std::vector<FloatType> chunk_epsilon(j);
                std::copy(data.begin() + i, data.end(), chunk_data.begin());
                std::copy(epsilon.begin() + i, epsilon.end(), chunk_epsilon.begin());                
                std::copy(time.begin() + i, time.end(), chunk_time.begin());
                std::vector<int> chunk_label(j, 0);
                std::vector<FloatType> chunk_score(j, 0.0f);
                predictOnly_impl(chunk_label, chunk_score, chunk_data, chunk_epsilon, chunk_time);                
                buffer.add(chunk_data, chunk_epsilon, chunk_time);
                std::move(chunk_label.begin(), chunk_label.end(), label.begin() + i);
                std::move(chunk_score.begin(), chunk_score.end(), score.begin() + i);
            }
        } else { fitPredict_impl(label, score, data, epsilon, time); }
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
        // int getIndex() {return it->getIndex()};
        Vector<FloatType> getData() { return it->getData(); }
        int getColor() { return it->color; }
        FloatType getObservations(FloatType now) {
            return it->getObservations() * std::pow(fading, now - it->time_touched);
        }
        FloatType getAvObservations(FloatType now) {
            return (1-fading) * it->getObservations() * std::pow(fading, now - it->time_touched) / it->age;
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

#include "SDOstreamclust_fitpred.h"
#include "SDOstreamclust_observer.h"
#include "SDOstreamclust_buffer.h"

#endif  // SDOSTREAMCLUST_H

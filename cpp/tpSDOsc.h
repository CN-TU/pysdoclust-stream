#ifndef TPSDOSC_H
#define TPSDOSC_H

#include <algorithm>
#include <boost/container/set.hpp>
#include <cmath>
#include <functional>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <queue>
#include <complex>

#include "Vector.h"
#include "MTree.h"

template<typename FloatType=double>
class tpSDOsc {
  private:
    typedef std::pair<Vector<FloatType>, FloatType> Point; // data, epsilon
  public:
    // typedef std::function<FloatType(const Vector<FloatType>&, const Vector<FloatType>&)> DistanceFunction;
    // Define a new DistanceFunction type with epsilon handling
    typedef std::function<FloatType(const Point&, const Point&)> DistanceFunction;
  private:
    const std::complex<FloatType> imag_unit{0.0, 1.0};
  
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
    // number of nearest observers to consider
    std::size_t neighbor_cnt;
    // number of nearest observer relative to active_observers
    std::size_t freq_bins;    
    // number of bin
    FloatType max_freq;
    // frequency

    FloatType k_tanh;
    // tanh(( k_tanh * (outlier_threshold-1)) = 0.5 where outlier_threshold is a factor of h_bar(Observer)
    bool outlier_handling;
    // flag for outlier handling
    bool rel_outlier_score;
    // relative or absolute distance for outlier score
    FloatType perturb;

    bool random_sampling;
    
    // std::vector<FloatType> obs_scaler;
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

    DistanceFunction distance_function;
    std::mt19937 rng;

    std::size_t chi_min;
    FloatType chi_prop;
    FloatType zeta;
    FloatType h; // global h (mean of all active h)
    std::size_t e; // unused by now

    int last_color;

    typedef MTree< Point, int, FloatType, MTreeDescendantCounter<Point,int> > Tree;
    typedef typename Tree::iterator TreeIterator;
    typedef std::vector<std::pair<TreeIterator, FloatType>> TreeNeighbors;

    // Observer Structures
    struct Observer;
    struct ObserverCompare;
    ObserverCompare observer_compare;   

    typedef boost::container::multiset<Observer,ObserverCompare> MapType;
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

    Tree tree;
    Tree treeA; 
    
    //print
    void printClusters(); 
    void printDistanceMatrix(); 
    void printObservers(FloatType now);

     // util
    bool hasEdge(FloatType distance, const MapIterator& it);
    FloatType calcBatchAge(const std::vector<FloatType>& time_data, FloatType score = 1);
//     void setObsScaler();
    void initNowVector(FloatType now, std::vector<std::complex<FloatType>>& now_vector, FloatType score); 
    void initNowVector(FloatType now, std::vector<std::complex<FloatType>>& now_vector);
    FloatType getActiveObservationsThreshold(std::size_t active_threshold, FloatType now);
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
            const std::vector<FloatType>& time_data,
            const std::unordered_set<int>& sampled,
            int first_index);
    void fit_point(
            std::unordered_map<int, std::pair<std::vector<std::complex<FloatType>>, FloatType>>& temporary_scores,
            const Point& point,
            FloatType now,           
            std::size_t current_observer_cnt,
            std::size_t current_neighbor_cnt,
            int observer_index);
    void fit_point(
            std::unordered_map<int, std::pair<std::vector<std::complex<FloatType>>, FloatType>>& temporary_scores,
            const Point& point,
            FloatType now,           
            std::size_t current_observer_cnt,
            std::size_t current_neighbor_cnt);
    void update_model(
            const std::unordered_map<int, std::pair<std::vector<std::complex<FloatType>>, FloatType>>& temporary_scores);

    // predict
    void predict_impl(
            std::vector<int>& label,
            std::vector<FloatType>& score,
            const std::vector<Vector<FloatType>>& data,
            const std::vector<FloatType>& epsilon,
            const std::unordered_set<int>& sampled,
            int first_index);
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
            int observer_index); 
    void predict_point(
            int& label,
            FloatType& score,
            const Point& point,
            std::size_t current_neighbor_cnt);

    // sample
    void sample(
            std::unordered_set<int>& sampled,
            const std::vector<Vector<FloatType>>& data,
            const std::vector<FloatType>& epsilon,
            const std::vector<FloatType>& time_data,
            int first_index);
    bool sample_point( 
            std::unordered_set<int>& sampled,
            FloatType now,
            std::size_t batch_size,
            FloatType batch_time,
            int current_index);
    void sample_point(
            std::unordered_set<int>& sampled,
            const Point& point,
            FloatType now,
            FloatType observations_sum,
            std::size_t current_observer_cnt,
            std::size_t current_neighbor_cnt,
            int current_index);
    void replaceObservers(
            Point data,
            std::priority_queue<MapIterator,std::vector<MapIterator>,IteratorAvCompare>& worst_observers,
            FloatType now,
            std::size_t current_observer_cnt,
            std::size_t current_neigbor_cnt,
            int current_index);

    // graph
    void update(
            const std::vector<FloatType>& time_data,
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
            const std::vector<FloatType>& time_data); 

    void fitOnly_impl(
            const std::vector<Vector<FloatType>>& data,
            const std::vector<FloatType>& epsilon,
            const std::vector<FloatType>& time_data); 

public:
    tpSDOsc(
        std::size_t observer_cnt, 
        FloatType T, 
        FloatType idle_observers, 
        std::size_t neighbor_cnt,
        std::size_t chi_min,
        FloatType chi_prop,
        FloatType zeta,
        std::size_t e,
        std::size_t freq_bins, 
        FloatType max_freq,
        FloatType outlier_threshold,
        bool outlier_handling = false,
        bool rel_outlier_score = true,
        FloatType perturb = 0,
        bool random_sampling = true,
        tpSDOsc<FloatType>::DistanceFunction distance_function = Vector<FloatType>::euclideanE, 
        int seed = 0
    ) : observer_cnt(observer_cnt), 
        active_observers(1-idle_observers), 
        sampling_prefactor(observer_cnt / T),
        fading(std::exp(-1/T)),
        neighbor_cnt(neighbor_cnt),
        freq_bins(freq_bins),
        max_freq(max_freq),
        k_tanh( (!outlier_handling) ? 0 : atanh(0.5f) / (outlier_threshold-1) ),
        outlier_handling(outlier_handling),
        rel_outlier_score(rel_outlier_score),
        perturb(perturb),
        random_sampling(random_sampling),        
        // obs_scaler(observer_cnt+1),
        binomial(observer_cnt,neighbor_cnt),
        last_index(0),
        last_added_index(0),
        last_time(0),
        last_added_time(0),
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
    {
        // Print out the input parameters
        // std::cout << "Input parameters:\n"
        //           << "observer_cnt: " << observer_cnt << "\n"
        //           << "T: " << T << "\n"
        //           << "idle_observers: " << idle_observers << "\n"
        //           << "neighbor_cnt: " << neighbor_cnt << "\n"
        //           << "chi_min: " << chi_min << "\n"
        //           << "chi_prop: " << chi_prop << "\n"
        //           << "zeta: " << zeta << "\n"
        //           << "e: " << e << "\n"
        //           << "freq_bins: " << freq_bins << "\n"
        //           << "max_freq: " << max_freq << "\n"
        //           << "outlier_threshold: " << outlier_threshold << "\n"
        //           << "outlier_handling: " << outlier_handling << "\n"
        //           << "perturb: " << perturb << "\n"
        //           << "random_sampling: " << random_sampling << "\n"
        //           << "seed: " << seed << std::endl;
    }

    void fit(
            const std::vector<Vector<FloatType>>& data, 
            const std::vector<FloatType>& time_data) {        
        std::vector<FloatType> epsilon(data.size(), 0.0);
        if (perturb > 0) {
            std::uniform_real_distribution<FloatType> distribution(-perturb, perturb);
            std::generate(epsilon.begin(), epsilon.end(), [&distribution, this] () {
                return distribution(rng);
            });
        }
        fitOnly_impl(data, epsilon, time_data);
    }

    void fitPredict(
            std::vector<int>& label, 
            std::vector<FloatType>& score,
            const std::vector<Vector<FloatType>>& data, 
            const std::vector<FloatType>& time_data) {
        std::vector<FloatType> epsilon(data.size(), 0.0);
        if (perturb > 0) {
            std::uniform_real_distribution<FloatType> distribution(-perturb, perturb);
            std::generate(epsilon.begin(), epsilon.end(), [&distribution, this] () {
                return distribution(rng);
            });
        }
        fitPredict_impl(label, score, data, epsilon, time_data);
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

#include "tpSDOsc_fitpred.h"

#endif  // TPSDOSC_H
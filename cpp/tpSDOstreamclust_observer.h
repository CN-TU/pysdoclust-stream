#ifndef TPSDOSTREAMCLUST_OBSERVER_H
#define TPSDOSTREAMCLUST_OBSERVER_H

#include<limits>



template<typename FloatType>
struct tpSDOstreamclust<FloatType>::Observer {
    Point data;
    std::vector<std::complex<FloatType>> observations;
    FloatType time_touched;       
	FloatType age;    
    int index;

    bool active;
    TreeIterator treeIt;
    TreeIterator treeItA;

    int color;
    std::unordered_map<int, FloatType> color_observations; // color, score
    std::unordered_map<int, FloatType> color_distribution; // color, score normalized

    FloatType h;
    std::vector<std::pair<TreeIterator,FloatType>> nearestNeighbors;

    // Constructor for Observer
    Observer(
        Point data,
        std::vector<std::complex<FloatType>> observations,
        FloatType time_touched,
        int index,
        Tree* tree,
        Tree* treeA // should contain index and data soon
    ) : data(data),
        observations(observations),
        time_touched(time_touched),
        age(0),
        index(index),
        active(false),
        treeIt(tree->end()),
        treeItA(treeA->end()),
        color(0),
        color_observations(),
        color_distribution(),
        h(0),
        nearestNeighbors() {
            treeIt = tree->insert(tree->end(), std::make_pair(data, index)); 
        }

    FloatType getProjObservations(
            const std::vector<std::complex<FloatType>>& now_vector, 
            FloatType fading_factor) const {
        FloatType proj_observations(0);
        int freq_ind = 0;
        for (const auto& now : now_vector) {
            proj_observations += real(observations[freq_ind] * conj(now)) * fading_factor;
            freq_ind++;
        }
        return proj_observations;
    }

    int getIndex() const { return index; }
    Vector<FloatType> getData() const { return data.first; } // without Epsilon
    FloatType getObservations() const { return real(observations[0]); }
    FloatType getH() const { return h; }

    void updateAge(FloatType age_factor, FloatType score) {
        age *= age_factor;
        age += score;
    }

    void updateObservations(
            FloatType fading_factor,
            const std::vector<std::complex<FloatType>>& score_vector) {
        int freq_ind = 0;
        for (const auto& score : score_vector) {
            observations[freq_ind] *= fading_factor;
            observations[freq_ind] += score;
            freq_ind++;
        }
    }

    bool activate(Tree* treeA) {
        if (!active) {
            treeItA = treeA->insert(treeA->end(), std::make_pair(data, index));  
            active = true;
            return true;
        }
        return false;
    }
    bool deactivate(Tree* treeA) {
        if (active) {
            treeA->erase(treeItA);
            treeItA = treeA->end();
            active = false;
            return true;
        }
        return false;
    }

    void setH(Tree* treeA, std::size_t chi) {
        nearestNeighbors = treeA->knnSearch(data, chi+1, true, 0, std::numeric_limits<FloatType>::infinity(), false, true); // one more cause one point is Observer
        h = nearestNeighbors[chi].second;
    }
    // n>=chi is necessary
    void setH(Tree* treeA, std::size_t chi, std::size_t n) {
        nearestNeighbors = treeA->knnSearch(data, n+1, true, 0, std::numeric_limits<FloatType>::infinity(), false, true); // one more cause one point is Observer
        h = nearestNeighbors[chi].second;
    }

    void reset(
        Point _data,
        std::vector<std::complex<FloatType>> _observations,
        FloatType _time_touched,
        int _index,
        Tree* tree,
        Tree* treeA
    ) {
        data = _data;
        observations = _observations;
        time_touched = _time_touched;
        age = 0;
        index = _index;        
        color_observations.clear();
        color_distribution.clear();
        color = 0;
        h = 0;
        nearestNeighbors.clear();
        // TreeNodeUpdater updater(_data, _index);
        // tree->modify(treeIt, updater);
        tree->erase(treeIt);
        treeIt = tree->insert(tree->end(), std::make_pair(_data, _index));    
        if (active) treeA->erase(treeItA);
        treeItA = treeA->end();
        active = false;
    }

    // graph
    void updateColorDistribution(); 
    void updateColorObservations(
            int colorObs,
            FloatType age_factor,
            FloatType score);
    // print
    // void printColorObservations(FloatType now, FloatType fading_cluster) const;
    // void printData() const;
    // void printColorDistribution() const;
};

template<typename FloatType>
struct tpSDOstreamclust<FloatType>::ObserverCompare{
    FloatType fading;
    ObserverCompare(FloatType fading) : fading(fading) {}
    bool operator()(const Observer& a, const Observer& b) const {
        FloatType common_touched = std::max(a.time_touched, b.time_touched);        
        FloatType observations_a = a.getObservations()
            * std::pow(fading, common_touched - a.time_touched);        
        FloatType observations_b = b.getObservations()
            * std::pow(fading, common_touched - b.time_touched);        
        // tie breaker for reproducibility
        if (observations_a == observations_b)
            return a.index > b.index;
        return observations_a > observations_b;
    }
};

template<typename FloatType>
struct tpSDOstreamclust<FloatType>::IteratorAvCompare{
    FloatType fading;
    IteratorAvCompare(FloatType fading) : fading(fading) {}
    bool operator()(const MapIterator& it_a, const MapIterator& it_b) {
        const Observer& a = *it_a;
        const Observer& b = *it_b;
        FloatType common_touched = std::max(a.time_touched, b.time_touched);        
        FloatType observations_a = a.getObservations() * std::pow(fading, common_touched - a.time_touched);  
        FloatType observations_b = b.getObservations() * std::pow(fading, common_touched - b.time_touched);
        return observations_a * b.age > observations_b * a.age;
    }
};

#include "tpSDOstreamclust_graph.h"

#endif  // TPSDOSTREAMCLUST_OBSERVER_H
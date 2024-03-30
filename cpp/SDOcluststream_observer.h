#ifndef SDOCLUSTSTREAM_OBSERVER_H
#define SDOCLUSTSTREAM_OBSERVER_H

template<typename FloatType, typename ObservationType>
template<typename T>
struct SDOcluststream<FloatType, ObservationType>::Observer {
    Vector<FloatType> data;
    T observations;
    FloatType time_touched;        
    FloatType time_added;
    int index;
    TreeIterator treeIt;
    bool active;
    TreeIterator treeItA;

    FloatType time_cluster_touched;
    int color;
    std::unordered_map<int, FloatType> color_observations; // color, score
    std::unordered_map<int, FloatType> color_distribution; // color, score normalized

    FloatType h;
    std::vector<std::pair<TreeIterator,FloatType>> nearestNeighbors;

    // Constructor for Observer
    Observer(
        Vector<FloatType> data,
        T observations,
        FloatType time_touched,
        FloatType time_added,
        int index,
        Tree* tree,
        Tree* treeA // should contain index and data soon
    ) : data(data),
        observations(observations),
        time_touched(time_touched),
        time_added(time_added),
        index(index),
        treeIt(tree->end()),
        active(false),
        treeItA(treeA->end()),
        time_cluster_touched(time_touched),
        color(0),
        color_observations(),
        color_distribution(),
        h(0),
        nearestNeighbors() {
            treeIt = tree->insert(tree->end(), std::make_pair(data, index)); 
        }
    
    int getIndex() const {
        return index;
    }

    Vector<FloatType> getData() {
        return data;
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

    void setH(Tree* treeA, int chi) {
        nearestNeighbors = treeA->knnSearch(data, chi+1, true, 0, std::numeric_limits<FloatType>::infinity(), false, true); // one more cause one point is Observer
        h = nearestNeighbors[chi].second;
    }

    // n>=chi is necessary
    void setH(Tree* treeA, int chi, int n) {
        nearestNeighbors = treeA->knnSearch(data, n+1, true, 0, std::numeric_limits<FloatType>::infinity(), false, true); // one more cause one point is Observer
        h = nearestNeighbors[chi].second;
    }

    void reset(
        Vector<FloatType> _data,
        T _observations,
        FloatType _time_touched,
        FloatType _time_added,
        int _index,
        Tree* tree,
        Tree* treeA
    ) {
        data = _data;
        observations = _observations;
        time_touched = _time_touched;
        time_added = _time_added;
        time_cluster_touched = _time_touched;
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
        active = false;
        treeItA = treeA->end();
    }

    void updateColorDistribution(); // graph
    void updateColorObservations(
            int colorObs, 
            FloatType now, 
            FloatType fading_cluster); // graph

    void printColorObservations(FloatType now, FloatType fading_cluster) const;
    void printData() const;
    void printColorDistribution() const;
};

template<typename FloatType, typename ObservationType>
struct SDOcluststream<FloatType,ObservationType>::ObserverCompare{
    FloatType fading;

    // ObserverCompare() : fading(1.0) {}
    ObserverCompare(FloatType fading) : fading(fading) {}

    template<typename T=ObservationType>
    bool operator()(const Observer<T>& a, const Observer<T>& b) const {
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
};

template<typename FloatType, typename ObservationType>
struct SDOcluststream<FloatType,ObservationType>::ObserverAvCompare{
    FloatType fading;
    ObserverAvCompare(FloatType fading) : fading(fading) {}

    template<typename T=ObservationType>
    bool operator()(FloatType now, const Observer<T>& a, const Observer<T>& b) {
        FloatType common_touched = std::max(a.time_touched, b.time_touched);
        
        FloatType observations_a = a.observations * std::pow(fading, common_touched - a.time_touched);
        FloatType age_a = 1-std::pow(fading, now-a.time_added);
        
        FloatType observations_b = b.observations * std::pow(fading, common_touched - b.time_touched);
        FloatType age_b = 1-std::pow(fading, now-b.time_added);
        
        // do not necessarily need a tie breaker here
        return observations_a * age_b > observations_b * age_a;
    }
};

template<typename FloatType, typename ObservationType>
struct SDOcluststream<FloatType,ObservationType>::IteratorAvCompare{
    FloatType fading;
    FloatType now;
    IteratorAvCompare(FloatType fading, FloatType now) : fading(fading), now(now) {}
    template<typename T=ObservationType>
    bool operator()(const MapIterator& it_a, const MapIterator& it_b) {
        const Observer<T>& a = *it_a;
        const Observer<T>& b = *it_b;
        FloatType common_touched = std::max(a.time_touched, b.time_touched);
        
        FloatType observations_a = a.observations * std::pow(fading, common_touched - a.time_touched);
        FloatType age_a = 1-std::pow(fading, now-a.time_added);
        
        FloatType observations_b = b.observations * std::pow(fading, common_touched - b.time_touched);
        FloatType age_b = 1-std::pow(fading, now-b.time_added);
        
        // do not necessarily need a tie breaker here
        return observations_a * age_b > observations_b * age_a;
    }
};

#endif  // SDOCLUSTSTREAM_OBSERVER_H
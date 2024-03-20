#ifndef SDOCLUSTSTREAM_OBSERVER_H
#define SDOCLUSTSTREAM_OBSERVER_H

template<typename FloatType>
struct SDOcluststream<FloatType>::Observer {
    Vector<FloatType> data;
    FloatType observations;
    FloatType time_touched;        
    FloatType time_added;
    int index;
    bool active;

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
    
    int getIndex() const {
        return index;
    }

    Vector<FloatType> getData() {
        return data;
    }

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

    void updateColorDistribution(); // graph
    void updateColorObservations(
            int colorObs, 
            FloatType now, 
            FloatType fading_cluster); // graph

    void printColorObservations(FloatType now, FloatType fading_cluster) const;
    void printData() const;
    void printColorDistribution() const;
};

template<typename FloatType>
struct SDOcluststream<FloatType>::ObserverCompare{
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
};

template<typename FloatType>
struct SDOcluststream<FloatType>::ObserverAvCompare{
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
};

template<typename FloatType>
struct SDOcluststream<FloatType>::IteratorAvCompare{
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

#endif  // SDOCLUSTSTREAM_OBSERVER_H
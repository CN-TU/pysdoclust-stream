#ifndef TPSDOSC_OBSERVER_H
#define TPSDOSC_OBSERVER_H

template<typename FloatType>
struct tpSDOsc<FloatType>::Observer {
    Vector<FloatType> data;
    std::vector<std::complex<FloatType>> observations;
    FloatType time_touched;       
	FloatType age;    
    // FloatType time_added;
    int index;
    TreeIterator treeIt;
    bool active;
    TreeIterator treeItA;

    int color;
    std::unordered_map<int, FloatType> color_observations; // color, score
    std::unordered_map<int, FloatType> color_distribution; // color, score normalized

    FloatType h;
    std::vector<std::pair<TreeIterator,FloatType>> nearestNeighbors;

    // Constructor for Observer
    Observer(
        Vector<FloatType> data,
        std::vector<std::complex<FloatType>> observations,
        FloatType time_touched,
        FloatType age,
        int index,
        Tree* tree,
        Tree* treeA // should contain index and data soon
    ) : data(data),
        observations(observations),
        time_touched(time_touched),
        age(age),
        index(index),
        treeIt(tree->end()),
        active(false),
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
            std::size_t freq_bins, 
            FloatType fading_factor) const {
        FloatType proj_observations(0);
        for (std::size_t freq_ind = 0; freq_ind < freq_bins; freq_ind++) {
            proj_observations += real(observations[freq_ind] * conj(now_vector[freq_ind])) * fading_factor;
        }
        return proj_observations;
    }

    FloatType getObservations() {
        return real(observations[0]);
    }

    void updateAge(FloatType age_factor, FloatType score = 1) {
        age *= age_factor;
        age += score;
    }

    void updateObservations(
            std::size_t freq_bins, 
            FloatType fading_factor,
            const std::vector<std::complex<FloatType>>& score_vector) {
        for (std::size_t freq_ind = 0; freq_ind < freq_bins; freq_ind++) {
            observations[freq_ind] *= fading_factor;
            observations[freq_ind] += score_vector[freq_ind];
        }
        updateAge(fading_factor, real(score_vector[0]));
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
        std::vector<std::complex<FloatType>> _observations,
        FloatType _time_touched,
        FloatType _age, 
        int _index,
        Tree* tree,
        Tree* treeA
    ) {
        data = _data;
        observations = _observations;
        time_touched = _time_touched;
        age = _age;
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

template<typename FloatType>
struct tpSDOsc<FloatType>::ObserverCompare{
    FloatType fading;

    // ObserverCompare() : fading(1.0) {}
    ObserverCompare(FloatType fading) : fading(fading) {}

    bool operator()(const Observer& a, const Observer& b) const {
        FloatType common_touched = std::max(a.time_touched, b.time_touched);        
        FloatType observations_a = real(a.observations[0])
            * std::pow(fading, common_touched - a.time_touched);        
        FloatType observations_b = real(b.observations[0])
            * std::pow(fading, common_touched - b.time_touched);        
        // tie breaker for reproducibility
        if (observations_a == observations_b)
            return a.index > b.index;
        return observations_a > observations_b;
    }
};

template<typename FloatType>
struct tpSDOsc<FloatType>::ObserverAvCompare{
    FloatType fading;
    ObserverAvCompare(FloatType fading) : fading(fading) {}
    bool operator()(FloatType now, const Observer& a, const Observer& b) {
        FloatType common_touched = std::max(a.time_touched, b.time_touched);
        
        FloatType observations_a = real(a.observations[0]) * std::pow(fading, common_touched - a.time_touched);
        // FloatType age_a = 1-std::pow(fading, now-a.time_added);
        
        FloatType observations_b = real(b.observations[0]) * std::pow(fading, common_touched - b.time_touched);
        // FloatType age_b = 1-std::pow(fading, now-b.time_added);
        
        // do not necessarily need a tie breaker here
        return observations_a * b.age > observations_b * a.age;
        // return observations_a * age_b > observations_b * age_a;
    }
};

template<typename FloatType>
struct tpSDOsc<FloatType>::IteratorAvCompare{
    FloatType fading;
    FloatType now;
    IteratorAvCompare(FloatType fading, FloatType now) : fading(fading), now(now) {}
    bool operator()(const MapIterator& it_a, const MapIterator& it_b) {
        const Observer& a = *it_a;
        const Observer& b = *it_b;
        FloatType common_touched = std::max(a.time_touched, b.time_touched);
        
        FloatType observations_a = real(a.observations[0]) * std::pow(fading, common_touched - a.time_touched);
        // FloatType age_a = 1-std::pow(fading, now-a.time_added);
        
        FloatType observations_b = real(b.observations[0]) * std::pow(fading, common_touched - b.time_touched);
        // FloatType age_b = 1-std::pow(fading, now-b.time_added);
        
        // do not necessarily need a tie breaker here
        return observations_a * b.age > observations_b * a.age;
        // return observations_a * age_b > observations_b * age_a;
    }
};

#endif  // TPSDOSC_OBSERVER_H
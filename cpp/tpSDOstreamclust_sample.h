#ifndef TPSDOSTREAMCLUST_SAMPLE_H
#define TPSDOSTREAMCLUST_SAMPLE_H

template<typename FloatType>
void tpSDOstreamclust<FloatType>::sample(
        std::unordered_set<int>& sampled,
        const std::vector<Vector<FloatType>>& data,
        const std::vector<FloatType>& epsilon,
        const std::vector<FloatType>& time,
        int first_index) {  
    std::size_t active_threshold(0), active_threshold2(0);
    std::size_t current_neighbor_cnt(0), current_neighbor_cnt2(0);
    std::size_t current_observer_cnt(0), current_observer_cnt2(0);
    std::size_t current_e(0);
    std::size_t chi(0); 
    if (!observers.empty()) {
        setModelParameters(
            current_observer_cnt, current_observer_cnt2,
            active_threshold, active_threshold2,
            current_neighbor_cnt, current_neighbor_cnt2,
            current_e,
            chi,
            false); // true for print 
    }    
    FloatType observations_sum(0);   
    if (!random_sampling) { // only in use if           
        for (auto it = observers.begin(); it != observers.end(); ++it) {
            observations_sum += it->getObservations() * std::pow<FloatType>(fading, last_time- it->time_touched);
        }
    }
    if (observers.empty()) {
        bool firstPointSampled(false);
        for (std::size_t i = 0; i < data.size(); ++i) { 
            if (sample_point(
                    sampled,
                    time[i],
                    data.size(),
                    time.back() - time.front(),
                    last_index++)
                ) { firstPointSampled = true; }
        }
        if (!firstPointSampled) {
            std::uniform_int_distribution<int> dist(0, data.size() - 1);
            int indexToSample = dist(rng);
            sampled.insert(indexToSample);
            last_added_index = indexToSample;            
        }
    } else {
        for (std::size_t i = 0; i < data.size(); ++i) {     
            if (!random_sampling) {
                sample_point(
                    sampled,
                    std::make_pair(data[i], epsilon[i]),                    
                    time[i],
                    observations_sum * std::pow<FloatType>(fading, time[i] - last_time),
                    current_observer_cnt,
                    current_neighbor_cnt,
                    last_index++);      
            } else {
                sample_point(
                    sampled,
                    time[i],
                    data.size(),
                    time.back() - last_time,
                    last_index++);
            }       
        }
    }
    // Can not replace more observers than max size of model
    if (sampled.size()>observer_cnt) {
        std::vector<typename std::unordered_set<int>::value_type> shuffled_elements(sampled.begin(), sampled.end());
        std::shuffle(shuffled_elements.begin(), shuffled_elements.end(), rng);
        sampled.clear();
        sampled.insert(shuffled_elements.begin(), shuffled_elements.begin() + observer_cnt);
    }

    // Queue worst observers, not perfectly efficient as all observers are queued, only n sampled d be necessary
    IteratorAvCompare iterator_av_compare(fading);
    std::priority_queue<MapIterator,std::vector<MapIterator>,IteratorAvCompare> worst_observers(iterator_av_compare);
    for (auto it = observers.begin(); it != observers.end(); ++it) {
        worst_observers.push(it);            
    }
    for (std::size_t i = 0; i < data.size(); ++i) {
        int current_index = first_index + i;
        bool is_observer = (sampled.count(current_index) > 0);     
        if (is_observer) {            
            replaceObservers(
                std::make_pair(data[i], epsilon[i]),
                worst_observers,
                time[i],
                current_observer_cnt,
                current_neighbor_cnt,
                current_index
            );
            last_added_index = current_index;
            last_added_time = time[i];
        }
    }   
}

template<typename FloatType>
bool tpSDOstreamclust<FloatType>::sample_point( 
    std::unordered_set<int>& sampled,
    FloatType now,
    std::size_t batch_size,
    FloatType batch_time,
    int current_index) {
    bool add_as_observer = 
        (rng() - rng.min()) * batch_size < sampling_prefactor * (rng.max() - rng.min()) * batch_time; // observer_cnt / T
    if (add_as_observer) {            
        sampled.insert(current_index);   
        last_added_index = current_index;
        last_added_time = now;
        return true;
    }
    return false;
};

template<typename FloatType>
void tpSDOstreamclust<FloatType>::sample_point(
        std::unordered_set<int>& sampled,
        const Point& point,
        FloatType now,
        FloatType observations_sum,
        std::size_t current_observer_cnt,
        std::size_t current_neighbor_cnt,
        int current_index) {        
    bool add_as_observer;
    if (!observers.empty()) {            
        auto nearestNeighbors = tree.knnSearch(point, current_neighbor_cnt);
        FloatType observations_nearest_sum(0);
        for (const auto& neighbor : nearestNeighbors) {
            int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered               
            const MapIterator& it = indexToIterator[idx];
            observations_nearest_sum += it->getObservations() * std::pow<FloatType>(fading, now-it->time_touched);
        }   
        add_as_observer = 
            (rng() - rng.min()) * current_neighbor_cnt * observations_sum * (current_index - last_added_index) < 
                sampling_prefactor * (rng.max() - rng.min()) * current_observer_cnt * observations_nearest_sum * (now - last_added_time);
    }   
    if (add_as_observer) {            
        sampled.insert(current_index);   
        last_added_index = current_index;
        last_added_time = now;
    }
};

template<typename FloatType>
void tpSDOstreamclust<FloatType>::replaceObservers(
        Point data,
        std::priority_queue<MapIterator,std::vector<MapIterator>,IteratorAvCompare>& worst_observers,
        FloatType now,
        std::size_t current_observer_cnt,
        std::size_t current_neighbor_cnt,
        int current_index) {        
    MapIterator obsIt = observers.end();
    std::vector<std::complex<FloatType>> init_score_vector;
    FloatType score = (observer_cnt==current_observer_cnt) ? FloatType(1) : binomial.calc(current_observer_cnt, current_neighbor_cnt)/binomial.calc(observer_cnt, neighbor_cnt);
    initNowVector(now, init_score_vector, score);
    if (observers.size() < observer_cnt) {
        obsIt = observers.insert(Observer(data, init_score_vector, now, current_index, &tree, &treeA)); // maybe init_score instead of 1
    } else {
        // find worst observer
        obsIt = worst_observers.top();  // Get iterator to the "worst" element         
        worst_observers.pop(); 
        int indexToRemove = obsIt->index;
        // do index handling          
        indexToIterator.erase(indexToRemove);
        // update Observer(s)
        auto node = observers.extract(obsIt);
        Observer& observer = node.value();
        observer.reset(data, init_score_vector, now, current_index, &tree, &treeA); // maybe init_score instead of 1
        observers.insert(std::move(node));    
    }
    indexToIterator[current_index] = obsIt;
};

#endif  // TPSDOSTREAMCLUST_SAMPLE_H
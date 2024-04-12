#ifndef SDOCLUSTSTREAM_FIT_H
#define SDOCLUSTSTREAM_FIT_H

template<typename FloatType> 
void SDOcluststream<FloatType>::fit_impl(
        const std::vector<Vector<FloatType>>& data,
        const std::vector<FloatType>& epsilon,
        const std::vector<FloatType>& time_data,
        const std::unordered_set<int>& sampled,
        int first_index) {
    int active_threshold(0), active_threshold2(0);
    int current_neighbor_cnt(0), current_neighbor_cnt2(0);
    int current_observer_cnt(0), current_observer_cnt2(0);
    size_t current_e(0);
    size_t chi(0);    
    setModelParameters(
        current_observer_cnt, current_observer_cnt2,
        active_threshold, active_threshold2,
        current_neighbor_cnt, current_neighbor_cnt2,
        current_e,
        chi,
        false); // true for print
    std::unordered_map<int, std::pair<FloatType, FloatType>> temporary_scores;
    for (size_t i = 0; i < data.size(); ++i) {   
        int current_index = first_index + 1;
        bool is_observer = (sampled.count(current_index) > 0);
        if (is_observer) {
            fit_point(
                temporary_scores,
                std::make_pair(data[i], epsilon[i]),
                time_data[i],
                current_observer_cnt2,
                current_neighbor_cnt2,
                current_index); 
        } else {
            fit_point(
                temporary_scores,
                std::make_pair(data[i], epsilon[i]),
                time_data[i],
                current_observer_cnt,
                current_neighbor_cnt); 
        }
    }
    update_model(temporary_scores);
}

template<typename FloatType>
void SDOcluststream<FloatType>::fit_point(
        std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
        const Point& point,
        const FloatType& now,           
        const int& current_observer_cnt,
        const int& current_neighbor_cnt,
        const int& observer_index) {  
    TreeNeighbors nearestNeighbors = tree.knnSearch(point, current_neighbor_cnt + 1); 
    for (const auto& neighbor : nearestNeighbors) {
        int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
        if (idx!=observer_index) {
            if (temporary_scores.count(idx) > 0) {                
                auto& value_pair = temporary_scores[idx];
                value_pair.first *= std::pow<FloatType>(fading, now-value_pair.second);
                value_pair.second = now;
                value_pair.first += obs_scaler[current_observer_cnt];
            } else { temporary_scores[idx] = std::make_pair(obs_scaler[current_observer_cnt], now); }
        }            
    }  
};

template<typename FloatType>
void SDOcluststream<FloatType>::fit_point(
        std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
        const Point& point,
        const FloatType& now,           
        const int& current_observer_cnt,
        const int& current_neighbor_cnt) {   
    TreeNeighbors nearestNeighbors = tree.knnSearch(point, current_neighbor_cnt); // one more cause one point is Observer
    for (const auto& neighbor : nearestNeighbors) {
        int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
        if (temporary_scores.count(idx) > 0) {                
            auto& value_pair = temporary_scores[idx];
            value_pair.first *= std::pow<FloatType>(fading, now-value_pair.second);
            value_pair.second = now;
            value_pair.first += obs_scaler[current_observer_cnt];
        } else { temporary_scores[idx] = std::make_pair(obs_scaler[current_observer_cnt], now); }
    }  
};

template<typename FloatType>
void SDOcluststream<FloatType>::update_model(
        const std::unordered_map<int,std::pair<FloatType, FloatType>>& temporary_scores) {
    for (auto& [key, value_pair] : temporary_scores) {
        const MapIterator& it = indexToIterator[key];
        // Access the value pair:
        FloatType score = value_pair.first;
        FloatType time_touched = value_pair.second;
        auto node = observers.extract(it);    
        Observer& observer = node.value();
        observer.updateObservations(std::pow(fading, time_touched - observer.time_touched), score);
        if (observer.time_touched < time_touched) { observer.time_touched = time_touched; }        
        observers.insert(std::move(node));
    }
};

#endif  // SDOCLUSTSTREAM_FIT_H
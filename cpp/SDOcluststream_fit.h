#ifndef SDOCLUSTSTREAM_FIT_H
#define SDOCLUSTSTREAM_FIT_H

template<typename FloatType>
void SDOcluststream<FloatType>::fit_impl(
        std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
        const Vector<FloatType>& point,
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
void SDOcluststream<FloatType>::fit_impl(
        std::unordered_map<int, std::pair<FloatType, FloatType>>& temporary_scores,
        const Vector<FloatType>& point,
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

#endif  // SDOCLUSTSTREAM_FIT_H
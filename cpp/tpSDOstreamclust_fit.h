#ifndef TPSDOSTREAMCLUST_FIT_H
#define TPSDOSTREAMCLUST_FIT_H

template<typename FloatType> 
void tpSDOstreamclust<FloatType>::fit_impl(
        const std::vector<Vector<FloatType>>& data,
        const std::vector<FloatType>& epsilon,
        const std::vector<FloatType>& time,
        int first_index) {
    std::size_t active_threshold(0), active_threshold2(0);
    std::size_t current_neighbor_cnt(0), current_neighbor_cnt2(0);
    std::size_t current_observer_cnt(0), current_observer_cnt2(0);
    std::size_t current_e(0);
    std::size_t chi(0);    
    setModelParameters(
        current_observer_cnt, current_observer_cnt2,
        active_threshold, active_threshold2,
        current_neighbor_cnt, current_neighbor_cnt2,
        current_e,
        chi,
        false); // true for print
   std::unordered_map<int, std::pair<std::vector<std::complex<FloatType>>, FloatType>> temporary_scores;
    for (std::size_t i = 0; i < data.size(); ++i) {   
        int current_index = first_index + i;
        auto it = indexToIterator.find(current_index);
        bool is_observer = (it != indexToIterator.end()) ? true : false;
        if (is_observer) {
            fit_point(
                temporary_scores,
                std::make_pair(data[i], epsilon[i]),
                time[i],
                current_observer_cnt2,
                current_neighbor_cnt2,
                current_index); 
        } else {
            fit_point(
                temporary_scores,
                std::make_pair(data[i], epsilon[i]),
                time[i],
                current_observer_cnt,
                current_neighbor_cnt); 
        }
    }
    update_model(temporary_scores);
}

template<typename FloatType>
void tpSDOstreamclust<FloatType>::fit_point(
        std::unordered_map<int, std::pair<std::vector<std::complex<FloatType>>, FloatType>>& temporary_scores,
        const Point& point,
        FloatType now,           
        std::size_t current_observer_cnt,
        std::size_t current_neighbor_cnt,
        int observer_index) {  
    TreeNeighbors nearestNeighbors = tree.knnSearch(point, current_neighbor_cnt + 1);     
    for (const auto& neighbor : nearestNeighbors) {
        int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
        if (idx!=observer_index) {
            FloatType score = (observer_cnt==current_observer_cnt) ? FloatType(1) : binomial.calc(current_observer_cnt, current_neighbor_cnt)/binomial.calc(observer_cnt, neighbor_cnt);
            std::vector<std::complex<FloatType>> score_vector;
            initNowVector(now, score_vector, score);
            if (temporary_scores.count(idx) > 0) {                
                auto& value_pair = temporary_scores[idx];
                std::vector<std::complex<FloatType>>& observations = value_pair.first;
                FloatType fading_factor = std::pow<FloatType>(fading, now-value_pair.second);
                for (std::size_t freq_ind = 0; freq_ind < freq_bins; freq_ind++) {
                    observations[freq_ind] *= fading_factor;
                    observations[freq_ind] += score_vector[freq_ind];
                }
                value_pair.second = now;
            } else { 
                std::vector<std::complex<FloatType>> observations(freq_bins);
                for (std::size_t freq_ind = 0; freq_ind < freq_bins; freq_ind++) {
                    observations[freq_ind] = score_vector[freq_ind];
                }
                temporary_scores[idx] = std::make_pair(observations, now);
            }
        }            
    }  
};


template<typename FloatType>
void tpSDOstreamclust<FloatType>::fit_point(
        std::unordered_map<int, std::pair<std::vector<std::complex<FloatType>>, FloatType>>& temporary_scores,
        const Point& point,
        FloatType now,           
        std::size_t current_observer_cnt,
        std::size_t current_neighbor_cnt) {   
    TreeNeighbors nearestNeighbors = tree.knnSearch(point, current_neighbor_cnt); // one more cause one point is Observer
    for (const auto& neighbor : nearestNeighbors) {
        int idx = neighbor.first->second; // second is distance, first->first Vector, Output is not ordered
        FloatType score = (observer_cnt==current_observer_cnt) ? FloatType(1) : binomial.calc(current_observer_cnt, current_neighbor_cnt)/binomial.calc(observer_cnt, neighbor_cnt);
        std::vector<std::complex<FloatType>> score_vector;
        initNowVector(now, score_vector, score);     
        if (temporary_scores.count(idx) > 0) {                
            auto& value_pair = temporary_scores[idx];
            std::vector<std::complex<FloatType>>& observations = value_pair.first;
            FloatType fading_factor = std::pow<FloatType>(fading, now-value_pair.second);
            for (std::size_t freq_ind = 0; freq_ind < freq_bins; freq_ind++) {
                observations[freq_ind] *= fading_factor;
                observations[freq_ind] += score_vector[freq_ind];
            }
            value_pair.second = now;
        } else { 
            std::vector<std::complex<FloatType>> observations(freq_bins);
            for (std::size_t freq_ind = 0; freq_ind < freq_bins; freq_ind++) {
                observations[freq_ind] = score_vector[freq_ind];
            }
            temporary_scores[idx] = std::make_pair(observations, now);
        }
    }  
};


template<typename FloatType>
void tpSDOstreamclust<FloatType>::update_model(
        const std::unordered_map<int, std::pair<std::vector<std::complex<FloatType>>, FloatType>>& temporary_scores) {
    for (auto it0 = temporary_scores.begin(); it0 != temporary_scores.end(); ++it0) {
        int key = it0->first;
        auto value_pair = it0->second;
        const MapIterator& it = indexToIterator[key];
        // Access the value pair:
        const std::vector<std::complex<FloatType>>& score_vector = value_pair.first;
        FloatType time_touched = value_pair.second;
        auto node = observers.extract(it);  
        Observer& observer = node.value();        
        observer.updateObservations(std::pow(fading, time_touched - observer.time_touched), score_vector);
        if (observer.time_touched < time_touched) { observer.time_touched = time_touched; }        
        observers.insert(std::move(node));
    }
};

#endif  // TPSDOSTREAMCLUST_FIT_H
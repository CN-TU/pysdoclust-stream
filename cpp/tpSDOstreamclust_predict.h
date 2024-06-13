#ifndef TPSDOSTREAMCLUST_PREDICT_H
#define TPSDOSTREAMCLUST_PREDICT_H

template<typename FloatType>
void tpSDOstreamclust<FloatType>::predict_impl(
        std::vector<int>& label,
        std::vector<FloatType>& score,
        const std::vector<Vector<FloatType>>& data,
        const std::vector<FloatType>& epsilon,
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
    for (std::size_t i = 0; i < data.size(); ++i) {
        int current_index = first_index + i;
        auto it = indexToIterator.find(current_index);
        bool is_observer = (it != indexToIterator.end()) ? true : false;
        if (is_observer) {
            if (indexToIterator[current_index]->active) {
                predict_point(
                    label[i],
                    score[i],
                    current_neighbor_cnt2,
                    current_index);
            } else {
                predict_point(
                    label[i],
                    score[i],
                    std::make_pair(data[i], epsilon[i]),
                    current_neighbor_cnt2);
            }
        } else {
            predict_point(
                label[i],
                score[i],
                std::make_pair(data[i], epsilon[i]),
                current_neighbor_cnt);
        }
    }
}

template<typename FloatType>
void tpSDOstreamclust<FloatType>::determineLabelVector(
        std::unordered_map<int, FloatType>& label_vector,
        std::vector<FloatType>& score_vector,
        const std::pair<TreeIterator, FloatType>& neighbor) {
    int idx = neighbor.first->second; // second is distance, first->first Vector, Output is ordered
    const MapIterator& it = indexToIterator[idx];
    const auto& color_distribution = it->color_distribution;
    FloatType distance = neighbor.second;
    // outlier score
    if (rel_outlier_score) { score_vector.emplace_back(distance / (zeta * it->h + (1 - zeta) * h)); }
    else { score_vector.emplace_back(distance); }
    FloatType outlier_factor = FloatType(0);
    if (outlier_handling) {
        if (!hasEdge(distance, it)) {   
            FloatType h_bar = (zeta * it->h + (1 - zeta) * h);   
            outlier_factor = tanh( k_tanh * (distance - h_bar) / h_bar );
        }
        label_vector[-1] += outlier_factor;
    }    
    for (const auto& pair : color_distribution) {
        label_vector[pair.first] += (1-outlier_factor) * pair.second;
    }    
}

template<typename FloatType>
void tpSDOstreamclust<FloatType>::setLabel(
        int& label,
        const std::unordered_map<int, FloatType>& label_vector,
        std::size_t current_neighbor_cnt) {
    FloatType maxColorScore(0);
    if (outlier_handling && (label_vector.find(-1) != label_vector.end())) {
        if ( label_vector.at(-1)<(current_neighbor_cnt*0.5) ) {
            for (const auto& pair : label_vector) {            
                if (pair.first<0) { continue; }
                if (pair.second > maxColorScore || (pair.second == maxColorScore && pair.first < label) ) {
                    label = pair.first;
                    maxColorScore = pair.second;
                }
            }
        } else { label = -1; }
    } else {
        for (const auto& pair : label_vector) {   
            if (pair.second > maxColorScore || (pair.second == maxColorScore && pair.first < label) ) {
                label = pair.first;
                maxColorScore = pair.second;
            }
        }
    }
}

template<typename FloatType>
void tpSDOstreamclust<FloatType>::predict_point(
        int& label,
        FloatType& score,
        std::size_t current_neighbor_cnt,
        int observer_index) {    
    if (current_neighbor_cnt>0) {
        std::unordered_map<int, FloatType> label_vector;
        std::vector<FloatType> score_vector;
        score_vector.reserve(current_neighbor_cnt);
        const MapIterator& it0 = indexToIterator[observer_index];
        TreeNeighbors& nearestNeighbors = it0->nearestNeighbors;
        std::size_t i = 1;
        for (const auto& neighbor : nearestNeighbors) {        
            if (observer_index!= neighbor.first->second) {            
                determineLabelVector(label_vector, score_vector, neighbor);          
                ++i;
                if (i > current_neighbor_cnt) { break; }
            }
        }  
        // set score
        std::sort(score_vector.begin(), score_vector.end());
        if (current_neighbor_cnt % 2 == 0) { // If the size is even
            score = (score_vector[current_neighbor_cnt / 2 - 1] + score_vector[current_neighbor_cnt / 2]) / 2.0;
        } else { // If the size is odd
            score = score_vector[current_neighbor_cnt / 2];
        }
        // set label
        label = 0;
        setLabel(label, label_vector, current_neighbor_cnt);
    } else {
        score = 0.0f;
        label = 0;
    }
}

template<typename FloatType>
void tpSDOstreamclust<FloatType>::predict_point(
        int& label,
        FloatType& score,
        const Point& point,
        std::size_t current_neighbor_cnt) {
    std::unordered_map<int, FloatType> label_vector;
    std::vector<FloatType> score_vector;
    score_vector.reserve(current_neighbor_cnt);
    TreeNeighbors nearestNeighbors = treeA.knnSearch(point, current_neighbor_cnt, true, 0, std::numeric_limits<FloatType>::infinity(), false, false);
    for (const auto& neighbor : nearestNeighbors) {
        determineLabelVector(label_vector, score_vector, neighbor);  
    }      
    // set score
    std::sort(score_vector.begin(), score_vector.end());
    if (current_neighbor_cnt % 2 == 0) { // If the size is even
        score = (score_vector[current_neighbor_cnt / 2 - 1] + score_vector[current_neighbor_cnt / 2]) / 2.0;
    } else { // If the size is odd
        score = score_vector[current_neighbor_cnt / 2];
    }
    // set label
    label = 0;
    setLabel(label, label_vector, current_neighbor_cnt);
};

#endif  // TPSDOSTREAMCLUST_PREDICT_H
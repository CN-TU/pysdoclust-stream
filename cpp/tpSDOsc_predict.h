#ifndef TPSDOSC_PREDICT_H
#define TPSDOSC_PREDICT_H

template<typename FloatType>
void tpSDOsc<FloatType>::predict_impl(
        std::vector<int>& labels,
        const std::vector<Vector<FloatType>>& data,
        const std::vector<FloatType>& epsilon,
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
    for (size_t i = 0; i < data.size(); ++i) {
        int current_index = first_index + i;
        bool is_observer = sampled.count(current_index) > 0;
        if (is_observer) {
            if (indexToIterator[current_index]->active) {
                predict_point(
                    labels[i],
                    current_neighbor_cnt2,
                    current_index);
            } else {
                predict_point(
                    labels[i],
                    std::make_pair(data[i], epsilon[i]),
                    current_neighbor_cnt2);
            }
        } else {
            predict_point(
                labels[i],
                std::make_pair(data[i], epsilon[i]),
                current_neighbor_cnt);
        }
    }
}

template<typename FloatType>
void tpSDOsc<FloatType>::determineLabelVector(
        std::unordered_map<int, FloatType>& label_vector,
        const std::pair<TreeIterator, FloatType>& neighbor) {
    int idx = neighbor.first->second; // second is distance, first->first Vector, Output is ordered
    const MapIterator& it = indexToIterator[idx];
    const auto& color_distribution = it->color_distribution;
    FloatType distance = neighbor.second;
    FloatType outlier_factor = FloatType(0);
    if (!hasEdge(distance, it)) {   
        FloatType h_bar = (zeta * it->h + (1 - zeta) * h);   
        outlier_factor = tanh( k_tanh * (distance - h_bar) / h_bar );
    }
    for (const auto& pair : color_distribution) {
        label_vector[pair.first] += (1-outlier_factor) * pair.second;
    }
    label_vector[-1] += outlier_factor; // outlier weight    
}

template<typename FloatType>
void tpSDOsc<FloatType>::setLabel(
        int& label,
        const std::unordered_map<int, FloatType>& label_vector,
        int current_neighbor_cnt) {
    FloatType maxColorScore(0);
    if (label_vector.find(-1) != label_vector.end()) {
        if ( label_vector.at(-1)<(current_neighbor_cnt*0.5) ) {
            for (const auto& pair : label_vector) {            
                if (pair.first<0) { continue; }
                if (pair.second > maxColorScore || (pair.second == maxColorScore && pair.first < label) ) {
                    label = pair.first;
                    maxColorScore = pair.second;
                }
            }
        }
    }
}

template<typename FloatType>
void tpSDOsc<FloatType>::predict_point(
        int& label,
        const int& current_neighbor_cnt,
        const int& observer_index) {
    std::unordered_map<int, FloatType> label_vector;
    const MapIterator& it0 = indexToIterator[observer_index];
    TreeNeighbors& nearestNeighbors = it0->nearestNeighbors;
    int i = 0;
    for (const auto& neighbor : nearestNeighbors) {        
        if (observer_index!= neighbor.first->second) {            
            determineLabelVector(label_vector, neighbor);          
            ++i;
            if (i > current_neighbor_cnt) { break; }
        }
    }  
    // set label
    label = -1;
    setLabel(label, label_vector, current_neighbor_cnt);
}

template<typename FloatType>
void tpSDOsc<FloatType>::predict_point(
        int& label,
        const Point& point,
        const int& current_neighbor_cnt) {
    std::unordered_map<int, FloatType> label_vector;
    TreeNeighbors nearestNeighbors = treeA.knnSearch(point, current_neighbor_cnt, true, 0, std::numeric_limits<FloatType>::infinity(), false, false);
    for (const auto& neighbor : nearestNeighbors) {
        determineLabelVector(label_vector, neighbor);  
    }      
    // set label
    label = -1;
    setLabel(label, label_vector, current_neighbor_cnt);
};

#endif  // TPSDOSC_PREDICT_H
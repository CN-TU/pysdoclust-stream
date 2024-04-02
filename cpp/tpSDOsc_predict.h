#ifndef TPSDOSC_PREDICT_H
#define TPSDOSC_PREDICT_H

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
void tpSDOsc<FloatType>::predict_impl(
        int& label,
        const Vector<FloatType>& point, // could be accessed as with observer_index
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
    FloatType maxColorScore(0);
    if ( label_vector[-1]<(current_neighbor_cnt*0.5) ) {
        for (const auto& pair : label_vector) {            
            if (pair.first<0) { continue; }
            if (pair.second > maxColorScore || (pair.second == maxColorScore && pair.first < label) ) {
                label = pair.first;
                maxColorScore = pair.second;
            }
        }
    }
}

template<typename FloatType>
void tpSDOsc<FloatType>::predict_impl(
        int& label,
        const Vector<FloatType>& point,
        const int& current_neighbor_cnt) {
    std::unordered_map<int, FloatType> label_vector;
    TreeNeighbors nearestNeighbors = treeA.knnSearch(point, current_neighbor_cnt, true, 0, std::numeric_limits<FloatType>::infinity(), false, false);
    int i = 0;
    for (const auto& neighbor : nearestNeighbors) {
        determineLabelVector(label_vector, neighbor);  
        ++i;
        if (i > current_neighbor_cnt) { break; }
    }  
    // set label
    label = -1;
    FloatType maxColorScore(0);
    if ( label_vector[-1]<(current_neighbor_cnt*0.5) ) {
        for (const auto& pair : label_vector) {            
            if (pair.first<0) { continue; }
            if (pair.second > maxColorScore || (pair.second == maxColorScore && pair.first < label) ) {
                label = pair.first;
                maxColorScore = pair.second;
            }
        }
    }
};

#endif  // TPSDOSC_PREDICT_H
#ifndef SDOCLUSTSTREAM_PRINT_H
#define SDOCLUSTSTREAM_PRINT_H

#include "SDOcluststream_observer.h"

template<typename FloatType>
void SDOcluststream<FloatType>::printClusters() {
    for (const auto& cluster : clusters) {
        cluster.printColor();
        cluster.printObserverIndices();            
        cluster.printDistribution();
    }
};

template<typename FloatType>
void SDOcluststream<FloatType>::printDistanceMatrix() {
    std::cout << std::endl << "Distance Matrix" << std::endl;
    for (const auto& entry : distance_matrix) {
        std::cout << "[" << entry.first << "]: ";
        const DistanceMapType& distance_map = entry.second;

        for (const auto& item : distance_map.template get<1>()) {
            std::cout << "(" << item.index << ", " << item.distance << ") ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    for (const auto& entry : distance_matrix) {
        std::cout << "[" << entry.first << "]: ";
        const DistanceMapType& distance_map = entry.second;

        for (const auto& item : distance_map.template get<0>()) {
            std::cout << "" << item.index << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
};

template<typename FloatType>
void SDOcluststream<FloatType>::printObservers(
    FloatType now) {

    std::cout << std::endl << "Observers" << std::endl;
    for (const auto& observer : observers) {
        FloatType pow_fading = std::pow(fading, now - observer.time_touched);
        FloatType age = 1-std::pow(fading, now - observer.time_added);
        std::cout << "(" << observer.index 
                << ", " << observer.observations 
                << ", " << observer.observations * pow_fading 
                << ", " << observer.observations * pow_fading / age
                << ", " << observer.time_added
                << ", " << observer.time_touched << ") ";
        // observer.printData();
        std::cout << std::endl;
    }
    std::cout << std::endl;
};

template<typename FloatType>
void SDOcluststream<FloatType>::Observer::printData() const {
    std::cout << "[ ";
    for (const auto& value : data) {
        std::cout << value << " ";
    }
    std::cout << "]";
};

template<typename FloatType>
void SDOcluststream<FloatType>::Observer::printColorObservations(
        FloatType now, 
        FloatType fading_cluster) const {

    std::cout << std::endl << "Color Observations Observer: " << index << ": ";
    for (auto& entry : color_observations) {
        std::cout << "(" << entry.first << "," << entry.second * std::pow(fading_cluster, now - time_cluster_touched) << ") ";
    }
    std::cout << std::endl;
};

template<typename FloatType>
void SDOcluststream<FloatType>::Observer::printColorDistribution() const {
    std::cout << std::endl << "Color Distribution Observer: " << index << ": ";
    for (auto& entry : color_distribution) {
        std::cout << "(" << entry.first << "," << entry.second << ") ";
    }
    std::cout << std::endl;
};


#endif  // SDOCLUSTSTREAM_PRINT_H
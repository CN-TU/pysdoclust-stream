#ifndef TPSDOSTREAMCLUST_PRINT_H
#define TPSDOSTREAMCLUST_PRINT_H

#include "tpSDOstreamclust_observer.h"

template<typename FloatType>
void tpSDOstreamclust<FloatType>::printClusters() {
    for (const auto& cluster : clusters) {
        cluster.printColor();
        cluster.printObserverIndices();            
        cluster.printDistribution();
    }
};

template<typename FloatType>
void tpSDOstreamclust<FloatType>::printObservers(
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
void tpSDOstreamclust<FloatType>::Observer::printData() const {
    std::cout << "[ ";
    for (const auto& value : data) {
        std::cout << value << " ";
    }
    std::cout << "]";
};

template<typename FloatType>
void tpSDOstreamclust<FloatType>::Observer::printColorObservations(
        FloatType now, 
        FloatType fading_cluster) const {

    std::cout << std::endl << "Color Observations Observer: " << index << ": ";
    for (auto& entry : color_observations) {
        std::cout << "(" << entry.first << "," << entry.second << ") ";
    }
    std::cout << std::endl;
};

template<typename FloatType>
void tpSDOstreamclust<FloatType>::Observer::printColorDistribution() const {
    std::cout << std::endl << "Color Distribution Observer: " << index << ": ";
    FloatType sum(0);
    for (auto& entry : color_distribution) {
        sum += entry.second;
        std::cout << "(" << entry.first << "," << entry.second << ") ";
    }
    std::cout << "sum " << sum << std::endl;
};


#endif  // TPSDOSTREAMCLUST_PRINT_H
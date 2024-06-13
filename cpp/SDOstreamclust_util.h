#ifndef SDOSTREAMCLUST_UTIL_H
#define SDOSTREAMCLUST_UTIL_H

#include "SDOstreamclust_observer.h"

template<typename FloatType>
bool SDOstreamclust<FloatType>::hasEdge(
        FloatType distance, 
        const MapIterator& it) {
    return distance < (zeta * it->h + (1 - zeta) * h);
};

template<typename FloatType>
FloatType SDOstreamclust<FloatType>::calcBatchAge(const std::vector<FloatType>& time_data, FloatType score) {
    FloatType age(0);
    for (std::size_t i = 0; i < time_data.size(); ++i) {
        if (i > 0) {
            age *= std::pow(fading, time_data[i] - time_data[i - 1]);
        }
        age += score;
    }
    return age;
}

template<typename FloatType>
class SDOstreamclust<FloatType>::BinomialCalculator {
  private:
    // Use outer and inner sizes for construction
    std::size_t outerSize;
    std::size_t innerSize;
    std::vector<std::vector<long long>> cache;

    long long calc_long(std::size_t n, std::size_t k) {        
        // If k is 0 or equal to n, return 1
        if (k == 0 || k == n) {
            return 1;
        }
        // Check if the value is already calculated
        
        if (cache[n][k] != -1.0) {            
            return cache[n][k];
        }
        // Calculate binomial coefficient recursively and store it in the cache
        FloatType result = calc_long(n - 1, k - 1) + calc_long(n - 1, k);
        cache[n][k] = result;
        return result;
    }

  public:
    // Constructor with outer and inner size arguments
    BinomialCalculator(std::size_t maxN, std::size_t maxK) : outerSize(maxN+1), innerSize(maxK+1), cache(outerSize, std::vector<long long>(innerSize, -1.0)) {}

    FloatType calc(std::size_t n, std::size_t k) {
        return static_cast<FloatType>( calc_long(n,k) );
    }
    // Clear the cache
    void clearCache() {
        cache.clear();
    }
};

template<typename FloatType>
void SDOstreamclust<FloatType>::setModelParameters(
        std::size_t& current_observer_cnt, std::size_t&current_observer_cnt2,
        std::size_t& active_threshold, std::size_t& active_threshold2,
        std::size_t& current_neighbor_cnt, std::size_t& current_neighbor_cnt2,
        std::size_t& current_e,
        std::size_t& chi,
        bool print) {

    current_observer_cnt = observers.size();
    current_observer_cnt2 = observers.size()-1;

    active_threshold = (current_observer_cnt - 1) * active_observers; // active_threshold+1 active observers
    active_threshold2 = (observers.size() == 1) ? 0 : (current_observer_cnt2 - 1) * active_observers; // active_threshold+1 active observers

    current_neighbor_cnt = (observers.size() == observer_cnt) ?
                        neighbor_cnt :
                        static_cast<std::size_t>((current_observer_cnt - 1) / static_cast<FloatType>(observer_cnt - 1) * neighbor_cnt + 1);
    current_neighbor_cnt2 = (observers.size() == 1) ?
                        0 :
                        static_cast<std::size_t>((current_observer_cnt2 - 1) / static_cast<FloatType>(observer_cnt - 1) * neighbor_cnt + 1);
    
    current_e = (observers.size() == observer_cnt) ?
            e :
            static_cast<size_t>((current_observer_cnt - 1) / static_cast<FloatType>(observer_cnt - 1) * e + 1);

    int current_chi_min = (observers.size() == observer_cnt) ?
                    chi_min :
                    static_cast<std::size_t>((current_observer_cnt - 1) / static_cast<FloatType>(observer_cnt - 1) * chi_min + 1);
    chi = std::max(static_cast<std::size_t>(current_observer_cnt * chi_prop), static_cast<std::size_t>(current_chi_min));
    
    if (print) {
        std::cout << std::endl;
        std::cout << "Observers: " << current_observer_cnt << ", " << current_observer_cnt2;
        std::cout << ", Active Observers: " << active_threshold + 1 << ", " << active_threshold2 + 1;
        std::cout << ", Neighbors: " << current_neighbor_cnt << ", " << current_neighbor_cnt2;
        std::cout << ", e: " << current_e;
        std::cout << ", chi: " << chi;
        std::cout << std::endl;     
    }            
};

#endif  // SDOSTREAMCLUST_UTIL_H
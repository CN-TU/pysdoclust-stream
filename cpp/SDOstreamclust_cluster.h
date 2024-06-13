#ifndef SDOSTREAMCLUST_CLUSTER_H
#define SDOSTREAMCLUST_CLUSTER_H

template<typename FloatType>
struct SDOstreamclust<FloatType>::ClusterModel {
    int color;
    FloatType color_score; // score of set color
    IndexSetType cluster_observers;
    std::unordered_map<int, FloatType> color_distribution;

    ClusterModel() {} // Default constructor

    ClusterModel(const IndexSetType& cluster_observers, const IteratorMapType& indexToIterator) : color(0), color_score(FloatType(0)), cluster_observers(cluster_observers), color_distribution() {
        calcColorDistribution(indexToIterator);
        setColor();
    }

    // Member function to calculate color distribution
    void calcColorDistribution(const IteratorMapType& indexToIterator) {
        // Clear existing color distribution
        color_distribution.clear();
        // Iterate over observers and accumulate color distributions
        for (const int& id: cluster_observers) {
            auto iIt = indexToIterator.find(id);
            if (iIt != indexToIterator.end()) {
                const MapIterator& it = iIt->second;   
                const Observer& observer = *it; // Dereference the iterator to get the Observer

                // Add color_distribution of the observer to colorDistribution
                for (const auto& entry : observer.color_distribution) {
                    color_distribution[entry.first] += entry.second;
                }
            } else {
                std::cerr << "Error (calcColorDistribution): id " << id << " not found in indexToIterator" << std::endl;
            }                
        }
    }

    // void printDistribution() const {
    //     std::cout << "Cluster Distribution: " << std::endl;
    //     for (const auto& entry : color_distribution) {
    //         std::cout << "(" << entry.first << ", " << entry.second << ") ";
    //     }
    //     std::cout << std::endl;
    // }
    // void printColor() const {
    //     // Print color and color_score
    //     std::cout << std::endl << "Color: " << color << ", Score: " << color_score << std::endl;
    // }
    // void printObserverIndices() const {
    //     std::cout << "Cluster Indices: " << std::endl;
    //     for (const int& id : cluster_observers) {
    //         std::cout << id << " ";
    //     }
    //     std::cout << std::endl;
    // }

    void setColor() {
        if (!color_distribution.empty()) {
            // Find the iterator with the maximum value in color_distribution
            auto maxIt = std::max_element(color_distribution.begin(), color_distribution.end(),
                [](const auto& a, const auto& b) {
                    if (a.second == b.second) {
                        return a.first > b.first;
                    }
                    return a.second < b.second;
                });

            // Set the color to the key with the maximum value
            color = maxIt->first;
            color_score = color_distribution[color];
        } 
    }

    void setColor(
            const std::unordered_set<int>& takenColors) {

        color = 0;
        color_score = FloatType(0);

        if (!color_distribution.empty()) {
            // Find the keys in color_distribution that are not in takenColors
            std::unordered_map<int, FloatType> difference;
            std::copy_if(
                color_distribution.begin(), color_distribution.end(),
                std::inserter(difference, difference.begin()),
                [&takenColors](const auto& entry) {
                    return takenColors.find(entry.first) == takenColors.end();
                }
            );

            // Check if the difference is non-empty
            if (!difference.empty()) {
                // Find the iterator with the maximum value in colorDistribution,
                // excluding colors in the difference set
                auto maxIt = std::max_element(difference.begin(), difference.end(),
                    [](const auto& a, const auto& b) {
                        return (a.second == b.second) ? (a.first > b.first) : (a.second < b.second);
                    });

                // Set the color to the key with the maximum value
                color = maxIt->first;
                color_score = color_distribution[color];
            }
        }
    }

    void setColor(
            int c) {
        color = c;
        color_score = FloatType(0);
    }
};

template<typename FloatType>
struct SDOstreamclust<FloatType>::ClusterModelCompare {
    bool operator()(const ClusterModel& CM_a, const ClusterModel& CM_b) const {
        return (CM_a.color_score == CM_b.color_score) ? 
            CM_a.cluster_observers.size() > CM_b.cluster_observers.size() :
            CM_a.color_score > CM_b.color_score;
    }
};

#endif  // SDOSTREAMCLUST_CLUSTER_H
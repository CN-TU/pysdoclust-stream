#include "MTree.h"
#include <iostream>


// Define the vector type and distance function for your specific needs


// This function calculates the Euclidean distance between two vectors
double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

// Function to generate random double values within a range
double random_double(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

int main() {
    // Create an MTree with a minimum node size of 5 and a maximum of 10 elements
    // You can adjust these values based on your expected data size and performance requirements.
    MTree<std::vector<double>, long long> tree(euclideanDistance);

    // Number of data points to generate
    const int num_points = 5000; // Change this to the desired number of points

    // Generate random data points
    std::vector<std::vector<double>> data;
    for (int i = 0; i < num_points; ++i) {
        std::vector<double> point;
        point.push_back(random_double(0.0, 10.0)); // Generate random x-coordinate
        point.push_back(random_double(0.0, 10.0)); // Generate random y-coordinate
        data.push_back(point);
    }

    std::unordered_map<long long, MTree<std::vector<double>, long long>::iterator> iteratorMap;
    std::unordered_map<long long, std::vector<double>> pointMap;

    long long id = 0;
    for (auto& point : data) {
        std::pair<std::vector<double>, long long> input(point, id);
        auto it = tree.insert(tree.end(), std::move(input));
        iteratorMap[id] = it;
        pointMap[id] = point;
        id++;
    }

    // Check if points and iterators dereferenced from the map still match
    for (const auto& entry : iteratorMap) {
        long long id = entry.first;
        auto it = entry.second;
        if (it->second!=id) {
            std::cout << "ID: " << id << " " << it->second << ", Point: [";
            for (const auto& coord : it->first) {
                std::cout << coord << ", ";
            }
            std::cout << "] does not matche original point: [";
            for (const auto& coord : pointMap[id]) {
                std::cout << coord << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }

    // Choose a random ID to erase
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<long long> dis(0, num_points - 1); // ID range from 0 to (id - 1)
    
    const int num_iterations = num_points / 100; // Adjust this value based on your requirement
    for (int i = 0; i < num_iterations; ++i) {
        long long randomId = dis(gen);
        // Erase the randomly chosen ID
        auto itToErase = iteratorMap.find(randomId);
        if (itToErase != iteratorMap.end()) {
            tree.erase(itToErase->second);
            pointMap.erase(randomId);
            iteratorMap.erase(itToErase);
            // std::cout << "ID " << randomId << " erased successfully." << std::endl;
        } else {
            // std::cout << "ID " << randomId << " not found." << std::endl;
        }
    }

    // Check if points and iterators dereferenced from the map still match
    for (const auto& entry : iteratorMap) {
        long long id = entry.first;
        auto it = entry.second;
        if (it->second!=id) {
            std::cout << "ID: " << id << " " << it->second << ", Point: [";
            for (const auto& coord : it->first) {
                std::cout << coord << ", ";
            }
            std::cout << "] does not matche original point: [";
            for (const auto& coord : pointMap[id]) {
                std::cout << coord << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }



    return 0;
}

// int main() {
    // // Create an MTree with a minimum node size of 5 and a maximum of 10 elements
    // // You can adjust these values based on your expected data size and performance requirements.
    // MTree<std::vector<double>,long long> tree(euclideanDistance);

    // // Example data points (replace with your actual data)
    // std::vector<std::vector<double>> data = {
    //     {1.0, 2.0},
    //     {3.0, 4.0},
    //     {5.0, 6.0},
    //     {7.0, 8.0},
    //     {2.0, 3.0},
    //     {0.0, 1.0}
    //     // Add more data points as needed
    // };

    // std::unordered_map<long long, MTree<std::vector<double>, long long>::iterator> iteratorMap;
    // std::unordered_map<long long, std::vector<double>> pointMap;

    // long long i = 0;
    // for (auto point: data) {
    //     std::pair<std::vector<double>, long long> input(point, i++);
    //     auto it = tree.insert(tree.begin(), std::move(input));
    //     iteratorMap[i] = it;
    //     pointMap[i] = point;
    // }

    // // Insert the data points into the MTree
    // long long i = 0;
    // std::vector<MTree<std::vector<double>, long long>::iterator> iterators;
    // for (auto point : data) {
    //     std::pair<std::vector<double>, long long> input(point, i++);
    //     iterators.emplace_back(tree.insert(tree.end(), std::move(input)));
    // }

    // // Example usage of kNN search
    // std::vector<double> queryPoint = {2.5, 3.5}; // Replace with your actual query point
    // unsigned int k = 5; // Number of nearest neighbors to find
    // std::vector<std::pair<MTree<std::vector<double>, long long>::iterator, double>> nearestNeighbors = tree.knnSearch(queryPoint, k);

    // // Output the nearest neighbors
    // for (const auto& neighbor : nearestNeighbors) {
    //     std::cout << "Nearest neighbor: " << neighbor.first->second << " with distance: " << neighbor.second << std::endl;
    // }
    // std::cout << std::endl;

    // auto it = iterators[2];
    // std::vector<double> newPoint0 = {2.5, 3.5}; // Replace with your new point    

    // class Updater {
    //     std::vector<double> new_data;
    //     long long new_key;
    // public:
    //     Updater(std::vector<double> new_data, long long new_key) : new_data(new_data), new_key(new_key) {}
    //     void operator() (std::vector<double>& vector, long long& key) {
    //         int i = 0;
    //         for (double& element : vector) {
    //             element = new_data[i];
    //             i++;
    //         }
    //         key = new_key;
    //     }
    // };

    // Updater updater(newPoint0, i++);
    // tree.modify(it, updater);

    // nearestNeighbors = tree.knnSearch(queryPoint, k);

    // // Output the nearest neighbors
    // for (const auto& neighbor : nearestNeighbors) {
    //     std::cout << "Nearest neighbor: " << neighbor.first->second << " with distance: " << neighbor.second << std::endl;
    // }
    // std::cout << std::endl;

    // // Replace the existing point with a new point (modify as needed)
    // std::vector<double> newPoint = {2.5, 2.5}; // Replace with your new point
    // tree.insert(iterators[3], std::make_pair(newPoint, i++)); // Update the point

    // // Output the nearest neighbors
    // nearestNeighbors = tree.knnSearch(queryPoint, 6);
    // for (const auto& neighbor : nearestNeighbors) {
    //     std::cout << "Nearest neighbor: " << neighbor.first->second << " with distance: " << neighbor.second << std::endl;
    // }

    
    // for (auto it = tree.begin(); it != tree.end(); ++it) {
    //     auto node = *it;
    //     std::cout << "Key: " << node.second << std::endl;
    // }
    
    // double radius = 1.0;
    // auto rangeNeighbors = tree.rangeSearch(queryPoint, radius);

    // while (!rangeNeighbors.atEnd()) {
    //     // Dereference the iterator to get the current element
    //     auto neighbor = *rangeNeighbors;

    //     std::cout << "neighbor: " << neighbor.first->second << " with distance: " << neighbor.second << std::endl;
        
    //     ++rangeNeighbors;
    // }

    // queryPoint = {2.5, 2.5}; // Replace with your new point
    // radius = 0.0;
    // rangeNeighbors = tree.rangeSearch(queryPoint, radius);

    // std::cout << "l" << std::endl;

    // while (!rangeNeighbors.atEnd()) {
    //     // Dereference the iterator to get the current element
    //     auto neighbor = *rangeNeighbors;

    //     std::cout << "neighbor: " << neighbor.first->second << " with distance: " << neighbor.second << std::endl;
        
    //     ++rangeNeighbors;
    // }

    // tree.erase(rangeNeighbors)

//     return 0;
// }

#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>

#include "cpp/Vector.h"
#include "cpp/SDOcluststream.h"
#include "cpp/tpSDOsc.h"


std::vector<double> generateRandomDoubles(std::size_t N, double minVal = 0.0, double maxVal = 1.0) {
    std::vector<double> randomNumbers(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(minVal, maxVal);
    std::generate(randomNumbers.begin(), randomNumbers.end(), [&]() { return dis(gen); });
    return randomNumbers;
}

std::vector<Vector<double>> generateRandomVectors(std::size_t N, std::size_t k, double minVal = 0.0, double maxVal = 1.0) {
    std::vector<Vector<double>> randomVectors;
    std::vector<double> randomNumbers = generateRandomDoubles(N * k, minVal, maxVal);

    randomVectors.reserve(N);
    auto it = randomNumbers.begin();
    for (std::size_t i = 0; i < N; ++i) {
        randomVectors.emplace_back(&(*it), k);
        std::advance(it, k);
    }

    return randomVectors;
}

int main() {
    // Set data parameters
    int m = 50; // Number of Batches
    int n = 200; // Batch size
    int k = 5;   // Dimensionality

    // Seed the random number generator for reproducibility
    // std::random_device rd;
    // std::mt19937 gen(rd());

    // initialise algorithm
    SDOcluststream<double> sdoclust(
            500, 
            2000, 
            0.3f, 
            6, // x
            7, // chi
            0.15f,
            0.6f, // zeta
            7, // e
            5.0f);

    // Measure time taken for fitPredict
    auto start0 = std::chrono::steady_clock::now();
    for (int i = 0; i < m; ++i) {
        std::vector<Vector<double>> data = generateRandomVectors(n, k);

        // Generate time_data for this batch
        std::vector<double> time_data(n);  // Create a vector of size n
        std::iota(time_data.begin(), time_data.end(), i * n);  // Fill with values starting from i * n, incrementing by 1

        // Measure time taken for fitPredict
        auto start = std::chrono::steady_clock::now();
        sdoclust.fitPredict(data, time_data);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time taken for fitPredict: " << duration.count() << " milliseconds" << std::endl;
    }

    auto end0 = std::chrono::steady_clock::now();
    auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0);
    std::cout << "Time taken for fitPredict: " << duration0.count() << " milliseconds" << std::endl;

    tpSDOsc<double> tpsdoclust(
            500, 
            2000, 
            0.3f, 
            6, // x
            7, // chi
            0.15f,
            0.6f, // zeta
            7, // e
            1, // freq_bins
            10000.0f, // max_freq
            5.0f // outlier threshold            
            );

    // Measure time taken for fitPredict
    auto start1 = std::chrono::steady_clock::now();
    for (int i = 0; i < m; ++i) {
        std::vector<Vector<double>> data = generateRandomVectors(n, k);

        // Generate time_data for this batch
        std::vector<double> time_data(n);  // Create a vector of size n
        std::iota(time_data.begin(), time_data.end(), i * n);  // Fill with values starting from i * n, incrementing by 1

        // Measure time taken for fitPredict
        auto start = std::chrono::steady_clock::now();
        tpsdoclust.fitPredict(data, time_data);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time taken for fitPredict: " << duration.count() << " milliseconds" << std::endl;
    }

    auto end1 = std::chrono::steady_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "Time taken for fitPredict: " << duration1.count() << " milliseconds" << std::endl;

    return 0;
}

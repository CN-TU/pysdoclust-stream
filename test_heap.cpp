#include "KDistanceHeap.h"
#include <iostream>

void print(KDistanceHeap<int, double>& kdHeap) {
    std::cout << std::endl;
    // Print elements from kLowest heap    
    std::cout << "Elements in kLowest heap:" << std::endl;    
    for (auto it = kdHeap.ordered_begin(); it != kdHeap.ordered_end(); ++it) {
        std::cout << it->first << ": " << it->second << std::endl;
    }
    

    // Print elements from others heap
    std::cout << "\nElements in others heap:" << std::endl;
    for (auto it = kdHeap.ordered_begin2(); it != kdHeap.ordered_end2(); ++it) {
        std::cout << it->first << ": " << it->second << std::endl;
    }
};

int main() {
    // Create a KDistanceHeap with k = 3
    KDistanceHeap<int, double> kdHeap(3);

    // Insert some elements
    kdHeap.insert(1, 0.5);
    print(kdHeap);
    kdHeap.insert(2, 0.2);
    print(kdHeap);
    kdHeap.insert(3, 0.8);
    print(kdHeap);
    kdHeap.insert(4, 0.6);
    print(kdHeap);
    kdHeap.insert(5, 0.1);
    print(kdHeap);
    kdHeap.insert(6, 0.3);
    print(kdHeap);
    kdHeap.insert(7, 0.9);
    print(kdHeap);
    kdHeap.insert(8, 0.15);
    print(kdHeap);
    kdHeap.insert(9, 0.2);
    print(kdHeap);
    kdHeap.insert(0, 0.2);
    print(kdHeap);

    kdHeap.balance(7);
    print(kdHeap);
    kdHeap.balance(4);
    print(kdHeap);

    kdHeap.update(7, 10, 0.1);
    print(kdHeap);
    kdHeap.update(5, 11, 1.0);
    print(kdHeap);

    kdHeap.erase(1);
    print(kdHeap);
    kdHeap.erase(9);
    print(kdHeap);
    kdHeap.erase(8);
    print(kdHeap);

    return 0;
}
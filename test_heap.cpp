#include "KHeap.h"
#include <iostream>

void print(KBufferHeap<double,int>& kdHeap) {
    std::cout << std::endl;
    // Print elements from kLowest heap    
    std::cout << "Elements in kLowest heap:" << std::endl;    
    for (auto it = kdHeap.ordered_begin(); it != kdHeap.ordered_end(); ++it) {
        std::cout << it->second << ": " << it->first << ", ";
    }
    std::cout << std::endl;    

    // Print elements from others heap
    std::cout << "\nElements in others heap:" << std::endl;
    for (auto it = kdHeap.ordered_begin2(); it != kdHeap.ordered_end2(); ++it) {
        std::cout << it->second << ": " << it->first << ", ";
    }
    std::cout << std::endl;

    // Print elements from mirrors heap
    std::cout << "\nElements in mirrors heap:" << std::endl;
    for (auto it = kdHeap.ordered_rbegin2(); it != kdHeap.ordered_rend2(); ++it) {
        std::cout << it->second << ": " << it->first << ", ";
    }
    std::cout << std::endl;
};

int main() {
    // Create a KDistanceHeap with k = 3
    KBufferHeap<double, int> kdHeap(3, 8);

    // Insert some elements

    kdHeap.insert(1, 0.5);
    kdHeap.insert(2, 0.2);    
    kdHeap.insert(3, 0.8);
    kdHeap.insert(4, 0.6);    
    kdHeap.insert(5, 0.1);    
    kdHeap.insert(6, 0.3);
    kdHeap.insert(7, 0.9);
    kdHeap.insert(8, 0.15);
    kdHeap.insert(9, 0.2);
    kdHeap.insert(0, 0.2);
    kdHeap.insert(10, 0.3);
    kdHeap.insert(11, 0.5);
    kdHeap.insert(12, 0);
    kdHeap.insert(13, 1);
    kdHeap.insert(14, 0.2);
    kdHeap.insert(15, 0.3);
    kdHeap.print();

    print(kdHeap);

    std::cout << std::endl << "Balance: " << std::endl;

    kdHeap.balanceK(5);
    print(kdHeap);
    kdHeap.balanceK(2);
    print(kdHeap);
    // std::cout << "Median:" << kdHeap.median() << std::endl;

    // kdHeap.setK(5);
    // print(kdHeap);
    // kdHeap.setK(2);
    // print(kdHeap);

    // kdHeap.update(7, 10, 0.1);
    // print(kdHeap);
    // kdHeap.update(5, 11, 1.0);
    // print(kdHeap);

    // kdHeap.erase(1);
    // print(kdHeap);
    // kdHeap.erase(9);
    // print(kdHeap);
    // kdHeap.erase(8);
    // print(kdHeap);
    // kdHeap.erase(2);
    // print(kdHeap);
    // kdHeap.erase(4);
    // print(kdHeap);
    // kdHeap.erase(6);
    // print(kdHeap);
    // kdHeap.erase(10);
    // print(kdHeap);

    return 0;
}
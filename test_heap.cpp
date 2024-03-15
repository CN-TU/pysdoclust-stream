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
    KBufferHeap<double, int> kdHeap(1, 11);

    // Insert some elements

    kdHeap.insert(6, 0.621);
    kdHeap.insert(2, 0.972);   
    kdHeap.deactivate(2); 
    kdHeap.insert(11, 0.738);
    kdHeap.insert(12, 0.858);   
    kdHeap.print(); 
    kdHeap.insert(13, 0.59);    
    kdHeap.print();
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

    // std::cout << std::endl << "Balance: " << std::endl << std::endl;

    // kdHeap.balanceK(2);
    // kdHeap.print();
    // kdHeap.balanceK(5);
    // kdHeap.print();
    
    // std::cout << "Update: " << std::endl << std::endl;
    // kdHeap.update(6, 16, 0.1);
    // kdHeap.print();
    // kdHeap.update(5, 17, 0.4);
    // kdHeap.print();

    // std::cout << "erase: " << std::endl << std::endl;
    // kdHeap.erase(5);
    // kdHeap.erase(17);
    // kdHeap.erase(9);
    // kdHeap.print();

    // std::cout << "deactivate: " << std::endl << std::endl;
    // kdHeap.deactivate(11);
    // kdHeap.deactivate(16);
    // kdHeap.deactivate(9);
    // kdHeap.print();

    // std::cout << "swap: " << std::endl << std::endl;
    // kdHeap.swap_active(10, 11);
    // kdHeap.swap_active(15, 16);
    // kdHeap.print();

    // std::cout << "activate: " << std::endl << std::endl;
    // kdHeap.activate(10);
    // kdHeap.activate(15);
    // kdHeap.activate(9);
    // kdHeap.print();

    return 0;
}
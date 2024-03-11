// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU pair_lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_KDISTANCEHEAP_H
#define DSALMON_KDISTANCEHEAP_H

#include <iostream>
#include <vector>
#include <utility>
#include <boost/heap/binomial_heap.hpp>

// Define a template class for KDistanceHeap
template<typename T, typename FloatType>
class KDistanceHeap {
  public:  
    typedef std::pair<T, FloatType> PairType;
  private:  
    struct GreaterPair {    
        bool operator()(const PairType& lhs, const PairType& rhs) const {
            return (lhs.second != rhs.second) ? (lhs.second > rhs.second) : (lhs.first > rhs.first);
        }
    } greater;
    // Define custom comparator for max heap
    struct LessPair {
        bool operator()(const PairType& lhs, const PairType& rhs) const {
            return (lhs.second != rhs.second) ? (lhs.second < rhs.second) : (lhs.first < rhs.first);
        }
    } less;
  public:
    typedef boost::heap::binomial_heap<PairType, boost::heap::compare<LessPair>> MaxHeap;    
    typedef typename MaxHeap::handle_type handle_type;
    typedef typename MaxHeap::iterator iterator;  // Iterator for kLowest heap
    typedef typename MaxHeap::ordered_iterator ordered_iterator;  // Ordered iterator for kLowest heap    
    typedef boost::heap::binomial_heap<PairType, boost::heap::compare<GreaterPair>> MinHeap;
    typedef typename MinHeap::handle_type handle_type2;
    typedef typename MinHeap::iterator iterator2;  // Iterator for others heap
    typedef typename MinHeap::ordered_iterator ordered_iterator2;  // Ordered iterator for others heap   
  private:
    int k;
    MaxHeap kLowest; // Min heap for k lowest elements, largest on top
    std::unordered_map<T, handle_type> kLowestMap;
    MinHeap others; // Max heap for other elements, smalles on top    
    std::unordered_map<T, handle_type2> othersMap;    

  public:
    // Constructor to initialize with a specific value of k
    KDistanceHeap(int k) : k(k) {} 

    // Insert a new element with its distance
    void insert(
            const T& element, 
            const FloatType& distance) {
        PairType newPair(element, distance);
        if (kLowest.size() < k) {
            kLowestMap[element] = kLowest.push(newPair);
        } else {            
            // if (pair_greater<PairType>(kLowest.top(), newPair)) { 
            if (greater(kLowest.top(), newPair)) {     
                PairType pair = kLowest.top();
                handle_type ha = kLowestMap[pair.first];
                kLowest.update(ha, newPair);
                kLowestMap.erase(pair.first);
                kLowestMap[element] = ha;
                // push old greatest Pair to others
                othersMap[pair.first] = others.push(pair);                
            } else {                
                othersMap[element] = others.push(newPair);
            }
        }
    }
    void insert(const PairType& newPair) { insert(newPair.first, newPair.second); }
    
    void erase(
            const T& element) {
        if (kLowestMap.count(element)>0) {
            handle_type ha = kLowestMap[element];            
            kLowestMap.erase(element);
            if (!others.empty()) {
                // pop smallest from others
                PairType pair = others.top();
                others.pop();
                othersMap.erase(pair.first); 
                // add to kLowest replace
                kLowest.update(ha, pair);
                kLowestMap[pair.first] = ha;
            }
        } else { // must be in othersMap
            handle_type2 ha = othersMap[element];    
            others.erase(ha);
            othersMap.erase(element);            
        }        
    }

    void update(
            const T& elementToUpdate,
            const T& element, 
            const FloatType& distance) {
        PairType newPair(element, distance);
        if (kLowestMap.count(elementToUpdate)>0) {
            if ( greater(kLowest.top(), newPair) || 
                (greater(others.top(), newPair) && kLowest.top().first == elementToUpdate) ) {
                handle_type ha = kLowestMap[elementToUpdate];                
                kLowest.update(ha, newPair);
                kLowestMap.erase(elementToUpdate);
                kLowestMap[element] = ha;
            } else {
                erase(elementToUpdate);
                insert(element, distance);
            }
        } else {
            if ( less(others.top(), newPair) ||
                (less(kLowest.top(), newPair) && others.top().first == elementToUpdate) ) {
                handle_type2 ha = othersMap[elementToUpdate];                
                others.update(ha, newPair);
                othersMap.erase(elementToUpdate);
                othersMap[element] = ha;
            } else {
                erase(elementToUpdate);
                insert(element, distance);
            }
        }
    }

    void balance(int k_new) {
        k = k_new;
        while (kLowest.size() > k) {
            PairType pair = kLowest.top();
            handle_type ha = kLowestMap[pair.first];
            kLowest.erase(ha);
            kLowestMap.erase(pair.first);            
            othersMap[pair.first] = others.push(pair);
        }
        while (kLowest.size() < k && !others.empty()) {
            PairType pair = others.top();
            handle_type2 ha = othersMap[pair.first];
            others.erase(ha);
            othersMap.erase(pair.first);
            kLowestMap[pair.first] = kLowest.push(pair);
        }
    }

    int getK() {
        return k;
    }

    FloatType top() {
        return kLowest.top().second;
    }

    // Get iterators for kLowest heap
    iterator begin() const { return kLowest.begin(); }    
    iterator end() const { return kLowest.end(); }
    ordered_iterator ordered_begin() const { return kLowest.ordered_begin(); }
    ordered_iterator ordered_end() const { return kLowest.ordered_end(); }

    // Get iterators for others
    iterator2 begin2() const { return others.begin(); }    
    iterator2 end2() const { return others.end(); }
    ordered_iterator2 ordered_begin2() const { return others.ordered_begin(); }
    ordered_iterator2 ordered_end2() const { return others.ordered_end(); }
};

#endif
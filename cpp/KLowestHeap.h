// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU pair_lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_KLOWESTHEAP_H
#define DSALMON_KLOWESTHEAP_H

#include <iostream>
#include <vector>
#include <utility>
#include <unordered_map>
#include <boost/heap/binomial_heap.hpp>

// Define a template class for KLowestHeap
template<typename T, typename FloatType>
class KLowestHeap {
  public:  
    typedef std::pair<T, FloatType> PairType;
  private:  
    struct greater {    
        bool operator()(const PairType& lhs, const PairType& rhs) const {
            return (lhs.second != rhs.second) ? (lhs.second > rhs.second) : (lhs.first > rhs.first);
        }
    };
    // Define custom comparator for max heap
    struct less {
        bool operator()(const PairType& lhs, const PairType& rhs) const {
            return (lhs.second != rhs.second) ? (lhs.second < rhs.second) : (lhs.first < rhs.first);
        }
    };
  public:
    typedef boost::heap::binomial_heap<PairType, boost::heap::compare<less>> MaxHeap;    
    typedef typename MaxHeap::handle_type handle_type;
    typedef typename MaxHeap::iterator iterator;  // Iterator for kLowest heap
    typedef typename MaxHeap::ordered_iterator ordered_iterator;  // Ordered iterator for kLowest heap    
    typedef boost::heap::binomial_heap<PairType, boost::heap::compare<greater>> MinHeap;
    typedef typename MinHeap::handle_type handle_type2;
    typedef typename MinHeap::iterator iterator2;  // Iterator for others heap
    typedef typename MinHeap::ordered_iterator ordered_iterator2;  // Ordered iterator for others heap   
  private:
    int k;
    MaxHeap kLowest; // Max heap for k lowest elements, largest on top
    std::unordered_map<T, handle_type> kLowestMap;
    MinHeap others; // Min heap for other elements, smallest on top    
    std::unordered_map<T, handle_type2> othersMap;

    PairType replace0(
            PairType newPair,
            MaxHeap& heap,
            std::unordered_map<int, handle_type>& map) {
        PairType pair = heap.top();
        handle_type ha = map[pair.first];
        heap.update(ha, newPair);
        map.erase(pair.first);
        map[newPair.first] = ha;
        return pair;
    } 
    PairType replace0(
            PairType newPair,
            MinHeap& heap,
            std::unordered_map<int, handle_type2>& map) {
        PairType pair = heap.top();
        handle_type2 ha = map[pair.first];
        heap.update(ha, newPair);
        map.erase(pair.first);
        map[newPair.first] = ha;
        return pair;
    } 
    void update0(
            int element,
            PairType pair,
            MaxHeap& heap,
            std::unordered_map<int, handle_type>& map) {
        handle_type ha = map[element];
        heap.update(ha, pair);
        map.erase(element);
        map[pair.first] = ha;
    }
    void update0(
            int element,
            PairType pair,
            MinHeap& heap,
            std::unordered_map<int, handle_type2>& map) {
        handle_type2 ha = map[element];
        heap.update(ha, pair);
        map.erase(element);
        map[pair.first] = ha;
    }
    void erase0(
            int element,
            MaxHeap& heap,
            std::unordered_map<int, handle_type>& map) {
        handle_type ha = map[element];    
        heap.erase(ha);       
        map.erase(element);
    }
    void erase0(
            int element,
            MinHeap& heap,
            std::unordered_map<int, handle_type2>& map) {
        handle_type2 ha = map[element];    
        heap.erase(ha);       
        map.erase(element);
    }  
  public:
    // Constructor to initialize with a specific value of k
    KLowestHeap(int k) : k(k) {} 

    // Insert a new element with its distance
    void insert(
            const T& element, 
            const FloatType& distance) {
        PairType newPair(element, distance);
        if (kLowest.size() < k) {
            kLowestMap[element] = kLowest.push(newPair);
        } else {  
            if (kLowest.value_comp()(newPair, kLowest.top())) { 
                PairType pair = replace0(newPair, kLowest, kLowestMap); 
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
            if (!others.empty()) {
                // pop smallest from others
                PairType pair = others.top();                
                erase0(pair.first, others, othersMap);
                update0(element, pair, kLowest, kLowestMap);
            } else {
                erase0(element, kLowest, kLowestMap);                
            }       
        } else { // must be in othersMap            
            erase0(element, others, othersMap);             
        }        
    }

    void update(
            const T& elementToUpdate,
            const T& element, 
            const FloatType& distance) {
        PairType newPair(element, distance);
        if (kLowestMap.count(elementToUpdate)>0) {
            if ( kLowest.value_comp()(newPair, kLowest.top()) || 
                (others.value_comp()(others.top(), newPair) && kLowest.top().first == elementToUpdate) ) {
                update0(elementToUpdate, newPair, kLowest, kLowestMap);
            } else {
                erase(elementToUpdate);
                insert(element, distance);
            }
        } else {
            if ( others.value_comp()(newPair, others.top()) ||
                (kLowest.value_comp()(kLowest.top(), newPair) && others.top().first == elementToUpdate) ) {
                update0(elementToUpdate, newPair, others, othersMap);
            } else {
                erase(elementToUpdate);
                insert(element, distance);
            }
        }
    }

    void setK(
            int k_new) {
        k = k_new;
        while (kLowest.size() > k) {
            PairType pair = kLowest.top();
            erase0(pair.first, kLowest, kLowestMap);              
            othersMap[pair.first] = others.push(pair);
        }
        while (kLowest.size() < k && !others.empty()) {
            PairType pair = others.top();
            erase0(pair.first, others, othersMap);            
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

    // Get iterators for others heap
    iterator2 begin2() const { return others.begin(); }    
    iterator2 end2() const { return others.end(); }
    ordered_iterator2 ordered_begin2() const { return others.ordered_begin(); }
    ordered_iterator2 ordered_end2() const { return others.ordered_end(); }
};




template<typename T, typename FloatType>
class KLowestBufferHeap {
  public:  
    typedef std::pair<T, FloatType> PairType;
  private:  
    struct greater {    
        bool operator()(const PairType& lhs, const PairType& rhs) const {
            return (lhs.second != rhs.second) ? (lhs.second > rhs.second) : (lhs.first > rhs.first);
        }
    };
    // Define custom comparator for max heap
    struct less {
        bool operator()(const PairType& lhs, const PairType& rhs) const {
            return (lhs.second != rhs.second) ? (lhs.second < rhs.second) : (lhs.first < rhs.first);
        }
    };
  public:
    typedef boost::heap::binomial_heap<PairType, boost::heap::compare<less>> MaxHeap;    
    typedef typename MaxHeap::handle_type handle_type;
    typedef typename MaxHeap::iterator iterator;  // Iterator for kLowest heap
    typedef typename MaxHeap::ordered_iterator ordered_iterator;  // Ordered iterator for kLowest heap    
    typedef boost::heap::binomial_heap<PairType, boost::heap::compare<greater>> MinHeap;
    typedef typename MinHeap::handle_type handle_type2;
    typedef typename MinHeap::iterator iterator2;  // Iterator for others heap
    typedef typename MinHeap::ordered_iterator ordered_iterator2;  // Ordered iterator for others heap 
  private:
    typedef std::pair<handle_type, FloatType> PairType2;

    int k;
    int buffer_size;
    MaxHeap kLowest; // Max heap for k lowest elements, largest on top
    std::unordered_map<T, PairType2> kLowestMap;
    MinHeap others; // Min heap for other elements, smallest on top       
    std::unordered_map<T, handle_type2> othersMap;     
    MaxHeap mirror;          
    std::unordered_map<T, PairType2> mirrorMap;    
    std::unordered_map<T, FloatType> inactiveMap;

    PairType replace0(
        PairType newPair,
        MaxHeap& heap,
        std::unordered_map<int, PairType2>& map) {
            PairType pair = heap.top();
            handle_type ha = map[pair.first].first;
            heap.update(ha, newPair);
            map.erase(pair.first);
            map[newPair.first] = PairType2(ha, newPair.second);
            return pair;
    } 
    PairType replace0(
        PairType newPair,
        MinHeap& heap,
        std::unordered_map<int, handle_type2>& map) {
            PairType pair = heap.top();
            handle_type2 ha = map[pair.first];
            heap.update(ha, newPair);
            map.erase(pair.first);
            map[newPair.first] = ha;
            return pair;
    } 
    void update0(
            int element,
            PairType pair,
            MaxHeap& heap,
            std::unordered_map<int, PairType2>& map) {
        handle_type ha = map[element].first;
        heap.update(ha, pair);
        map.erase(element);
        map[pair.first] = PairType2(ha, pair.second);
    }
    void update0(
            int element,
            PairType pair,
            MinHeap& heap,
            std::unordered_map<int, handle_type2>& map) {
        handle_type2 ha = map[element];
        heap.update(ha, pair);
        map.erase(element);
        map[pair.first] = ha;
    }
    void erase0(
            int element,
            MaxHeap& heap,
            std::unordered_map<int, PairType2>& map) {
        handle_type ha = map[element].first;    
        heap.erase(ha);       
        map.erase(element);
    }
    void erase0(
            int element,
            MinHeap& heap,
            std::unordered_map<int, handle_type2>& map) {
        handle_type2 ha = map[element];    
        heap.erase(ha);       
        map.erase(element);
    }
  public:
    // Constructor to initialize with a specific value of k and buffer_size
    KLowestBufferHeap(int k, int buffer_size) : k(k), buffer_size(buffer_size) {} 

    // Insert a new element with its distance
    bool insert(
            const T& element, 
            const FloatType& distance,
            bool inactive = false) {
        PairType newPair(element, distance);
        if (inactive) { inactiveMap[element] = distance; return true; }
        if (kLowest.size() < k) {
            kLowestMap[element] = PairType2(kLowest.push(newPair), distance);
            return true;
        } else {   
            if (kLowest.value_comp()(newPair, kLowest.top())) {     
                PairType pair = replace0(newPair, kLowest, kLowestMap);
                if (others.size() < buffer_size) {
                    othersMap[pair.first] = others.push(pair);   
                    mirrorMap[pair.first] = PairType2(mirror.push(pair), pair.second); 
                    return true;            
                } else if (mirror.value_comp()(pair, mirror.top()) ) {
                    PairType oldPair = replace0(pair, mirror, mirrorMap);
                    update0(oldPair.first, pair, others, othersMap);
                    return true;
                }                
            } else {   
                if (others.size() < buffer_size) {                       
                    othersMap[element] = others.push(newPair);
                    mirrorMap[element] = PairType2(mirror.push(newPair), distance);
                    return true;
                } else if (mirror.value_comp()(newPair, mirror.top()) ) {
                    PairType oldPair = replace0(newPair, mirror, mirrorMap);
                    update0(oldPair.first, newPair, others, othersMap);
                    return true;
                }
            }
        }
        return false;
    }
    bool insert(const PairType& newPair, bool inactive = false) { return insert(newPair.first, newPair.second, inactive); }
    
    bool erase(
            const T& element) {
        if (kLowestMap.count(element)>0) {            
            if (!others.empty()) {
                // pop smallest from others
                PairType pair = others.top();                
                erase0(pair.first, others, othersMap);
                erase0(pair.first, mirror, mirrorMap);
                // update kLowest
                update0(element, pair, kLowest, kLowestMap);
            } else {
                erase0(element, kLowest, kLowestMap);                
            }  
            return true;           
        } else if (othersMap.count(element)>0) {
            erase0(element, others, othersMap);  
            erase0(element, mirror, mirrorMap);  
            return true;      
        } else if (inactiveMap.count(element>0)) {
            inactiveMap.erase(element);
        }
        return false;      
    }

    void update(
            const T& elementToUpdate,
            const T& element, 
            const FloatType& distance) {
        PairType newPair(element, distance);
        if (kLowestMap.count(elementToUpdate)>0) {
            if ( kLowest.value_comp()(newPair, kLowest.top()) || 
                (others.value_comp()(others.top(), newPair) && kLowest.top().first == elementToUpdate) ) {
                update0(elementToUpdate, newPair, kLowest, kLowestMap);
            } else {
                erase(elementToUpdate);
                insert(element, distance);
            }
        } else {
            if ( others.value_comp()(newPair, others.top()) ||
                (kLowest.value_comp()(kLowest.top(), newPair) && others.top().first == elementToUpdate) ) {
                update0(elementToUpdate, newPair, others, othersMap);
                update0(elementToUpdate, newPair, mirror, mirrorMap);
            } else {
                erase(elementToUpdate);
                insert(element, distance);
            }
        }
    }

    bool active(
            const T& element) {
        return inactiveMap.count(element)>0;
    }

    void activate(
            const T& element) {
        FloatType distance = inactiveMap[element];
        inactiveMap.erase(element);
        insert(element, distance);
    }

    void deactivate(
            const T& element) {
        if (mirrorMap.count(element)>0) {
            inactiveMap[element] = mirrorMap[element].second;
        } else {
            inactiveMap[element] = kLowestMap[element].second;
        }
        erase(element);
    }

    void setK(
            int k_new) {  
        if (k_new < k) {
            buffer_size += k - k_new; // increase buffer_size
        } 
        if (k < k_new && (k_new-k) > buffer_size) {
            buffer_size = 0;
        }           
        k = k_new;
        while (kLowest.size() > k) {
            PairType pair = kLowest.top();
            erase0(pair.first, kLowest, kLowestMap);              
            othersMap[pair.first] = others.push(pair); 
            mirrorMap[pair.first] = PairType2(mirror.push(pair), pair.second); 
        }
        while (kLowest.size() < k && !others.empty()) {
            PairType pair = others.top();
            erase0(pair.first, others, othersMap); 
            erase0(pair.first, mirror, mirrorMap);            
            kLowestMap[pair.first] = PairType2(kLowest.push(pair), pair.second);
        }
    }
    void setBufferSize(int bs_new) { buffer_size = bs_new; }

    int getK() {
        return k;
    }

    FloatType top() {
        return kLowest.top().second;
    }

    FloatType bottom() {
        return mirror.top().second;
    }

    bool isin(int element) {
        return (kLowestMap.count(element)>0) || (othersMap.count(element)>0);
    }

    bool empty() {
        return kLowest.empty();
    }
    
    std::size_t size() {
        return kLowest.size();
    }

    std::size_t sizeB() {
        return others.size();
    }

    std::size_t sizeA() {
        return kLowest.size() + others.size();
    }


    FloatType median() {
        ordered_iterator it = kLowest.ordered_begin();
        int steps = k/2 - (k - kLowest.size());
        if (steps > 0) {
            if (k % 2 == 0) {
                FloatType m(0);
                int steps = (k-1)/2 - (k - kLowest.size());
                std::advance(it, steps);
                m += it->second;
                it++;
                m += it->second;
                return 0.5 * m;
            } else {
                std::advance(it, steps);
                return it->second;            
            }
        } else {
            if (!kLowest.empty()) { return it->second; } // highest value
            else { return 0.0; }
        }
    }

    // Get iterators for kLowest heap
    iterator begin() const { return kLowest.begin(); }    
    iterator end() const { return kLowest.end(); }
    ordered_iterator ordered_begin() const { return kLowest.ordered_begin(); }
    ordered_iterator ordered_end() const { return kLowest.ordered_end(); }

    // Get iterators for others heap
    iterator2 begin2() const { return others.begin(); }    
    iterator2 end2() const { return others.end(); }
    ordered_iterator2 ordered_begin2() const { return others.ordered_begin(); }
    ordered_iterator2 ordered_end2() const { return others.ordered_end(); }

    // Get iterators for mirrors heap
    ordered_iterator ordered_rbegin2() const { return mirror.ordered_begin(); }
    ordered_iterator ordered_rend2() const { return mirror.ordered_end(); }
};



// template<typename T, typename FloatType>
// class KLowestBufferHeap {
//   public:  
//     typedef std::pair<T, FloatType> PairType;
//   private:  
//     struct greater {    
//         bool operator()(const PairType& lhs, const PairType& rhs) const {
//             return (lhs.second != rhs.second) ? (lhs.second > rhs.second) : (lhs.first > rhs.first);
//         }
//     };
//     // Define custom comparator for max heap
//     struct less {
//         bool operator()(const PairType& lhs, const PairType& rhs) const {
//             return (lhs.second != rhs.second) ? (lhs.second < rhs.second) : (lhs.first < rhs.first);
//         }
//     };
//   public:
//     typedef boost::heap::binomial_heap<PairType, boost::heap::compare<less>> MaxHeap;    
//     typedef typename MaxHeap::handle_type handle_type;
//     typedef typename MaxHeap::iterator iterator;  // Iterator for kLowest heap
//     typedef typename MaxHeap::ordered_iterator ordered_iterator;  // Ordered iterator for kLowest heap    
//     typedef boost::heap::binomial_heap<PairType, boost::heap::compare<greater>> MinHeap;
//     typedef typename MinHeap::handle_type handle_type2;
//     typedef typename MinHeap::iterator iterator2;  // Iterator for others heap
//     typedef typename MinHeap::ordered_iterator ordered_iterator2;  // Ordered iterator for others heap 
//   private:
//     typedef std::pair<handle_type, FloatType> PairType2;

//     int k;
//     int buffer_size;
//     MaxHeap kLowest; // Max heap for k lowest elements, largest on top
//     std::unordered_map<T, handle_type> kLowestMap;
//     MinHeap others; // Min heap for other elements, smallest on top       
//     std::unordered_map<T, handle_type2> othersMap;     
//     MaxHeap mirror;          
//     std::unordered_map<T, handle_type> mirrorMap;    
//     // std::unordered_map<T, FloatType> inactiveMap;

//     PairType replace0(
//         PairType newPair,
//         MaxHeap& heap,
//         std::unordered_map<int, handle_type>& map) {
//             PairType pair = heap.top();
//             handle_type ha = map[pair.first];
//             heap.update(ha, newPair);
//             map.erase(pair.first);
//             map[newPair.first] = ha;
//             return pair;
//     } 
//     PairType replace0(
//         PairType newPair,
//         MinHeap& heap,
//         std::unordered_map<int, handle_type2>& map) {
//             PairType pair = heap.top();
//             handle_type2 ha = map[pair.first];
//             heap.update(ha, newPair);
//             map.erase(pair.first);
//             map[newPair.first] = ha;
//             return pair;
//     } 
//     void update0(
//             int element,
//             PairType pair,
//             MaxHeap& heap,
//             std::unordered_map<int, handle_type>& map) {
//         handle_type ha = map[element];
//         heap.update(ha, pair);
//         map.erase(element);
//         map[pair.first] = ha;
//     }
//     void update0(
//             int element,
//             PairType pair,
//             MinHeap& heap,
//             std::unordered_map<int, handle_type2>& map) {
//         handle_type2 ha = map[element];
//         heap.update(ha, pair);
//         map.erase(element);
//         map[pair.first] = ha;
//     }
//     void erase0(
//             int element,
//             MaxHeap& heap,
//             std::unordered_map<int, handle_type>& map) {
//         handle_type ha = map[element];    
//         heap.erase(ha);       
//         map.erase(element);
//     }
//     void erase0(
//             int element,
//             MinHeap& heap,
//             std::unordered_map<int, handle_type2>& map) {
//         handle_type2 ha = map[element];    
//         heap.erase(ha);       
//         map.erase(element);
//     }
//   public:
//     // Constructor to initialize with a specific value of k and buffer_size
//     KLowestBufferHeap(int k, int buffer_size) : k(k), buffer_size(buffer_size) {} 

//     // Insert a new element with its distance
//     bool insert(
//             const T& element, 
//             const FloatType& distance) {
//         PairType newPair(element, distance);
//         if (kLowest.size() < k) {
//             kLowestMap[element] = kLowest.push(newPair);
//             return true;
//         } else {   
//             if (kLowest.value_comp()(newPair, kLowest.top())) {     
//                 PairType pair = replace0(newPair, kLowest, kLowestMap);
//                 if (others.size() < buffer_size) {
//                     othersMap[pair.first] = others.push(pair);   
//                     mirrorMap[pair.first] = mirror.push(pair); 
//                     return true;            
//                 } else if (mirror.value_comp()(pair, mirror.top()) ) {
//                     PairType oldPair = replace0(pair, mirror, mirrorMap);
//                     update0(oldPair.first, pair, others, othersMap);
//                     return true;
//                 }                
//             } else {   
//                 if (others.size() < buffer_size) {                       
//                     othersMap[element] = others.push(newPair);
//                     mirrorMap[element] = mirror.push(newPair);
//                     return true;
//                 } else if (mirror.value_comp()(newPair, mirror.top()) ) {
//                     PairType oldPair = replace0(newPair, mirror, mirrorMap);
//                     update0(oldPair.first, newPair, others, othersMap);
//                     return true;
//                 }
//             }
//         }
//         return false;
//     }
//     bool insert(const PairType& newPair) { return insert(newPair.first, newPair.second); }
    
//     bool erase(
//             const T& element) {
//         if (kLowestMap.count(element)>0) {            
//             if (!others.empty()) {
//                 // pop smallest from others
//                 PairType pair = others.top();                
//                 erase0(pair.first, others, othersMap);
//                 erase0(pair.first, mirror, mirrorMap);
//                 // update kLowest
//                 update0(element, pair, kLowest, kLowestMap);
//             } else {
//                 erase0(element, kLowest, kLowestMap);                
//             }  
//             return true;           
//         } else if (othersMap.count(element)>0) {
//             erase0(element, others, othersMap);  
//             erase0(element, mirror, mirrorMap);  
//             return true;      
//         }  
//         return false;      
//     }

//     void update(
//             const T& elementToUpdate,
//             const T& element, 
//             const FloatType& distance) {
//         PairType newPair(element, distance);
//         if (kLowestMap.count(elementToUpdate)>0) {
//             if ( kLowest.value_comp()(newPair, kLowest.top()) || 
//                 (others.value_comp()(others.top(), newPair) && kLowest.top().first == elementToUpdate) ) {
//                 update0(elementToUpdate, newPair, kLowest, kLowestMap);
//             } else {
//                 erase(elementToUpdate);
//                 insert(element, distance);
//             }
//         } else {
//             if ( others.value_comp()(newPair, others.top()) ||
//                 (kLowest.value_comp()(kLowest.top(), newPair) && others.top().first == elementToUpdate) ) {
//                 update0(elementToUpdate, newPair, others, othersMap);
//                 update0(elementToUpdate, newPair, mirror, mirrorMap);
//             } else {
//                 erase(elementToUpdate);
//                 insert(element, distance);
//             }
//         }
//     }

//     void setK(
//             int k_new) {  
//         if (k_new < k) {
//             buffer_size += k - k_new; // increase buffer_size
//         } 
//         if (k < k_new && (k_new-k) > buffer_size) {
//             buffer_size = 0;
//         }           
//         k = k_new;
//         while (kLowest.size() > k) {
//             PairType pair = kLowest.top();
//             erase0(pair.first, kLowest, kLowestMap);              
//             othersMap[pair.first] = others.push(pair); 
//             mirrorMap[pair.first] = mirror.push(pair); 
//         }
//         while (kLowest.size() < k && !others.empty()) {
//             PairType pair = others.top();
//             erase0(pair.first, others, othersMap); 
//             erase0(pair.first, mirror, mirrorMap);            
//             kLowestMap[pair.first] = kLowest.push(pair);
//         }
//     }
//     void setBufferSize(int bs_new) { buffer_size = bs_new; }

//     int getK() {
//         return k;
//     }

//     FloatType top() {
//         return kLowest.top().second;
//     }

//     FloatType bottom() {
//         return mirror.top().second;
//     }

//     bool isin(int element) {
//         return (kLowestMap.count(element)>0) || (othersMap.count(element)>0);
//     }

//     bool empty() {
//         return kLowest.empty();
//     }
    
//     std::size_t size() {
//         return kLowest.size();
//     }

//     std::size_t sizeB() {
//         return others.size();
//     }

//     std::size_t sizeA() {
//         return kLowest.size() + others.size();
//     }


//     FloatType median() {
//         ordered_iterator it = kLowest.ordered_begin();
//         int steps = k/2 - (k - kLowest.size());
//         if (steps > 0) {
//             if (k % 2 == 0) {
//                 FloatType m(0);
//                 int steps = (k-1)/2 - (k - kLowest.size());
//                 std::advance(it, steps);
//                 m += it->second;
//                 it++;
//                 m += it->second;
//                 return 0.5 * m;
//             } else {
//                 std::advance(it, steps);
//                 return it->second;            
//             }
//         } else {
//             if (!kLowest.empty()) { return it->second; } // highest value
//             else { return 0.0; }
//         }
//     }

//     // Get iterators for kLowest heap
//     iterator begin() const { return kLowest.begin(); }    
//     iterator end() const { return kLowest.end(); }
//     ordered_iterator ordered_begin() const { return kLowest.ordered_begin(); }
//     ordered_iterator ordered_end() const { return kLowest.ordered_end(); }

//     // Get iterators for others heap
//     iterator2 begin2() const { return others.begin(); }    
//     iterator2 end2() const { return others.end(); }
//     ordered_iterator2 ordered_begin2() const { return others.ordered_begin(); }
//     ordered_iterator2 ordered_end2() const { return others.ordered_end(); }

//     // Get iterators for mirrors heap
//     ordered_iterator ordered_rbegin2() const { return mirror.ordered_begin(); }
//     ordered_iterator ordered_rend2() const { return mirror.ordered_end(); }
// };

#endif
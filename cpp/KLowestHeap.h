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

// Define a template class for KHeap
template<typename ValueType, typename T>
class KHeap {
  public:  
    typedef std::pair<T, ValueType> PairType;
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
    typedef typename MaxHeap::iterator iterator;  // Iterator for kHeap heap
    typedef typename MaxHeap::ordered_iterator ordered_iterator;  // Ordered iterator for kHeap heap    
    typedef boost::heap::binomial_heap<PairType, boost::heap::compare<greater>> MinHeap;
    typedef typename MinHeap::handle_type rhandle_type;
    typedef typename MinHeap::iterator riterator;  // Iterator for rBuffer heap
    typedef typename MinHeap::ordered_iterator ordered_riterator;  // Ordered iterator for rBuffer heap   
  private:
    std::size_t k;
    MaxHeap kHeap; // Max heap for k lowest elements, largest on top
    std::unordered_map<T, handle_type> kHeapMap;
    MinHeap rBuffer; // Min heap for other elements, smallest on top    
    std::unordered_map<T, rhandle_type> rBufferMap;

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
            std::unordered_map<int, rhandle_type>& map) {
        PairType pair = heap.top();
        rhandle_type ha = map[pair.first];
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
            std::unordered_map<int, rhandle_type>& map) {
        rhandle_type ha = map[element];
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
            std::unordered_map<int, rhandle_type>& map) {
        rhandle_type ha = map[element];    
        heap.erase(ha);       
        map.erase(element);
    }  
  public:
    // Constructor to initialize with a specific value of k
    KHeap() : k(0) {}
    KHeap(std::size_t k) : k(k) {} 

    // Insert a new element with its distance
    void insert(
            const T& element, 
            const ValueType& distance) {
        PairType newPair(element, distance);
        if (kHeap.size() < k) {
            kHeapMap[element] = kHeap.push(newPair);
        } else {  
            if (kHeap.value_comp()(newPair, kHeap.top())) { 
                PairType pair = replace0(newPair, kHeap, kHeapMap); 
                rBufferMap[pair.first] = rBuffer.push(pair);                
            } else {                
                rBufferMap[element] = rBuffer.push(newPair);
            }
        }
    }
    void insert(const PairType& newPair) { insert(newPair.first, newPair.second); }
    
    void erase(
            const T& element) {
        if (kHeapMap.count(element)>0) {
            if (!rBuffer.empty()) {
                // pop smallest from rBuffer
                PairType pair = rBuffer.top();                
                erase0(pair.first, rBuffer, rBufferMap);
                update0(element, pair, kHeap, kHeapMap);
            } else {
                erase0(element, kHeap, kHeapMap);                
            }       
        } else { // must be in rBufferMap            
            erase0(element, rBuffer, rBufferMap);             
        }        
    }

    void update(
            const T& elementToUpdate,
            const T& element, 
            const ValueType& distance) {
        PairType newPair(element, distance);
        if (kHeapMap.count(elementToUpdate)>0) {
            if ( kHeap.value_comp()(newPair, kHeap.top()) || 
                (rBuffer.value_comp()(rBuffer.top(), newPair) && kHeap.top().first == elementToUpdate) ) {
                update0(elementToUpdate, newPair, kHeap, kHeapMap);
            } else {
                erase(elementToUpdate);
                insert(element, distance);
            }
        } else {
            if ( rBuffer.value_comp()(newPair, rBuffer.top()) ||
                (kHeap.value_comp()(kHeap.top(), newPair) && rBuffer.top().first == elementToUpdate) ) {
                update0(elementToUpdate, newPair, rBuffer, rBufferMap);
            } else {
                erase(elementToUpdate);
                insert(element, distance);
            }
        }
    }

    void setK(
            std::size_t k_new) {
        k = k_new;
        while (kHeap.size() > k) {
            PairType pair = kHeap.top();
            erase0(pair.first, kHeap, kHeapMap);              
            rBufferMap[pair.first] = rBuffer.push(pair);
        }
        while (kHeap.size() < k && !rBuffer.empty()) {
            PairType pair = rBuffer.top();
            erase0(pair.first, rBuffer, rBufferMap);            
            kHeapMap[pair.first] = kHeap.push(pair);
        }
    }

    int getK() {
        return k;
    }

    ValueType top() {
        return kHeap.top().second;
    }

    // Get iterators for kHeap heap
    iterator begin() const { return kHeap.begin(); }    
    iterator end() const { return kHeap.end(); }
    ordered_iterator ordered_begin() const { return kHeap.ordered_begin(); }
    ordered_iterator ordered_end() const { return kHeap.ordered_end(); }

    // Get iterators for rBuffer heap
    riterator begin2() const { return rBuffer.begin(); }    
    riterator end2() const { return rBuffer.end(); }
    ordered_riterator ordered_begin2() const { return rBuffer.ordered_begin(); }
    ordered_riterator ordered_end2() const { return rBuffer.ordered_end(); }
};


template<typename ValueType, typename T>
class KBufferHeap {
  public:  
    typedef std::pair<ValueType, T> PairType; // val, key
    typedef boost::heap::binomial_heap<PairType> MaxHeap;    
    typedef typename MaxHeap::handle_type handle_type;
    typedef typename MaxHeap::iterator iterator;  // Iterator for kHeap heap
    typedef typename MaxHeap::ordered_iterator ordered_iterator;  // Ordered iterator for kHeap heap    
    typedef boost::heap::binomial_heap<PairType, boost::heap::compare<std::greater>> MinHeap;
    typedef typename MinHeap::handle_type rhandle_type;
    typedef typename MinHeap::iterator riterator;  // Iterator for rBuffer heap
    typedef typename MinHeap::ordered_iterator ordered_riterator;  // Ordered iterator for rBuffer heap 
  private:
    typedef std::pair<handle_type, rhandle_type> HandlePairType;

    std::size_t k;
    std::size_t max_buffer_size;

    MaxHeap kHeap; // Max heap for k lowest elements, largest on top
    std::unordered_map<T, handle_type> kHeapMap;

    MaxHeap buffer;          
    MinHeap rBuffer; // Min heap for other elements, smallest on top 
    std::unordered_map<T, HandlePairType> bufferMap;  

    std::unordered_map<T, ValueType> inactiveMap;

    bool in_h(const T& key) const { return kHeapMap.count(key)>0; }
    bool in_b(const T& key) const { return bufferMap.count(key)>0; }
    bool in_ia(const T& key) const { return inactiveMap.count(key)>0; }

    bool erase_h(const T& key) {
        if (in_h(key)) {
            handle_type ha = kHeapMap[key];
            kHeap.erase(ha)
            kHeapMap.erase(key);
            return true;
        }
        return false;
    }
    bool erase_b(const T& key) {
        if (in_b(key)) {
            HandlePairType hpair = bufferMap[key];
            buffer.erase(hpair.first);
            rBuffer.erase(hpair.second);
            bufferMap.erase(key);
            return true;
        }
        return false;
    }
    bool erase_ia (const T& key) {
        if (in_ia(key)) {
            inactiveMap.erase(key);
            return true;
        }
        return false;
    }

    bool insert_h(const T key, const ValueType val) {
        if ( kHeap.size()<k && ((kHeap.value_comp()(PairType(val, key), kHeap.top())) || kHeap.empty()) ) {
            handle_type ha = kHeap.insert(PairType(val, key));
            kHeapMap[key] = ha;
            return true;
        }
        return false;
    }
    bool insert_h(const PairType pair) { return insert_h(pair.second, pair.first); }
    bool insert_b(const T key, const ValueType val) {
        if ( buffer.size() < max_buffer_size && ((buffer.value_comp()(PairType(val, key), buffer.top())) || buffer.empty()) ) {
            handle_type = buffer.insert(PairType(val, key));
            rhandle_type = rBuffer.insert(PairType(val, key));
            bufferMap[key] = HandlePairType(handle_type, rhandle_type);
            return true;
        }
        return false;
    }
    bool insert_b(const PairType pair) { return insert_b(pair.second, pair.first); }
    bool insert_ia(const T key, const ValueType val) {
        inactiveMap[key] = val;
        return true;
    }
    bool insert_ia(const PairType pair) { return insert_ia(pair.second, pair.first); }

    bool update_h(const T& oldKey, const T key, const ValueType val) {
        if ( in_h(oldKey) && ((kHeap.value_comp()(PairType(val, key), rBuffer.top())) || buffer.empty()) ) {
            handleType ha = kHeapMap(oldKey);
            kHeap.update(ha, PairType(val, key));
            kHeapMap.erase(oldKey);
            kHeapMap[key] = ha;
            return true;
        } else if (in_h(oldKey)) { // shift top pair from buffer to heap in oldKey s place
            PairType topPair = rBuffer.top();            
            if (!update_b(pair.second, key, val)) { throw std::runtime_error("Should be impossible to happen: " + std::to_string(oldKey)); } // key, val must be larger than top pair 
            if (!update_h(oldKey, topPair)) { throw std::runtime_error("Should be impossible to happen: " + std::to_string(oldKey));}
            return true;
        }
        return false;
    }
    bool update_h(const T& oldKey, const PairType pair) { return update_h(oldKey, pair.second, pair.first); }
    bool update_b(const T& oldKey, const T key, const ValueType val) {
        if ( in_b(oldKey) && ((rBuffer.value_comp()(PairType(val, key), kHeap.top())) || kHeap.empty()) ) {
            HandlePairType hpair = bufferMap(oldKey);
            buffer.update(hpair.first, PairType(val, key));
            rBuffer.update(hpair.second, PairType(val, key));
            bufferMap.erase(oldKey);
            bufferMap[key] = hpair;
            return true;
        } else if (in_b(oldKey)) { // shift top pair from heap to buffer in oldKey s place
            PairType topPair = kHeap.top();            
            if (!update_h(pair.second, key, val)) { throw std::runtime_error("Should be impossible to happen: " + std::to_string(oldKey)); } // key, val must be larger than top pair 
            if (!update_n(oldKey, topPair)) { throw std::runtime_error("Should be impossible to happen: " + std::to_string(oldKey));}
            return true;
        }

        return false;
    }
    bool update_b(const T& oldKey, const PairType pair) { return update_b(oldKey, pair.second, pair.first); }

    bool update_and_shift_h(const T& oldKey, const T key, const ValueType val) {
        if ( in_h(oldKey) && buffer.size()>0) {
            
        }
        return false;
    }
    
    bool shift_or_erase_h(const T& oldKey) { // shift top buffer to kHeap
        if (in_h(oldKey)) {
            if (buffer.size()>0) {
                PairType topPair = rBuffer.top();
                erase_b(topPair.second);
                update_h(oldKey, topPair);
            } 
            else {
                erase_h(oldKey);
            }
            return true;
        } 
        return false;       
    }

    bool shift_to_h() { // shift top buffer to kHeap
        if (buffer.size()>0 && kHeap.size()<k) {
            PairType topPair = rBuffer.top();
            erase_b(topPair.second);
            insert_h(topPair);            
            return true;
        } 
        return false;       
    }
    bool shift_to_b() { // shift top buffer to kHeap
        if (kHeap.size()>k) {
            PairType topPair = kHeap.top();
            erase_h(topPair.second);
            insert_b(topPair);            
            return true;
        } 
        return false;       
    }

  public:
    // Constructor to initialize with a specific value of k and max_buffer_size
    KBufferHeap() : k(0), max_buffer_size(0) {}
    KBufferHeap(int k, int max_buffer_size) : k(k), max_buffer_size(max_buffer_size) {} 

    // Insert a new element with its distance
    bool insert(const T key, const ValueType val, bool inactive = false) {
        if (inactive) { return insert_ia(key, val); }
        if (insert_b(key, val)) { return true; } 
        if (insert_h(key, val)) { return true; }
        return false;
    }
    bool insert(const PairType pair, bool inactive = false) { return insert(pair.second, pair.first, inactive); }
    
    bool erase(const T& key) {
        if (erase_ia(key)) { return true; } 
        if (erase_b(key)) { return true; } 
        if (erase_h(key)) { return true; } 
        return false;      
    }

    void update(
            const T& oldKey,
            const T key, 
            const ValueType val) {
        if (update_ia(oldKey, key, val)) { return true; }
        if (update_b(oldKey, key, val)) { return true; }
        if (update_h(oldKey, key, val)) { return true; }
        return false;       
    }

    bool active(
            const T& element) {
        return !(inactiveMap.count(element)>0) && ((kHeapMap.count(element)>0) || (rBufferMap.count(element)>0));
    }

    bool inactive(
            const T& element) {
        return (inactiveMap.count(element)>0);
    }

    void activate(
            const T& element) {
        ValueType distance = inactiveMap[element];
        inactiveMap.erase(element);
        insert(element, distance);
    }

    void deactivate(
            const T& element) {
        if (bufferMap.count(element)>0) {
            inactiveMap[element] = bufferMap[element].second;
        } else {
            inactiveMap[element] = kHeapMap[element].second;
        }
        erase(element);
    }

    void setK(
            int k_new) {  
        if (k_new < k) {
            max_buffer_size += k - k_new; // increase max_buffer_size
        } 
        if (k < k_new && (k_new-k) > max_buffer_size) {
            max_buffer_size = 0;
        }           
        k = k_new;        
        
    }
    void setMaxBufferSize(int bs_new) { max_buffer_size = bs_new; }

    std::size_t getMaxBufferSize() { return max_buffer_size; }
    std::size_t getK() { return k; }

    ValueType top() { return kHeap.top().first; }
    ValueType bottom() { return buffer.top().first; }

    bool isin(int element) { return in_h() || in_b() || in_ia(); }
    bool empty() { return kHeap.empty(); }    
    std::size_t size() { return kHeap.size(); }
    std::size_t size_buffer() { return rBuffer.size(); }
    std::size_t size_all() { return kHeap.size() + rBuffer.size(); }
    ValueType median() {
        ordered_iterator it = kHeap.ordered_begin();
        int steps = k/2 - (k - kHeap.size());
        if (steps > 0) {
            if (k % 2 == 0) {
                ValueType m(0);
                int steps = (k-1)/2 - (k - kHeap.size());
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
            if (!kHeap.empty()) { return it->second; } // highest value
            else { return 0.0; }
        }
    }

    void print() {
        std::cout << "K lowest: " << std::endl;
        for (auto it = kHeap.begin(); it != kHeap.end(); ++it) {
            std::cout << "(" << it->first << ": " << it->second << ") " ;
        }
        std::cout << std::endl << "Buffer: " << std::endl;
        for (auto it = buffer.begin(); it != buffer.end(); ++it) {
            std::cout << "(" << it->first << ": " << it->second << ") ";
        }
        std::cout << std::endl << "Buffer rBuffer: " << std::endl;
        for (auto it = rBuffer.begin(); it != rBuffer.end(); ++it) {
            std::cout << "(" << it->first << ": " << it->second << ") ";
        }
        std::cout << std::endl << "Inactive: " << std::endl;
        for (auto pair: inactiveMap) {
            std::cout << "(" << pair.first << ": " << pair.second << ") ";
        }
        std::cout << std::endl;
    }

    // Get iterators for kHeap heap
    iterator begin() const { return kHeap.begin(); }    
    iterator end() const { return kHeap.end(); }
    ordered_iterator ordered_begin() const { return kHeap.ordered_begin(); }
    ordered_iterator ordered_end() const { return kHeap.ordered_end(); }

    // Get iterators for rBuffer heap
    riterator begin2() const { return rBuffer.begin(); }    
    riterator end2() const { return rBuffer.end(); }
    ordered_riterator ordered_begin2() const { return rBuffer.ordered_begin(); }
    ordered_riterator ordered_end2() const { return rBuffer.ordered_end(); }

    // Get iterators for buffers heap
    ordered_iterator ordered_rbegin2() const { return buffer.ordered_begin(); }
    ordered_iterator ordered_rend2() const { return buffer.ordered_end(); }
};



// template<typename T, typename ValueType>
// class KBufferHeap {
//   public:  
//     typedef std::pair<T, ValueType> PairType;
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
//     typedef typename MaxHeap::iterator iterator;  // Iterator for kHeap heap
//     typedef typename MaxHeap::ordered_iterator ordered_iterator;  // Ordered iterator for kHeap heap    
//     typedef boost::heap::binomial_heap<PairType, boost::heap::compare<greater>> MinHeap;
//     typedef typename MinHeap::handle_type rhandle_type;
//     typedef typename MinHeap::iterator riterator;  // Iterator for rBuffer heap
//     typedef typename MinHeap::ordered_iterator ordered_riterator;  // Ordered iterator for rBuffer heap 
//   private:
//     typedef std::pair<handle_type, ValueType> rhandle_type;

//     int k;
//     int max_buffer_size;
//     MaxHeap kHeap; // Max heap for k lowest elements, largest on top
//     std::unordered_map<T, handle_type> kHeapMap;
//     MinHeap rBuffer; // Min heap for other elements, smallest on top       
//     std::unordered_map<T, rhandle_type> rBufferMap;     
//     MaxHeap buffer;          
//     std::unordered_map<T, handle_type> bufferMap;    
//     // std::unordered_map<T, ValueType> inactiveMap;

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
//         std::unordered_map<int, rhandle_type>& map) {
//             PairType pair = heap.top();
//             rhandle_type ha = map[pair.first];
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
//             std::unordered_map<int, rhandle_type>& map) {
//         rhandle_type ha = map[element];
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
//             std::unordered_map<int, rhandle_type>& map) {
//         rhandle_type ha = map[element];    
//         heap.erase(ha);       
//         map.erase(element);
//     }
//   public:
//     // Constructor to initialize with a specific value of k and max_buffer_size
//     KBufferHeap(int k, int max_buffer_size) : k(k), max_buffer_size(max_buffer_size) {} 

//     // Insert a new element with its distance
//     bool insert(
//             const T& element, 
//             const ValueType& distance) {
//         PairType newPair(element, distance);
//         if (kHeap.size() < k) {
//             kHeapMap[element] = kHeap.push(newPair);
//             return true;
//         } else {   
//             if (kHeap.value_comp()(newPair, kHeap.top())) {     
//                 PairType pair = replace0(newPair, kHeap, kHeapMap);
//                 if (rBuffer.size() < max_buffer_size) {
//                     rBufferMap[pair.first] = rBuffer.push(pair);   
//                     bufferMap[pair.first] = buffer.push(pair); 
//                     return true;            
//                 } else if (buffer.value_comp()(pair, buffer.top()) ) {
//                     PairType oldPair = replace0(pair, buffer, bufferMap);
//                     update0(oldPair.first, pair, rBuffer, rBufferMap);
//                     return true;
//                 }                
//             } else {   
//                 if (rBuffer.size() < max_buffer_size) {                       
//                     rBufferMap[element] = rBuffer.push(newPair);
//                     bufferMap[element] = buffer.push(newPair);
//                     return true;
//                 } else if (buffer.value_comp()(newPair, buffer.top()) ) {
//                     PairType oldPair = replace0(newPair, buffer, bufferMap);
//                     update0(oldPair.first, newPair, rBuffer, rBufferMap);
//                     return true;
//                 }
//             }
//         }
//         return false;
//     }
//     bool insert(const PairType& newPair) { return insert(newPair.first, newPair.second); }
    
//     bool erase(
//             const T& element) {
//         if (kHeapMap.count(element)>0) {            
//             if (!rBuffer.empty()) {
//                 // pop smallest from rBuffer
//                 PairType pair = rBuffer.top();                
//                 erase0(pair.first, rBuffer, rBufferMap);
//                 erase0(pair.first, buffer, bufferMap);
//                 // update kHeap
//                 update0(element, pair, kHeap, kHeapMap);
//             } else {
//                 erase0(element, kHeap, kHeapMap);                
//             }  
//             return true;           
//         } else if (rBufferMap.count(element)>0) {
//             erase0(element, rBuffer, rBufferMap);  
//             erase0(element, buffer, bufferMap);  
//             return true;      
//         }  
//         return false;      
//     }

//     void update(
//             const T& elementToUpdate,
//             const T& element, 
//             const ValueType& distance) {
//         PairType newPair(element, distance);
//         if (kHeapMap.count(elementToUpdate)>0) {
//             if ( kHeap.value_comp()(newPair, kHeap.top()) || 
//                 (rBuffer.value_comp()(rBuffer.top(), newPair) && kHeap.top().first == elementToUpdate) ) {
//                 update0(elementToUpdate, newPair, kHeap, kHeapMap);
//             } else {
//                 erase(elementToUpdate);
//                 insert(element, distance);
//             }
//         } else {
//             if ( rBuffer.value_comp()(newPair, rBuffer.top()) ||
//                 (kHeap.value_comp()(kHeap.top(), newPair) && rBuffer.top().first == elementToUpdate) ) {
//                 update0(elementToUpdate, newPair, rBuffer, rBufferMap);
//                 update0(elementToUpdate, newPair, buffer, bufferMap);
//             } else {
//                 erase(elementToUpdate);
//                 insert(element, distance);
//             }
//         }
//     }

//     void setK(
//             int k_new) {  
//         if (k_new < k) {
//             max_buffer_size += k - k_new; // increase max_buffer_size
//         } 
//         if (k < k_new && (k_new-k) > max_buffer_size) {
//             max_buffer_size = 0;
//         }           
//         k = k_new;
//         while (kHeap.size() > k) {
//             PairType pair = kHeap.top();
//             erase0(pair.first, kHeap, kHeapMap);              
//             rBufferMap[pair.first] = rBuffer.push(pair); 
//             bufferMap[pair.first] = buffer.push(pair); 
//         }
//         while (kHeap.size() < k && !rBuffer.empty()) {
//             PairType pair = rBuffer.top();
//             erase0(pair.first, rBuffer, rBufferMap); 
//             erase0(pair.first, buffer, bufferMap);            
//             kHeapMap[pair.first] = kHeap.push(pair);
//         }
//     }
//     void setBufferSize(int bs_new) { max_buffer_size = bs_new; }

//     int getK() {
//         return k;
//     }

//     ValueType top() {
//         return kHeap.top().second;
//     }

//     ValueType bottom() {
//         return buffer.top().second;
//     }

//     bool isin(int element) {
//         return (kHeapMap.count(element)>0) || (rBufferMap.count(element)>0);
//     }

//     bool empty() {
//         return kHeap.empty();
//     }
    
//     std::size_t size() {
//         return kHeap.size();
//     }

//     std::size_t sizeB() {
//         return rBuffer.size();
//     }

//     std::size_t sizeA() {
//         return kHeap.size() + rBuffer.size();
//     }


//     ValueType median() {
//         ordered_iterator it = kHeap.ordered_begin();
//         int steps = k/2 - (k - kHeap.size());
//         if (steps > 0) {
//             if (k % 2 == 0) {
//                 ValueType m(0);
//                 int steps = (k-1)/2 - (k - kHeap.size());
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
//             if (!kHeap.empty()) { return it->second; } // highest value
//             else { return 0.0; }
//         }
//     }

//     // Get iterators for kHeap heap
//     iterator begin() const { return kHeap.begin(); }    
//     iterator end() const { return kHeap.end(); }
//     ordered_iterator ordered_begin() const { return kHeap.ordered_begin(); }
//     ordered_iterator ordered_end() const { return kHeap.ordered_end(); }

//     // Get iterators for rBuffer heap
//     riterator begin2() const { return rBuffer.begin(); }    
//     riterator end2() const { return rBuffer.end(); }
//     ordered_riterator ordered_begin2() const { return rBuffer.ordered_begin(); }
//     ordered_riterator ordered_end2() const { return rBuffer.ordered_end(); }

//     // Get iterators for buffers heap
//     ordered_iterator ordered_rbegin2() const { return buffer.ordered_begin(); }
//     ordered_iterator ordered_rend2() const { return buffer.ordered_end(); }
// };

#endif
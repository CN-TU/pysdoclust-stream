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

template<typename ValueType, typename T>
class KBufferHeap {
  public:  
    typedef std::pair<ValueType, T> PairType; // val, key
    typedef boost::heap::binomial_heap<PairType> MaxHeap;    
    typedef typename MaxHeap::handle_type handle_type;
    typedef typename MaxHeap::iterator iterator;  // Iterator for kHeap heap
    typedef typename MaxHeap::ordered_iterator ordered_iterator;  // Ordered iterator for kHeap heap    
    typedef boost::heap::binomial_heap< PairType, boost::heap::compare<std::greater<PairType>> > MinHeap;
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

    // MaxHeap inactive; // Max heap for k lowest elements, largest on top
    std::unordered_map<T, ValueType> inactiveMap;

    bool in_h(const T& key) const { return kHeapMap.count(key)>0; }
    bool in_b(const T& key) const { return bufferMap.count(key)>0; }
    bool in_ia(const T& key) const { return inactiveMap.count(key)>0; }
    bool full_h() const { return kHeap.size()==k; }    
    bool empty_h() const { return kHeap.empty(); }
    bool full_b() const { return buffer.size()==max_buffer_size; }
    bool empty_b() const { return buffer.empty() && rBuffer.empty(); }

    bool check_b(const T key, const ValueType val, bool insert = true) const {
        if ( !(max_buffer_size>0) ) {return false; }
        if ( in(key) ) { return false; }
        if (insert) { return !check_h(key, val) && 
            (!full_b() || buffer.value_comp()(PairType(val, key), buffer.top())); }
        return !empty_b() && check_b(key, val); 
    }
    bool check_h(const T key, const ValueType val, bool insert = true) const {
        if ( !(k>0) ) { return false; }
        if ( in(key) ) { return false; }
        if (insert) { return !full_h() || kHeap.value_comp()(PairType(val, key), kHeap.top()); }
        return !empty_h() && (empty_b() || kHeap.value_comp()(PairType(val, key), rBuffer.top()));
    }

    bool erase_h(const T key) {
        if (in_h(key)) {
            if (empty_b()) {
                handle_type ha = kHeapMap[key];
                kHeap.erase(ha);
                kHeapMap.erase(key);
            } else {
                PairType topPair = rBuffer.top();
                erase_b(topPair.second);
                update_h(key, topPair);
            }            
            return true;
        }
        return false;
    }
    bool erase_b(const T key) {
        if (in_b(key)) {
            HandlePairType hpair = bufferMap[key];
            buffer.erase(hpair.first);
            rBuffer.erase(hpair.second);
            bufferMap.erase(key);
            return true;
        }
        return false;
    }
    bool erase_ia (const T key) {
        if (in_ia(key)) {
            inactiveMap.erase(key);
            return true;
        }
        return false;
    }

    bool insert_h(const T key, const ValueType val) {
        if (check_h(key, val)) {
            if (!full_h()) {
                handle_type ha = kHeap.push(PairType(val, key));
                kHeapMap[key] = ha;
                return true;
            } else {
                PairType topPair = kHeap.top();
                if (update_h(topPair.second, key, val)) {
                    insert_b(topPair); // if buffer is empty this is false but element is inserted
                }  
            }      
            return true;
        }        
        return false;
    }
    bool insert_h(const PairType pair) { return insert_h(pair.second, pair.first); }
    bool insert_b(const T key, const ValueType val) {
        if ( check_b(key, val)) {             
            if (full_b()) {
                return update_b(buffer.top().second, key, val); // must be true              
            } else {
                handle_type ha = buffer.push(PairType(val, key));
                rhandle_type rha = rBuffer.push(PairType(val, key));
                bufferMap[key] = HandlePairType(ha, rha);
                return true;
            }
        }        
        return false;
    }
    bool insert_b(const PairType pair) { return insert_b(pair.second, pair.first); }
    bool insert_ia(const T key, const ValueType val) {
        inactiveMap[key] = val;
        return true;
    }
    bool insert_ia(const PairType pair) { return insert_ia(pair.second, pair.first); }

    bool update_h(const T oldKey, const T key, const ValueType val) {
        if ( in_h(oldKey) && check_h(key, val, false) ) {
            handle_type ha = kHeapMap[oldKey];
            kHeap.update(ha, PairType(val, key));
            kHeapMap.erase(oldKey);
            kHeapMap[key] = ha;
            return true;
        } 
        if ( in_h(oldKey) && check_b(key, val, false) ) { // shift top pair from buffer to heap in oldKey s place
            PairType topPair = rBuffer.top();            
            update_b(topPair.second, key, val);
            update_h(oldKey, topPair);
            return true;
        }
        return false;
    }
    bool update_h(const T oldKey, const PairType pair) { return update_h(oldKey, pair.second, pair.first); }
    bool update_b(const T oldKey, const T key, const ValueType val) {
        if ( in_b(oldKey) && check_b(key, val, false) ) {
            HandlePairType hpair = bufferMap[oldKey];
            buffer.update(hpair.first, PairType(val, key));
            rBuffer.update(hpair.second, PairType(val, key));
            bufferMap.erase(oldKey);
            bufferMap[key] = hpair;
            return true;
        }
        if ( in_b(oldKey) && check_h(key, val, false) ) { // shift top pair from heap to buffer in oldKey s place
            PairType topPair = kHeap.top();            
            update_h(topPair.second, key, val);
            update_b(oldKey, topPair);
        }
        return false;
    }
    bool update_b(const T oldKey, const PairType pair) { return update_b(oldKey, pair.second, pair.first); }

    void shift_to_left() {
        ++k;
        --max_buffer_size;
        if (!empty_b()) {
            PairType topPair = rBuffer.top();
            erase_b(topPair.second);
            insert_h(topPair);
        }        
    }
    void shift_to_right() {
        --k;
        ++max_buffer_size;
        if (!empty_h()) {
            PairType topPair = kHeap.top();
            erase_h(topPair.second);
            insert_b(topPair);
        }        
    }

   
  public:
    // Constructor to initialize with a specific value of k and max_buffer_size
    KBufferHeap() : k(0), max_buffer_size(0) {}
    KBufferHeap(std::size_t k, std::size_t max_buffer_size) : k(k), max_buffer_size(max_buffer_size) {} 

    // Insert a new element with its distance
    bool insert(const T key, const ValueType val, bool inactive = false) {
        if (inactive) { return insert_ia(key, val); }
        if (insert_h(key, val)) { return true; }
        if (insert_b(key, val)) { return true; }         
        return false;
    }
    bool insert(const PairType pair, bool inactive = false) { return insert(pair.second, pair.first, inactive); }
    
    bool erase(const T& key) {
        if (erase_ia(key)) { return true; } 
        if (erase_b(key)) { return true; } 
        if (erase_h(key)) { return true; } 
        return false;      
    }
    
    bool update(const T& oldKey, const T key,  const ValueType val) {
        if (update_b(oldKey, key, val)) { return true; }
        if (update_h(oldKey, key, val)) { return true; }
        return false;       
    }
    bool update(const T& oldKey, const PairType pair) { return update(oldKey, pair.second, pair.first); }

    bool activate(const T& key) {
        if (in_ia(key)) {
            PairType pair(inactiveMap[key], key);
            inactiveMap.erase(key);
            if (insert_b(pair)) { return true; }
            if (insert_h(pair)) { return true; }
        }
        return false;
    }

    bool deactivate(const T& key) {
        if (active(key)) {
            handle_type ha;
            if (in_h(key)) { ha = kHeapMap[key]; }
            if (in_b(key)) { ha = bufferMap[key].first; }      
            PairType pair = *ha;                
            inactiveMap[pair.second] = pair.first;
            if (erase_b(pair.second)) { return true; }
            if (erase_h(pair.second)) { return true; }
        }
        return false;
    }

    bool swap_active(const T& key_a, const T& key_ia) {
        if (active(key_a) && in_ia(key_ia)) {
            PairType pair_ia(inactiveMap[key_ia], key_ia);
            inactiveMap.erase(key_ia);
            handle_type ha;
            if (in_h(key_a)) { ha = kHeapMap[key_a]; }
            if (in_b(key_a)) { ha = bufferMap[key_a].first; }   
            PairType pair_a = *ha;       
            inactiveMap[key_a] = pair_a.first;
            if (update_b(key_a, pair_ia)) { return true; }
            if (update_h(key_a, pair_ia)) { return true; }
        }
        return false;
    }

    bool balanceK(int k_new) {
        if (k_new<0) { return false; }
        if (k_new>max_buffer_size) { return false; }
        while (k != k_new) {
            if (k<k_new) {                
                shift_to_left();                
            } else {
                shift_to_right();                
            }
        }
        return true;
    }

    bool setK(int k_new) {  
        if (k_new<0) { return false; }
        if (k_new>max_buffer_size) { return false; }   
        k = k_new;  
        return true;
    }
    bool setMaxBufferSize(int bs_new) { 
        if (bs_new<0) { return false; }
        max_buffer_size = bs_new; 
        return true;
    }

    std::size_t getMaxBufferSize() const { return max_buffer_size; }
    std::size_t getK() const { return k; }

    const ValueType kVal() const { return kHeap.top().first; }
    const PairType kPair() const { return kHeap.top(); }

    bool active(const T& key) const { return (in_h(key) || in_b(key)); }
    bool inactive(const T& key) const { return in_ia(key); }
    bool in(const T& key) const { return in_h(key) || in_b(key) || in_ia(key); }
    bool empty() const { return kHeap.empty(); }    
    std::size_t size() const { return kHeap.size(); }
    std::size_t size_buffer() const { return rBuffer.size(); }
    std::size_t size_all() const { return kHeap.size() + rBuffer.size(); }
    ValueType median() const {
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
    ValueType topK() const {
        return kHeap.top().first;
    }
    void print() const {
        std::cout << "K lowest: " << std::endl;
        for (auto it = kHeap.ordered_begin(); it != kHeap.ordered_end(); ++it) {
            std::cout << "(" << it->second << ": " << it->first << ") " ;
        }
        std::cout << std::endl << "Buffer: " << std::endl;
        for (auto it = buffer.ordered_begin(); it != buffer.ordered_end(); ++it) {
            std::cout << "(" << it->second << ": " << it->first << ") " ;
        }
        std::cout << std::endl << "rBuffer: " << std::endl;
        for (auto it = rBuffer.ordered_begin(); it != rBuffer.ordered_end(); ++it) {
            std::cout << "(" << it->second << ": " << it->first << ") " ;
        }
        std::cout << std::endl << "Inactive: " << std::endl;
        for (auto pair: inactiveMap) {
            std::cout << "(" << pair.second << ": " << pair.first << ") ";
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

#endif
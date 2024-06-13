#ifndef SDOSTREAMCLUST_BUFFER_H
#define SDOSTREAMCLUST_BUFFER_H

template<typename FloatType>
class SDOstreamclust<FloatType>::DataBuffer {
private:
    std::vector<Vector<FloatType>> data_b;
    std::vector<FloatType> epsilon_b;
    std::vector<FloatType> time_b;
public:
    std::size_t size;

    DataBuffer() : size(0) {}
    
    DataBuffer(
            std::size_t input_buffer
        ) : size(0) {
            if (input_buffer>0) {
                data_b.resize(input_buffer);
                time_b.resize(input_buffer);
                epsilon_b.resize(input_buffer);
            }                
    }

    void add(const std::vector<Vector<FloatType>>& data, const std::vector<FloatType>& epsilon, const std::vector<FloatType>& time) {
        std::copy(data.begin(), data.end(), data_b.begin() + size);
        std::copy(epsilon.begin(), epsilon.end(), epsilon_b.begin() + size);
        std::copy(time.begin(), time.end(),time_b.begin() + size);

        size += data.size();
    }

    void flush(std::vector<Vector<FloatType>>& data, std::vector<FloatType>& epsilon, std::vector<FloatType>& time) { 
        std::copy(data_b.begin(), data_b.begin() + size, data.begin());
        std::copy(epsilon_b.begin(), epsilon_b.begin() + size, epsilon.begin());
        std::copy(time_b.begin(), time_b.begin() + size, time.begin());

        size = 0;
    }

};

#endif  // SDOSTREAMCLUST_BUFFER_H
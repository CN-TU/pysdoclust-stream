#ifndef DSALMON_GAMMA_H
#define DSALMON_GAMMA_H

#include <cmath>
#include <boost/math/distributions/gamma.hpp>

template<typename FloatType>
class Gamma {
private:
    FloatType mn;
    FloatType var;
    FloatType n;

    FloatType shape;    // Shape parameter
    FloatType scale;    // Scale parameter
    
    FloatType last_added; // time wise

public:
    // Constructor
    Gamma()
        : mn(0), var(0), n(0), shape(0), scale(0), last_added(0) {}

    // Function to update mean / sd with a new data point
    void update(FloatType new_data_point, FloatType fading, FloatType now) {
        // Update mean
        n *= std::pow(fading, now - last_added);
        mn = (n * mn + new_data_point) / (n + 1);

        // Update standard deviation
        var = ((n * var) + ((new_data_point - mn) * (new_data_point - mn))) / (n + 1);

        n += 1;
        last_added = now;
    }

    // Function to update mean / sd with a new data point
    void update(FloatType new_data_point) {
        update(new_data_point, FloatType(1), last_added);        
    }

    // Fit shape / scale to mean and sd
    void update() {
        shape = pow(mn, 2) / var;
        scale = var / mn;
    }

    // Getter for shape parameter
    FloatType getShape() const {
        return shape;
    }

    // Getter for scale parameter
    FloatType getScale() const {
        return scale;
    }
    // Getter for age
    FloatType getN() const {
        return n;
    }

    // Check if a single data point in the sample is an outlier based on alpha quantile
    bool isOutlier(FloatType data_point, FloatType alpha) const {      
        if (!(shape > 0)) {return true;}  
        // Compute critical chi-squared value for the given alpha and degrees of freedom        
        boost::math::gamma_distribution<FloatType> gamma_dist(shape, scale);
        FloatType critical_value = quantile(gamma_dist, alpha);

        // Check if the data point exceeds the critical value
        // std::cout << std::endl 
        //     << n << ", "
        //     << boost::math::mean(gamma_dist) << ", "  
        //     << boost::math::variance(gamma_dist) << ", " 
        //     << data_point << ", "
        //     << critical_value << std::endl;
        return data_point > critical_value;
    }
};

#endif

#ifndef TPSDOSTREAMC_H
#define TPSDOSTREAMC_H

#include <cmath>
#include <complex>

#include "SDOcluststream.h"

template<typename FloatType=double, typename ObservationType=std::vector<std::complex<FloatType>>>
class tpSDOstreamc : public SDOcluststream<FloatType, ObservationType> {
  public: 
  protected:
    const std::complex<FloatType> imag_unit{0.0, 1.0};
    std::size_t freq_bins;    
	FloatType max_freq;
};

#include "tpSDOstreamc_tree.h"

#endif  // TPSDOSTREAMC_H
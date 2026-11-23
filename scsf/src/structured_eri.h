/* SCSF; GPL-3.0-or-later. Spatial integrals in chemist order at the interface. */
#pragma once
#include <algorithm>

namespace scsf::structured {

inline size_t orbital_pair(size_t p,size_t q) {
    if(p<q)std::swap(p,q);
    return p*(p+1)/2+q;
}

enum class ERIFormat { dense, packed };

struct SpatialERI {
    const double *data;
    size_t k;
    ERIFormat format;

    double operator()(int p,int q,int r,int s) const {
        if(format==ERIFormat::dense)return data[((size_t(p)*k+r)*k+q)*k+s];
        const size_t a=orbital_pair(p,q),b=orbital_pair(r,s);
        return data[orbital_pair(a,b)];
    }
};

}

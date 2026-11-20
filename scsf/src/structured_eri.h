/* SCSF; GPL-3.0-or-later. Spatial integrals in chemist order at the interface. */
#pragma once
#include <algorithm>
#include <numeric>

namespace scsf::structured {

inline size_t orbital_pair(size_t p,size_t q) {
    if(p<q)std::swap(p,q);
    return p*(p+1)/2+q;
}

enum class ERIFormat { dense, cholesky, packed, thc };

struct SpatialERI {
    const double *data;
    const double *core;
    size_t k,rank;
    ERIFormat format;
    const double *transformed=nullptr;

    void transform_density(int p,int q,double *row) const {
        std::fill_n(row,rank,0.);
        for(size_t u=0;u<rank;++u) {
            const double left=data[p*rank+u]*data[q*rank+u];
            for(size_t v=0;v<rank;++v)row[v]+=left*core[u*rank+v];
        }
    }

    double operator()(int p,int q,int r,int s) const {
        if(format==ERIFormat::dense)return data[((size_t(p)*k+r)*k+q)*k+s];
        const size_t a=orbital_pair(p,q),b=orbital_pair(r,s);
        if(format==ERIFormat::packed)return data[orbital_pair(a,b)];
        if(!rank)return 0.;
        if(format==ERIFormat::cholesky) {
            const double *left=data+a*rank,*right=data+b*rank;
            return std::inner_product(left,left+rank,right,0.);
        }
        double value=0.;
        if(transformed) {
            for(size_t v=0;v<rank;++v)value+=transformed[a*rank+v]*data[r*rank+v]*data[s*rank+v];
            return value;
        }
        for(size_t u=0;u<rank;++u) {
            double right=0.;
            for(size_t v=0;v<rank;++v)right+=core[u*rank+v]*data[r*rank+v]*data[s*rank+v];
            value+=data[p*rank+u]*data[q*rank+u]*right;
        }
        return value;
    }
};

}

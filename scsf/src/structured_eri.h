/* SCSF; GPL-3.0-or-later. Spatial integrals in chemist order at the interface. */
#pragma once
#include "point_group.h"
#include <algorithm>
#include <limits>
#include <memory>

namespace scsf::structured {

inline size_t orbital_pair(size_t p,size_t q) {
    if(p<q)std::swap(p,q);
    return p*(p+1)/2+q;
}

inline size_t triangle_size(size_t n) {
    const size_t a=n%2 ? n/2+1 : n/2,b=n%2 ? n : n+1;
    if(a && b>std::numeric_limits<size_t>::max()/a)
        throw std::overflow_error("integral dimensions exceed index capacity");
    return a*b;
}

enum class ERIFormat { dense, packed, blocks };

class PairBlocks {
    std::vector<size_t> indices_;
    std::vector<unsigned char> irreps_;
    std::array<size_t,OrbitalSymmetry::extent+1> offsets_{};
public:
    PairBlocks(size_t k,const OrbitalSymmetry &symmetry)
        :indices_(triangle_size(k)),irreps_(indices_.size()) {
        std::array<size_t,OrbitalSymmetry::extent> sizes{};
        size_t id=0;
        for(size_t p=0;p<k;++p)for(size_t q=0;q<=p;++q,++id) {
            const auto code=symmetry.transfer(p,q);
            irreps_[id]=code;
            indices_[id]=sizes[code]++;
        }
        for(size_t i=0;i<sizes.size();++i) {
            const size_t count=triangle_size(sizes[i]);
            if(count>std::numeric_limits<size_t>::max()-offsets_[i])
                throw std::overflow_error("integral dimensions exceed index capacity");
            offsets_[i+1]=offsets_[i]+count;
        }
    }
    size_t size() const {return offsets_.back();}
    size_t bytes() const {
        return sizeof(*this)+indices_.capacity()*sizeof(size_t)+irreps_.capacity();
    }
    double at(const double *data,size_t a,size_t b) const {
        const auto code=irreps_[a];
        return code==irreps_[b] ? data[offsets_[code]+orbital_pair(indices_[a],indices_[b])] : 0.;
    }
};

struct SpatialERI {
    const double *data;
    size_t k;
    ERIFormat format;
    std::shared_ptr<const PairBlocks> blocks;

    double packed(size_t a,size_t b) const {
        return blocks ? blocks->at(data,a,b) : data[orbital_pair(a,b)];
    }
    double operator()(int p,int q,int r,int s) const {
        if(format==ERIFormat::dense)return data[((size_t(p)*k+r)*k+q)*k+s];
        const size_t a=orbital_pair(p,q),b=orbital_pair(r,s);
        return packed(a,b);
    }
};

}

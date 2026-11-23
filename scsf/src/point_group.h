/* SCSF; GPL-3.0-or-later. Real one-dimensional spatial irreps. */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace scsf {

struct OrbitalSymmetry {
    static constexpr unsigned extent=8;
    std::vector<unsigned> labels;
    std::vector<size_t> positions;
    std::array<std::vector<int>,extent> orbitals;
    std::array<std::uint64_t,extent> masks{};
    unsigned classes=0;

    OrbitalSymmetry()=default;
    OrbitalSymmetry(int k,std::vector<unsigned> irreps):labels(std::move(irreps)),positions(labels.size()) {
        if(!labels.empty() && labels.size()!=size_t(k))
            throw std::invalid_argument("point group needs one irrep per orbital");
        for(size_t p=0;p<labels.size();++p) {
            const auto g=labels[p];
            if(g>=extent)throw std::invalid_argument("orbital irrep code out of range");
            classes+=orbitals[g].empty();
            positions[p]=orbitals[g].size();
            orbitals[g].push_back(p);
            if(p<64)masks[g]|=std::uint64_t(1)<<p;
        }
    }
    bool enabled() const {return !labels.empty();}
    bool nontrivial() const {return classes>1;}
    unsigned operator[](size_t p) const {return labels.empty() ? 0 : labels[p];}
    unsigned transfer(size_t p,size_t q) const {return (*this)[p]^(*this)[q];}
    template<class Config> unsigned configuration(const Config &cfg) const {
        unsigned result=0;
        for(size_t p=0;p<labels.size();++p)if(cfg[p]==1)result^=labels[p];
        return result;
    }
};

} // namespace scsf

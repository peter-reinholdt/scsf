/* This file is part of SCSF, licensed under GNU GPL version 3 or later. */
#pragma once
#include "csf.h"

namespace scsf {
template<class Callback>
void same_irrep_orbitals(const OrbitalSymmetry &sym,int k,int q,Callback callback) {
    if(sym.nontrivial()) {
        for(int p:sym.orbitals[sym[q]])callback(p);
    } else {
        for(int p=0;p<k;++p)callback(p);
    }
}

template<class Callback>
void particle_pairs(const OrbitalSymmetry &sym,int k,int q,int s,Callback callback,
                    const Config *source=nullptr) {
    const auto code=sym.transfer(q,s);
    for(int p=0;p<k;++p) {
        if(p==q || p==s || (source && (*source)[p]==2))continue;
        auto visit=[&](int r) {if(r!=q && r!=s)callback(p,r);};
        if(sym.nontrivial()) {
            const auto &orbitals=sym.orbitals[code^sym[p]];
            for(auto r=std::lower_bound(orbitals.begin(),orbitals.end(),p);r!=orbitals.end();++r)visit(*r);
        } else for(int r=p;r<k;++r)visit(r);
    }
}

// Enumerate every minimal zero/one/two occupation transfer exactly once.
// Disjoint particle/hole multisets eliminate unchanged/spectator duplicates.
template<class Callback>
void transfers(const Config &source, const OrbitalSymmetry &symmetry, Callback callback) {
    validate_config(source);
    callback(std::array<int, 2>{}, std::array<int, 2>{}, 0);
    int k = static_cast<int>(source.size());
    for (int q = 0; q < k; ++q) {
        if (!source[q]) continue;
        same_irrep_orbitals(symmetry,k,q,[&](int p) {
            if (p != q && source[p] < 2) callback(std::array<int, 2>{q, 0}, std::array<int, 2>{p, 0}, 1);
        });
        for (int s = q; s < k; ++s) {
            if (source[s] < (q == s ? 2 : 1)) continue;
            particle_pairs(symmetry,k,q,s,[&](int p,int r) {
                if(source[r]<(p==r ? 1 : 2))
                    callback(std::array<int,2>{q,s},std::array<int,2>{p,r},2);
            },&source);
        }
    }
}

inline Config endpoint(const Config &source, const std::array<int, 2> &holes,
                const std::array<int, 2> &particles, int order) {
    Config target = source;
    for (int i = 0; i < order; ++i) { --target[holes[i]]; ++target[particles[i]]; }
    return target;
}

} // namespace scsf

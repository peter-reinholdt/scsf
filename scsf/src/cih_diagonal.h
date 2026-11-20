/* SCSF; GPL-3.0-or-later. CIH diagonal from one-generator column norms. */
#pragma once
#include "structured_internal.h"
#include "structured_eri.h"

namespace scsf::structured {

inline int cih_template_electrons(const Primary &p) {
    return std::min(p.n,p.maximum_selected_open()+4);
}

inline void cih_diagonal(const Primary::Group &group,const OneBank &bank,
        const double *h,const double *coulomb,const double *exchange,int k,double *result) {
    const auto &cfg=*group.cfg;
    double constant=0.;
    for(int p=0;p<k;++p)if(cfg[p]) {
        double term=h[size_t(p)*(k+1)];
        for(int q=0;q<k;++q)term+=.5*(cfg[q]*coulomb[size_t(p)*k+q]-exchange[size_t(p)*k+q]);
        constant+=cfg[p]*term;
    }
    for(size_t i=0;i<group.size();++i)result[group.row(i)]=constant;
    for(int p=0;p<k;++p)if(cfg[p])
        for(int q=0;q<k;++q) {
            if(p==q || cfg[q]==2)continue;
            const double weight=.5*exchange[size_t(p)*k+q];
            if(weight==0.)continue;
            const auto &norms=bank.get(cfg,q,p).squared_columns;
            for(size_t i=0;i<group.size();++i)
                result[group.row(i)]+=weight*norms[group.rank(i)];
        }
}
} // namespace scsf::structured

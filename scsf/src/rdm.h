/* SCSF; GPL-3.0-or-later. Spin-free density matrices by intermediate occupation. */
#pragma once
#include "generator_postings.h"

namespace scsf::v2 {

// T[p,q] = E_pq c. For each intermediate occupation contract its
// channel block once. The normal-order correction converts E_pq E_rs
// to the spin-summed a+_p a+_r a_s a_q operator.
inline Stats occupation_rdms(const structured::Primary &p,const double *x,double *one,double *two,
                             size_t link_bytes,size_t workspace_bytes,size_t coefficient_bytes) {
    const size_t k=p.k;
    const GeneratorPostings postings(p,link_bytes,coefficient_bytes,x);
    std::vector<double> amplitudes;
    for(const auto &[cfg,channels]:postings.groups) {
        const size_t target=p.native->find(Occupation(cfg));
        const auto *group=target==no_row ? nullptr : &p.groups[target];
        const size_t dim=postings.rows(channels.front());
        if(dim>workspace_bytes/sizeof(double)/channels.size())
            throw BudgetError("one density matrix occupation exceeds workspace budget");
        const size_t entries=dim*channels.size();
        if(entries>amplitudes.capacity())amplitudes=std::vector<double>(entries);
        else amplitudes.assign(entries,0.);
        for(size_t a=0;a<channels.size();++a) {
            const auto &channel=channels[a];
            const int r=channel.create,s=channel.destroy;
            double *current=amplitudes.data()+a*dim;
            postings.contract<false>(channel,x,current);
            if(group)
                for(size_t j=0;j<group->size();++j)one[size_t(r)*k+s]+=x[group->row(j)]*current[group->rank(j)];
        }
        for(size_t a=0;a<channels.size();++a) {
            const auto &left=channels[a];
            const int p0=left.destroy,q=left.create;
            const auto *begin=amplitudes.data()+a*dim;
            for(size_t b=0;b<channels.size();++b) {
                const auto &right=channels[b];
                const int r=right.create,s=right.destroy;
                const double *other=amplitudes.data()+b*dim;
                two[((size_t(p0)*k+r)*k+q)*k+s]+=std::inner_product(begin,begin+dim,other,0.);
            }
        }
    }
    for(size_t p0=0;p0<k;++p0)for(size_t q=0;q<k;++q)for(size_t s=0;s<k;++s)
        two[((p0*k+q)*k+q)*k+s]-=one[p0*k+s];
    auto stats=postings.info();
    stats["workspace_bytes"]=amplitudes.capacity()*sizeof(double);
    return stats;
}
} // namespace scsf::v2

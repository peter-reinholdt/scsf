/* SCSF; GPL-3.0-or-later. Frozen-space prototype with shared generator intermediates. */
#pragma once
#include "generator_postings.h"

namespace scsf::structured {

inline Stats shared_generator_product(const Primary &p,const double *h,
        const SpatialERI &eri,const double *x,double *y,
        size_t link_bytes,size_t workspace_bytes,size_t coefficient_bytes) {
    const auto start=Clock::now();
    const GeneratorPostings postings(p,link_bytes,coefficient_bytes,nullptr,nullptr,no_row,
                                     GeneratorPostings::Order::sector);
    auto stats=postings.info();
    stats["setup_ns"]=ns(start);
    const auto begin_product=Clock::now();
    const size_t k=p.k;
    const size_t capacity=workspace_bytes/sizeof(double);
    if(k*k>capacity)throw BudgetError("normal-ordered one-body table exceeds generator workspace");
    std::vector<double> shifted(k*k),work;
    for(size_t q=0;q<k;++q)for(size_t r=0;r<k;++r) {
        if(p.native->symmetry().transfer(q,r))continue;
        double value=h[r*k+q];
        for(size_t l=0;l<k;++l)value-=.5*eri(r,l,l,q);
        shifted[r*k+q]=value;
    }
    for(const auto &[cfg,channels]:postings.groups) {
        const size_t count=channels.size(),dim=postings.rows(channels.front());
        const size_t stride=checked_add(count,count);
        if(dim>(capacity-shifted.size())/stride)
            throw BudgetError("one shared intermediate occupation exceeds product workspace");
        const size_t size=stride*dim;
        if(size>work.capacity())work=std::vector<double>(size);
        else work.assign(size,0.);
        auto *forward=work.data();
        auto *mixed=forward+count*dim;
        const auto target=p.native->find(Occupation(cfg));
        for(size_t a=0;a<count;++a) {
            const auto &c=channels[a];
            double *amp=forward+a*dim;
            postings.contract<false>(c,x,amp);
            if(target!=no_row) {
                const auto &g=p.groups[target];
                const double value=shifted[size_t(c.create)*k+c.destroy];
                for(size_t i=0;i<g.size();++i)y[g.row(i)]+=value*amp[g.rank(i)];
            }
        }
        for(size_t first=0;first<count;) {
            const auto code=p.groups[channels[first].source].block->irrep;
            size_t end=first+1;
            while(end<count && p.groups[channels[end].source].block->irrep==code)++end;
            for(size_t a=first;a<end;++a)for(size_t b=first;b<end;++b) {
                const auto &left=channels[a],&right=channels[b];
                // The adjoint of the channel E_qp represents the left E_pq.
                const double scale=.5*eri(left.destroy,left.create,right.create,right.destroy);
                for(size_t i=0;i<dim;++i)mixed[a*dim+i]+=scale*forward[b*dim+i];
            }
            first=end;
        }
        for(size_t a=0;a<count;++a)postings.contract<true>(channels[a],mixed+a*dim,y);
    }
    stats["product_ns"]=ns(begin_product);
    stats["workspace_bytes"]=(shifted.capacity()+work.capacity())*sizeof(double);
    return stats;
}

} // namespace scsf::structured

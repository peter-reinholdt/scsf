/* SCSF; GPL-3.0-or-later. Frozen-space prototype with shared generator intermediates. */
#pragma once
#include "generator_postings.h"

namespace scsf::structured {

inline Stats shared_generator_product(const Primary &p,const double *h,
        const SpatialERI &eri,const double *x,double *y,
        size_t link_bytes,size_t workspace_bytes,size_t coefficient_bytes) {
    const auto start=Clock::now();
    const GeneratorPostings postings(p,link_bytes,coefficient_bytes);
    auto stats=postings.info();
    stats["setup_ns"]=ns(start);
    const auto begin_product=Clock::now();
    const size_t k=p.k;
    const size_t capacity=workspace_bytes/sizeof(double);
    if(k*k>capacity)throw BudgetError("normal-ordered one-body table exceeds generator workspace");
    std::vector<double> shifted(k*k),work;
    for(size_t q=0;q<k;++q)for(size_t r=0;r<k;++r) {
        double value=h[r*k+q];
        for(size_t l=0;l<k;++l)value-=.5*eri(r,l,l,q);
        shifted[r*k+q]=value;
    }
    for(const auto &[cfg,channels]:postings.groups) {
        const size_t count=channels.size(),dim=postings.rows(channels.front());
        const bool factored=eri.format==ERIFormat::cholesky && eri.rank<count/2;
        size_t stride=checked_add(2*count,factored ? eri.rank : 0);
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
        if(factored) {
            ++stats["cholesky_blocks"];
            double *projection=mixed+count*dim;
            for(size_t a=0;a<count;++a) {
                const auto &c=channels[a];
                const double *v=eri.data+orbital_pair(c.create,c.destroy)*eri.rank;
                for(size_t l=0;l<eri.rank;++l)for(size_t i=0;i<dim;++i)
                    projection[l*dim+i]+=v[l]*forward[a*dim+i];
            }
            for(size_t a=0;a<count;++a) {
                const auto &c=channels[a];
                const double *v=eri.data+orbital_pair(c.create,c.destroy)*eri.rank;
                for(size_t l=0;l<eri.rank;++l)for(size_t i=0;i<dim;++i)
                    mixed[a*dim+i]+=.5*v[l]*projection[l*dim+i];
            }
        } else {
            for(size_t a=0;a<count;++a)for(size_t b=0;b<count;++b) {
                const auto &left=channels[a],&right=channels[b];
                // The adjoint of the channel E_qp represents the left E_pq.
                const double scale=.5*eri(left.destroy,left.create,right.create,right.destroy);
                for(size_t i=0;i<dim;++i)mixed[a*dim+i]+=scale*forward[b*dim+i];
            }
        }
        for(size_t a=0;a<count;++a)postings.contract<true>(channels[a],mixed+a*dim,y);
    }
    stats["product_ns"]=ns(begin_product);
    stats["workspace_bytes"]=(shifted.capacity()+work.capacity())*sizeof(double);
    return stats;
}

} // namespace scsf::structured

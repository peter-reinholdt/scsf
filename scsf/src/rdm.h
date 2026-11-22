/* SCSF; GPL-3.0-or-later. Spin-free density matrices by intermediate occupation. */
#pragma once
#include "generator_postings.h"

namespace scsf::structured {

inline double generator_bilinear(const Primary::Group &target,const Primary::Group &source,
        const Matrix &matrix,const double *bra,const double *ket) {
    double value=0.;
    if(source.canonical_full()) {
        for(size_t i=0;i<target.size();++i) {
            const size_t rank=target.rank(i);
            double amplitude=0.;
            for(size_t j=matrix.ptr[rank];j<matrix.ptr[rank+1];++j)
                amplitude+=matrix.data[j]*ket[matrix.indices[j]];
            value+=bra[i]*amplitude;
        }
    } else {
        for(size_t i=0;i<target.size();++i) {
            const auto *row=matrix.values.data()+target.rank(i)*matrix.cols;
            double amplitude=0.;
            for(size_t j=0;j<source.size();++j)amplitude+=row[source.rank(j)]*ket[j];
            value+=bra[i]*amplitude;
        }
    }
    return value;
}

template<bool Transition>
inline size_t occupation_rdm1(const Primary &p,const double *bra,const double *ket,double *one,
        Workers &workers,size_t budget) {
    if(p.groups.empty() || !p.n)return 0;
    const size_t k=p.k,cells=k*k,bytes=cells*sizeof(double);
    const OneBank bank(cih_template_electrons(p),p.s,budget);
    constexpr size_t chunk=128;
    const size_t tasks=1+(p.groups.size()-1)/chunk;
    const size_t active=std::min({workers.count(),tasks,1+(budget-bank.bytes())/bytes});
    std::vector<double> partial((active-1)*cells);
    auto accumulate=[&](size_t index,size_t worker) {
        double *local=worker ? partial.data()+(worker-1)*cells : one;
        const auto &source=p.groups[index];
        const auto *source_bra=bra+source.block->offset,*source_ket=ket+source.block->offset;
        double overlap=0.;
        bool nonzero=false;
        for(size_t i=0;i<source.size();++i) {
            overlap+=source_bra[i]*source_ket[i];
            if constexpr(Transition)nonzero|=source_bra[i]!=0. || source_ket[i]!=0.;
            else nonzero|=source_ket[i]!=0.;
        }
        if(!nonzero)return;
        const auto &cfg=*source.cfg;
        const auto &original=source.block->key;
        // Visit each adjoint pair once and reuse the packed key for every probe.
        Occupation moved=original;
        for(int q=0;q<p.k;++q)if(cfg[q]) {
            local[q*(k+1)]+=overlap*cfg[q];
            const int qi=original.singles_before(q);
            for(int r=q+1;r<p.k;++r)if(cfg[r]<2) {
                const std::array<int,2> hole{q,0},particle{r,0};
                moved.transfer(hole,particle,1);
                const size_t target_index=p.native->find(moved);
                moved.transfer(particle,hole,1);
                if(target_index==no_row)continue;
                const auto &target=p.groups[target_index];
                const int pi=original.singles_before(r)+(cfg[q]!=1);
                const auto &matrix=bank.get(source.block->open,cfg[r],cfg[q],pi,qi);
                if(matrix.data.empty())continue;
                const auto *target_bra=bra+target.block->offset;
                local[r*k+q]+=generator_bilinear(target,source,matrix,target_bra,source_ket);
                if constexpr(Transition) {
                    const auto *target_ket=ket+target.block->offset;
                    local[q*k+r]+=generator_bilinear(target,source,matrix,target_ket,source_bra);
                }
            }
        }
    };
    if(active==1) {
        for(size_t i=0;i<p.groups.size();++i)accumulate(i,0);
    } else workers.run(p.groups.size(),accumulate,chunk,active);
    for(size_t offset=0;offset<partial.size();offset+=cells)
        for(size_t i=0;i<cells;++i)one[i]+=partial[offset+i];
    if constexpr(!Transition)
        for(size_t r=0;r<k;++r)for(size_t q=0;q<r;++q)one[q*k+r]=one[r*k+q];
    return checked_add(bank.bytes(),partial.capacity()*sizeof(double));
}

// The left channel is E_qp applied to the bra, and the right channel is
// E_rs applied to the ket. Their dot product gives <bra|E_pq E_rs|ket>.
template<bool Transition>
inline Stats occupation_rdm2(const Primary &p,const double *bra,const double *ket,double *one,double *two,
        Workers &workers,size_t link_bytes,size_t workspace_bytes,size_t budget) {
    const size_t k=p.k,cells=k*k,sets=Transition ? 2 : 1;
    const GeneratorPostings postings(p,link_bytes,budget,ket,Transition ? bra : nullptr,budget);
    auto stats=postings.info();
    const size_t storage=checked_add(stats.at("posting_bytes"),stats.at("coefficient_bytes"));
    const size_t available=std::min(workspace_bytes,budget-storage);
    const size_t capacity=available/sizeof(double);
    size_t largest=0,parallel_stride=0;
    for(const auto &[cfg,channels]:postings.groups) {
        const size_t dim=postings.rows(channels.front()),count=channels.size();
        if(dim>capacity/sets/count)
            throw BudgetError("one density matrix occupation exceeds workspace budget");
        const size_t entries=sets*dim*count;
        largest=std::max(largest,entries);
        parallel_stride=std::max(parallel_stride,checked_add(entries,count));
    }
    // Keep one dense tensor and cap the per-worker amplitudes and 1-RDMs
    // by the available workspace.
    size_t active=1;
    if(parallel_stride && postings.groups.size()>1 && k*sizeof(std::mutex)<=available) {
        const size_t parallel_capacity=(available-k*sizeof(std::mutex))/sizeof(double);
        active=std::max(size_t(1),std::min({workers.count(),postings.groups.size(),
            (parallel_capacity+cells)/(parallel_stride+cells)}));
    }
    const size_t stride=active>1 ? parallel_stride : largest;
    std::vector<double> work(active*stride),partial((active-1)*cells);
    std::vector<std::mutex> locks(active>1 ? k : 0);
    auto accumulate=[&](const Config &cfg,const std::vector<GeneratorPostings::Channel> &channels,size_t worker) {
        double *local=worker ? partial.data()+(worker-1)*cells : one;
        const size_t target=p.native->find(Occupation(cfg));
        const auto *group=target==no_row ? nullptr : &p.groups[target];
        const size_t dim=postings.rows(channels.front()),count=channels.size(),entries=dim*count;
        double *right=work.data()+worker*stride;
        double *left=Transition ? right+entries : right;
        std::fill_n(right,sets*entries,0.);
        for(size_t a=0;a<count;++a) {
            const auto &channel=channels[a];
            double *current=right+a*dim;
            postings.contract<false>(channel,ket,current);
            if constexpr(Transition)postings.contract<false>(channel,bra,left+a*dim);
            if(group) {
                double &element=local[size_t(channel.create)*k+channel.destroy];
                for(size_t j=0;j<group->size();++j)element+=bra[group->row(j)]*current[group->rank(j)];
            }
        }
        double *dots=right+sets*entries;
        for(size_t a=0;a<count;++a) {
            const size_t p0=channels[a].destroy,q=channels[a].create;
            const auto *begin=left+a*dim;
            auto add=[&](size_t b,double value) {
                const size_t r=channels[b].create,s=channels[b].destroy;
                two[((p0*k+r)*k+q)*k+s]+=value;
            };
            if(active==1) {
                for(size_t b=0;b<count;++b)add(b,std::inner_product(begin,begin+dim,right+b*dim,0.));
            } else {
                for(size_t b=0;b<count;++b)dots[b]=std::inner_product(begin,begin+dim,right+b*dim,0.);
                // The first orbital selects a disjoint slice of the output.
                std::lock_guard<std::mutex> lock(locks[p0]);
                for(size_t b=0;b<count;++b)add(b,dots[b]);
            }
        }
    };
    if(active==1) {
        for(const auto &[cfg,channels]:postings.groups)accumulate(cfg,channels,0);
    } else {
        auto next=postings.groups.cbegin();
        const auto end=postings.groups.cend();
        std::mutex queue;
        workers.each_worker([&](size_t worker) {
            for(;;) {
                auto first=end,last=end;
                {
                    std::lock_guard<std::mutex> lock(queue);
                    if(next==end)break;
                    first=next;
                    for(int i=0;i<8 && next!=end;++i)++next;
                    last=next;
                }
                for(;first!=last;++first)accumulate(first->first,first->second,worker);
            }
        },active);
    }
    for(size_t offset=0;offset<partial.size();offset+=cells)
        for(size_t i=0;i<cells;++i)one[i]+=partial[offset+i];
    // Normal ordering subtracts delta_qr <bra|E_ps|ket>.
    for(size_t p0=0;p0<k;++p0)for(size_t q=0;q<k;++q)for(size_t s=0;s<k;++s)
        two[((p0*k+q)*k+q)*k+s]-=one[p0*k+s];
    stats["workspace_bytes"]=(work.capacity()+partial.capacity())*sizeof(double)+locks.capacity()*sizeof(std::mutex);
    return stats;
}
} // namespace scsf::structured

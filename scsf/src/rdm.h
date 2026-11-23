/* SCSF; GPL-3.0-or-later. Spin-free density matrices by intermediate occupation. */
#pragma once
#include "generator_postings.h"

namespace scsf::structured {

inline unsigned density_irreps(const Primary &p,const double *bra,const double *ket) {
    if(!p.native->symmetry().enabled())return 1;
    unsigned left=0,right=0;
    for(const auto &group:p.groups) {
        const size_t start=group.block->offset;
        auto nonzero=[&](const double *x) {
            return std::any_of(x+start,x+start+group.size(),[](double c){return c!=0.;});
        };
        const unsigned code=1u<<group.block->irrep;
        if(nonzero(bra))left|=code;
        if(bra!=ket && nonzero(ket))right|=code;
    }
    if(bra==ket)right=left;
    unsigned mask=0;
    for(unsigned g=0;g<OrbitalSymmetry::extent;++g)if(left&(1u<<g))
        for(unsigned h=0;h<OrbitalSymmetry::extent;++h)if(right&(1u<<h))mask|=1u<<(g^h);
    return mask;
}

class OneDensityRows {
    const OrbitalSymmetry &sym_;
    size_t k_,cells_;
    unsigned allowed_;
    std::array<std::array<size_t,OrbitalSymmetry::extent>,OrbitalSymmetry::extent> local_{};
    std::vector<size_t> offsets_;
public:
    OneDensityRows(const Primary &p,unsigned allowed):sym_(p.native->symmetry()),k_(p.k),
        cells_(k_*k_),allowed_(allowed) {
        if(!sym_.nontrivial())return;
        std::array<size_t,OrbitalSymmetry::extent> lengths{};
        for(unsigned g=0;g<OrbitalSymmetry::extent;++g)
            for(unsigned h=0;h<OrbitalSymmetry::extent;++h)if(allowed_&(1u<<(g^h))) {
                local_[g][h]=lengths[g];
                lengths[g]+=sym_.orbitals[h].size();
            }
        cells_=0;
        offsets_.resize(k_);
        for(size_t p=0;p<k_;++p) {
            offsets_[p]=cells_;
            cells_+=lengths[sym_[p]];
        }
    }
    size_t cells() const {return cells_;}
    size_t bytes() const {return offsets_.capacity()*sizeof(size_t);}
    bool allowed(size_t p,size_t q) const {return allowed_&(1u<<sym_.transfer(p,q));}
    size_t index(size_t p,size_t q) const {
        return offsets_.empty() ? p*k_+q : offsets_[p]+local_[sym_[p]][sym_[q]]+sym_.positions[q];
    }
    template<class Visit>void columns(size_t p,Visit callback) const {
        if(offsets_.empty()) {
            for(size_t q=0;q<k_;++q)callback(q);
        } else {
            for(unsigned d=allowed_;d;d&=d-1)
                for(int q:sym_.orbitals[sym_[p]^unsigned(__builtin_ctz(d))])callback(q);
        }
    }
    void reduce(double *output,const std::vector<double> &partial) const {
        if(!cells_)return;
        for(size_t off=0;off<partial.size();off+=cells_)
            for(size_t p=0;p<k_;++p)columns(p,[&](size_t q) {
                output[p*k_+q]+=partial[off+index(p,q)];
            });
    }
};

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
    const size_t k=p.k;
    const auto allowed=density_irreps(p,bra,ket);
    if(!allowed)return 0;
    const OneDensityRows layout(p,allowed);
    const size_t cells=layout.cells(),bytes=cells*sizeof(double);
    if(!cells)return layout.bytes();
    budget=memory_left(budget,layout.bytes(),"one-particle density layout");
    const OneBank bank(cih_template_electrons(p),p.s,budget);
    constexpr size_t chunk=128;
    const size_t tasks=1+(p.groups.size()-1)/chunk;
    const size_t active=std::min({workers.count(),tasks,1+(budget-bank.bytes())/bytes});
    std::vector<double> partial((active-1)*cells);
    auto accumulate=[&](size_t index,size_t worker) {
        double *local=worker ? partial.data()+(worker-1)*cells : one;
        auto add=[&](size_t p,size_t q,double value) {
            local[worker ? layout.index(p,q) : p*k+q]+=value;
        };
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
            if(layout.allowed(q,q))add(q,q,overlap*cfg[q]);
            const int qi=original.singles_before(q);
            layout.columns(q,[&](int r) {
                if(r<=q || cfg[r]==2)return;
                const std::array<int,2> hole{q,0},particle{r,0};
                moved.transfer(hole,particle,1);
                const size_t target_index=p.native->find(moved);
                moved.transfer(particle,hole,1);
                if(target_index==no_row)return;
                const auto &target=p.groups[target_index];
                const int pi=original.singles_before(r)+(cfg[q]!=1);
                const auto &matrix=bank.get(source.block->open,cfg[r],cfg[q],pi,qi);
                if(matrix.data.empty())return;
                const auto *target_bra=bra+target.block->offset;
                add(r,q,generator_bilinear(target,source,matrix,target_bra,source_ket));
                if constexpr(Transition) {
                    const auto *target_ket=ket+target.block->offset;
                    add(q,r,generator_bilinear(target,source,matrix,target_ket,source_bra));
                }
            });
        }
    };
    if(active==1) {
        for(size_t i=0;i<p.groups.size();++i)accumulate(i,0);
    } else workers.run(p.groups.size(),accumulate,chunk,active);
    layout.reduce(one,partial);
    if constexpr(!Transition)
        for(size_t r=0;r<k;++r)for(size_t q=0;q<r;++q)one[q*k+r]=one[r*k+q];
    return checked_add(checked_add(bank.bytes(),layout.bytes()),partial.capacity()*sizeof(double));
}

class DensityBlockLayout {
    const OrbitalSymmetry &sym_;
    size_t k_,size_=0;
    std::vector<size_t> local_;
    std::array<size_t,OrbitalSymmetry::extent> counts_{};
    std::array<std::array<size_t,OrbitalSymmetry::extent>,OrbitalSymmetry::extent> offsets_{};
public:
    DensityBlockLayout(const Primary &p,unsigned mask,size_t budget,size_t output_cells)
        :sym_(p.native->symmetry()),k_(p.k) {
        if(k_ && k_>budget/sizeof(size_t)/k_)
            throw BudgetError("two-particle density index exceeds --max-mem");
        local_.resize(k_*k_);
        for(size_t p=0;p<k_;++p)for(size_t q=0;q<k_;++q)
            local_[p*k_+q]=counts_[sym_.transfer(p,q)]++;
        for(unsigned g=0;g<counts_.size();++g)for(unsigned h=0;h<counts_.size();++h) {
            if(!(mask&(1u<<(g^h))))continue;
            offsets_[g][h]=size_;
            if(counts_[g] && counts_[h]>(output_cells-size_)/counts_[g])
                throw BudgetError("density matrices exceed output budget");
            size_+=counts_[g]*counts_[h];
        }
    }
    size_t size() const {return size_;}
    size_t bytes() const {return local_.capacity()*sizeof(size_t);}
    size_t index(size_t p,size_t r,size_t q,size_t s) const {
        const auto g=sym_.transfer(p,q),h=sym_.transfer(r,s);
        return offsets_[g][h]+local_[p*k_+q]*counts_[h]+local_[r*k_+s];
    }
};

struct DensityDestination {
    double *two;
    size_t k;
    const DensityBlockLayout *blocks=nullptr;
    // An energy-only destination accumulates the two-body term privately.
    const SpatialERI *eri=nullptr;
    double &at(size_t p,size_t r,size_t q,size_t s) const {
        return two[blocks ? blocks->index(p,r,q,s) : ((p*k+r)*k+q)*k+s];
    }
};

struct DensityContraction {
    double energy;
    Stats stats;
};

template<bool Transition>
inline bool density_representative(size_t p,size_t r,size_t q,size_t s,size_t k) {
    const auto coordinate=std::make_pair(p*k+q,s*k+r);
    // Particle interchange preserves the normal-ordered density, including
    // directed transitions. Real expectation values also have adjoint symmetry.
    if(std::make_pair(r*k+s,q*k+p)<coordinate)return false;
    if constexpr(!Transition)
        if(std::make_pair(s*k+r,p*k+q)<coordinate || std::make_pair(q*k+p,r*k+s)<coordinate)
            return false;
    return true;
}

// E_qp on the bra and E_rs on the ket give <bra|E_pq E_rs|ket>.
template<bool Transition>
inline DensityContraction occupation_rdm2(const Primary &p,const double *bra,const double *ket,double *one,
        DensityDestination output,unsigned allowed,Workers &workers,size_t link_bytes,
        size_t workspace_bytes,size_t budget) {
    const size_t k=p.k;
    Stats empty{{"workspace_bytes",0},{"posting_bytes",0},{"coefficient_bytes",0},{"density_pair_products",0}};
    if(!allowed || (output.eri && !(allowed&1)))return {0.,std::move(empty)};
    if(output.eri)allowed=1;
    const auto &sym=p.native->symmetry();
    const OneDensityRows layout(p,allowed);
    budget=memory_left(budget,layout.bytes(),"one-particle density layout");
    using Order=GeneratorPostings::Order;
    const auto order=output.eri ? Order::sector : Transition ? Order::natural : Order::directed;
    const GeneratorPostings postings(p,link_bytes,budget,ket,Transition ? bra : nullptr,budget,order);
    auto stats=postings.info();
    const size_t storage=checked_add(stats.at("posting_bytes"),stats.at("coefficient_bytes"));
    const size_t available=std::min(workspace_bytes,budget-storage);
    const size_t cells=layout.cells(),partial_bytes=cells*sizeof(double);
    const auto source_irrep=[&](const auto &channel){return p.groups[channel.source].block->irrep;};
    const auto supported_sectors=[&](const auto &channels) {
        if(!output.eri || !sym.nontrivial())return (1u<<OrbitalSymmetry::extent)-1;
        unsigned left=0,right=0;
        for(const auto &channel:channels) {
            const auto code=1u<<source_irrep(channel),sides=postings.support(channel.source);
            if(sides&1)right|=code;
            if(sides&2)left|=code;
        }
        return left&right;
    };
    size_t amplitudes=0,max_indices=0,max_right=0;
    for(const auto &[cfg,channels]:postings.groups) {
        size_t nl=0,nr=0;
        const unsigned sectors=supported_sectors(channels);
        for(const auto &channel:channels) {
            if(!(sectors&(1u<<source_irrep(channel))))continue;
            const unsigned sides=postings.support(channel.source);
            nr+=bool(sides&1);
            nl+=bool(sides&2);
        }
        if(!nl || !nr)continue;
        const size_t dim=postings.rows(channels.front()),count=Transition ? checked_add(nr,nl) : nr;
        if(count>(available/sizeof(double))/dim)
            throw BudgetError("one density matrix occupation exceeds workspace budget");
        amplitudes=std::max(amplitudes,dim*count);
        max_indices=std::max(max_indices,count);
        max_right=std::max(max_right,nr);
    }
    if(!max_indices) {
        stats["workspace_bytes"]=layout.bytes();
        stats["density_pair_products"]=0;
        return {0.,std::move(stats)};
    }
    struct alignas(64) Products {double energy=0.;size_t pairs=0;};
    const size_t core=checked_add(sizeof(Products),
        checked_add(amplitudes*sizeof(double),max_indices*sizeof(size_t)));
    if(core>available)throw BudgetError("one density matrix occupation exceeds workspace budget");
    const size_t mutex_bytes=output.eri ? 0 : k*sizeof(std::mutex);
    const size_t dots=output.eri ? 0 : max_right;
    const size_t parallel=checked_add(core,checked_add(partial_bytes,dots*sizeof(double)));
    size_t active=1;
    if(postings.groups.size()>1 && mutex_bytes<=available) {
        const size_t free=available-mutex_bytes;
        active=std::max(size_t(1),std::min({workers.count(),postings.groups.size(),
            free/parallel+(free%parallel>=parallel-partial_bytes)}));
    }
    const size_t stride=amplitudes+(active>1 ? dots : 0);
    std::vector<double> work(active*stride),partial((active-1)*cells);
    std::vector<size_t> indices(active*max_indices);
    std::vector<std::mutex> locks(active>1 && !output.eri ? k : 0);
    std::vector<Products> products(active);
    auto accumulate=[&](const Config &cfg,const std::vector<GeneratorPostings::Channel> &channels,size_t worker) {
        const unsigned sectors=supported_sectors(channels);
        if constexpr(Transition) {
            unsigned sides=0;
            for(const auto &channel:channels)if(sectors&(1u<<source_irrep(channel)))
                sides|=postings.support(channel.source);
            if(sides!=3)return;
        }
        auto *right_indices=indices.data()+worker*max_indices;
        size_t nr=0,nl=0;
        for(size_t i=0;i<channels.size();++i)
            if((sectors&(1u<<source_irrep(channels[i]))) && (postings.support(channels[i].source)&1))
                right_indices[nr++]=i;
        if(!nr)return;
        auto *left_indices=Transition ? right_indices+nr : right_indices;
        if constexpr(Transition) {
            for(size_t i=0;i<channels.size();++i)
                if((sectors&(1u<<source_irrep(channels[i]))) && (postings.support(channels[i].source)&2))
                    left_indices[nl++]=i;
        } else nl=nr;
        if(!nl)return;
        double *local=worker ? partial.data()+(worker-1)*cells : one;
        const size_t target=p.native->find(Occupation(cfg));
        const auto *group=target!=no_row && (postings.support(target)&2) ? &p.groups[target] : nullptr;
        const size_t dim=postings.rows(channels.front());
        double *right=work.data()+worker*stride;
        double *left=Transition ? right+nr*dim : right;
        std::fill_n(right,(Transition ? nr+nl : nr)*dim,0.);
        for(size_t i=0;i<nr;++i) {
            const auto &channel=channels[right_indices[i]];
            double *current=right+i*dim;
            postings.contract<false>(channel,ket,current);
            if(group && layout.allowed(channel.create,channel.destroy)) {
                const size_t at=worker ? layout.index(channel.create,channel.destroy)
                                       : size_t(channel.create)*k+channel.destroy;
                for(size_t j=0;j<group->size();++j)local[at]+=bra[group->row(j)]*current[group->rank(j)];
            }
        }
        if constexpr(Transition)
            for(size_t i=0;i<nl;++i)postings.contract<false>(channels[left_indices[i]],bra,left+i*dim);
        std::array<size_t,OrbitalSymmetry::extent+1> right_sectors{};
        if(output.eri) {
            for(size_t b=0;b<nr;++b)++right_sectors[source_irrep(channels[right_indices[b]])+1];
            std::partial_sum(right_sectors.begin(),right_sectors.end(),right_sectors.begin());
        }
        double *dots_row=right+amplitudes;
        for(size_t a=0;a<nl;++a) {
            const auto &l=channels[left_indices[a]];
            const size_t p0=l.destroy,q=l.create;
            const auto *begin=left+a*dim;
            const size_t first=Transition ? 0 : a;
            auto dot=[&](size_t b) {
                return std::inner_product(begin,begin+dim,right+b*dim,0.);
            };
            if(output.eri) {
                const auto code=source_irrep(l);
                for(size_t b=std::max(first,right_sectors[code]);b<right_sectors[code+1];++b) {
                    const auto &r=channels[right_indices[b]];
                    double scale=(*output.eri)(p0,q,r.create,r.destroy);
                    if constexpr(!Transition)if(a!=b)
                        scale+=output.eri->format==ERIFormat::dense
                            ? (*output.eri)(r.destroy,r.create,q,p0) : scale;
                    if(scale==0.)continue;
                    products[worker].energy+=.5*scale*dot(b);
                    ++products[worker].pairs;
                }
            } else {
                auto representative=[&](size_t b) {
                    const auto &r=channels[right_indices[b]];
                    return density_representative<Transition>(p0,r.create,q,r.destroy,k);
                };
                auto add=[&](size_t b,double value) {
                    const auto &r=channels[right_indices[b]];
                    output.at(p0,r.create,q,r.destroy)+=value;
                };
                if(active==1) {
                    for(size_t b=first;b<nr;++b)if(representative(b)) {
                        add(b,dot(b));
                        ++products[worker].pairs;
                    }
                } else {
                    for(size_t b=first;b<nr;++b) {
                        const bool keep=representative(b);
                        dots_row[b]=keep ? dot(b) : 0.;
                        products[worker].pairs+=keep;
                    }
                    std::lock_guard<std::mutex> lock(locks[p0]);
                    for(size_t b=first;b<nr;++b)if(dots_row[b]!=0.)add(b,dots_row[b]);
                }
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
    layout.reduce(one,partial);
    double energy=0.;
    for(const auto &worker:products) {
        energy+=worker.energy;
        stats["density_pair_products"]+=worker.pairs;
    }
    if(!output.eri) {
        // Normal-order the representative before filling its orbit. Each
        // orbit has one writer, so this pass needs no orbital locks.
        workers.run(k,[&](size_t p0,size_t) {
            for(size_t q=0;q<k;++q)for(size_t s=Transition ? 0 : p0;s<k;++s) {
                const size_t first=!Transition && s==p0 ? q : 0;
                const auto finish=[&](size_t r) {
                    if(!density_representative<Transition>(p0,r,q,s,k))return;
                    const double subtract=q==r ? one[p0*k+s] : 0.;
                    const double value=output.at(p0,r,q,s)-subtract;
                    output.at(p0,r,q,s)=value;
                    output.at(r,p0,s,q)=value;
                    if constexpr(!Transition) {
                        output.at(s,q,r,p0)=value;
                        output.at(q,s,p0,r)=value;
                    }
                };
                if(sym.nontrivial()) {
                    const auto base=sym.transfer(p0,q)^sym[s];
                    for(unsigned d=allowed;d;d&=d-1) {
                        const auto &orbitals=sym.orbitals[base^unsigned(__builtin_ctz(d))];
                        for(auto r=std::lower_bound(orbitals.begin(),orbitals.end(),int(first));r!=orbitals.end();++r)
                            finish(*r);
                    }
                } else for(size_t r=first;r<k;++r)finish(r);
            }
        });
    } else for(size_t p0=0;p0<k;++p0)layout.columns(p0,[&](size_t s) {
        const double value=one[p0*k+s];
        if(value==0.)return;
        double exchange=0.;
        for(size_t q=0;q<k;++q)exchange+=(*output.eri)(p0,q,q,s);
        energy-=.5*exchange*value;
    });
    stats["workspace_bytes"]=(work.capacity()+partial.capacity())*sizeof(double)
        +indices.capacity()*sizeof(size_t)+locks.capacity()*sizeof(std::mutex)+layout.bytes()
        +products.capacity()*sizeof(Products);
    return {energy,std::move(stats)};
}
} // namespace scsf::structured

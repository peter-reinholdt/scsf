/* SCSF; GPL-3.0-or-later. Grouped unitary-generator products. */
#pragma once
#include "structured_internal.h"
#include "structured_eri.h"
#include "cih_diagonal.h"
#include <cstring>

namespace scsf::structured {

class GroupedUGA {
    using Word=std::uint64_t;
    static constexpr Word matrix_mask=(1u<<20)-1;
    static constexpr unsigned no_plan=std::numeric_limits<unsigned>::max();
    static_assert((OneBank::mask_electron_limit+1)*4u*
        (OneBank::mask_electron_limit+3)*(OneBank::mask_electron_limit+3)<=matrix_mask+1);

    struct Occupations {
        Word occupied=0,doubled=0;
        bool operator==(const Occupations &o) const {return occupied==o.occupied && doubled==o.doubled;}
        void remove(Word bit) {
            occupied^=bit&~doubled;
            doubled&=~bit;
        }
        void add(Word bit) {
            doubled|=bit&occupied;
            occupied|=bit;
        }
    };
    struct Slot {Occupations key; unsigned first=0,count=0;};
    struct Parent {unsigned group,orbital;};
    struct Group {Occupations occ;size_t first;bool complete;};
    struct Plan {
        double constant=0.;
        size_t singles_end=0;
        std::vector<Word> words;
    };
    struct Probe {Occupations key;size_t bucket,index;};
    struct Scratch {
        Plan plan;
        std::vector<double> forward,backward;
        std::vector<Probe> probes;
        std::vector<size_t> resolved;
        size_t built=0,hits=0,swept=0;
    };

    const Primary &primary_;
    Workers &workers_;
    OneBank bank_;
    size_t k_,k2_;
    Word orbitals_;
    SpatialERI integrals_;
    bool slot_sweep_;
    std::vector<double> one_,eri_;
    std::vector<Group> groups_;
    std::vector<Slot> slots_;
    std::vector<Parent> parents_;
    std::vector<std::unique_ptr<Plan>> cache_;
    std::vector<Scratch> scratch_;
    size_t fixed_bytes_=0,cache_limit_=0,complete_=0;
    std::atomic<size_t> cache_bytes_{0},cached_plans_{0};

    static Word bit(unsigned i) {return Word(1)<<i;}
    static unsigned pop(Word &mask) {
        unsigned i=__builtin_ctzll(mask);
        mask&=mask-1;
        return i;
    }
    static Word hash(const Occupations &o) {
        Word x=o.occupied^(o.doubled*0x9e3779b97f4a7c15ULL);
        x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
        x=(x^(x>>27))*0x94d049bb133111ebULL;
        return x^(x>>31);
    }
    static size_t probe_slot(const std::vector<Slot> &table,const Occupations &key,size_t i) {
        size_t mask=table.size()-1;
        while(table[i].count && !(table[i].key==key))i=(i+1)&mask;
        return i;
    }
    static size_t slot_index(const std::vector<Slot> &table,const Occupations &key) {
        return probe_slot(table,key,hash(key)&(table.size()-1));
    }
    template<class F>void removals(F visit) {
        for(unsigned g=0;g<groups_.size();++g) {
            const auto cfg=groups_[g].occ;
            for(Word mask=cfg.occupied;mask;) {
                unsigned q=pop(mask);
                auto key=cfg;
                key.remove(bit(q));
                visit(key,g,q);
            }
        }
    }
    static Word pack_term(unsigned address,unsigned matrix,unsigned pair=0) {
        return (Word(address)<<32)|(Word(matrix)<<12)|pair;
    }
    static unsigned address(Word word) {return word>>32;}
    static unsigned matrix(Word word) {return (word>>12)&matrix_mask;}
    static unsigned pair(Word word) {return word&4095;}
    static Word encode(double value) {
        Word word;
        std::memcpy(&word,&value,sizeof(word));
        return word;
    }
    static double decode(Word word) {
        double value;
        std::memcpy(&value,&word,sizeof(value));
        return value;
    }
    size_t first(unsigned group) const {
        return group==groups_.size() ? primary_.size() : groups_[group].first;
    }
    size_t count(unsigned group) const {return first(group+1)-first(group);}
    unsigned rank(unsigned group,size_t local) const {
        if(groups_[group].complete)return local;
        return primary_.groups[group].rank(local);
    }

    double integral(size_t row,size_t col) const {
        if(!eri_.empty())return eri_[row*k2_+col];
        return integrals_(row%k_,row/k_,col%k_,col/k_);
    }

    void build_plan(unsigned g,Scratch &scratch) const {
        auto &plan=scratch.plan;
        const auto cfg=groups_[g].occ;
        auto &words=plan.words;
        words.clear();
        plan.constant=0.;
        for(Word ps=cfg.occupied;ps;) {
            unsigned p=pop(ps),np=1+bool(cfg.doubled&bit(p));
            unsigned pp=p*(k_+1);
            plan.constant+=np*one_[pp];
            for(Word rs=cfg.occupied;rs;) {
                unsigned r=pop(rs),nr=1+bool(cfg.doubled&bit(r));
                plan.constant+=0.5*np*nr*integral(pp,r*(k_+1));
            }
        }

        // A residue lookup finds all ways to put the removed electron back
        // into the variational space. Fold diagonal generators into singles.
        std::array<const Slot *,64> removed{};
        for(Word qs=cfg.occupied;qs;) {
            unsigned q=pop(qs);
            auto residue=cfg;
            residue.remove(bit(q));
            const auto &range=slots_[slot_index(slots_,residue)];
            removed[q]=&range;
            for(unsigned i=range.first;i<range.first+range.count;++i) {
                auto [target,p]=parents_[i];
                if(target<=g || p==q)continue;
                unsigned pq=q*k_+p;
                const auto dest=groups_[target].occ;
                double value=one_[pq];
                for(Word xs=cfg.occupied|dest.occupied;xs;) {
                    unsigned x=pop(xs),xx=x*(k_+1);
                    const auto state=xx<pq ? dest : cfg;
                    unsigned n=bool(state.occupied&bit(x))+bool(state.doubled&bit(x));
                    value+=n*integral(pq,xx);
                }
                if(value==0.)continue;
                words.push_back(pack_term(target,bank_.index(cfg.occupied,cfg.doubled,p,q)));
                words.push_back(encode(value));
            }
        }
        plan.singles_end=words.size();

        // The off-diagonal generators are ordered by (annihilator, creator).
        // Link products that share E_rs so its intermediate vector is built once.
        auto generator_groups=[&](auto visit) {
            for(Word ss=cfg.occupied;ss;) {
                unsigned s=pop(ss);
                for(Word rs=orbitals_&~(cfg.doubled|bit(s));rs;) {
                    unsigned r=pop(rs);
                    auto intermediate=cfg;
                    intermediate.remove(bit(s));
                    intermediate.add(bit(r));
                    const unsigned a=bank_.index(cfg.occupied,cfg.doubled,r,s);
                    if(!bank_.at(a).rows)continue;
                    const Word lower=s==63 ? ~Word(0) : bit(s+1)-1;
                    visit(s,r,a,intermediate,intermediate.occupied&lower);
                }
            }
        };
        auto &probes=scratch.probes;
        auto &resolved=scratch.resolved;
        probes.clear();
        const size_t occupied=__builtin_popcountll(cfg.occupied);
        if(slot_sweep_ && k_*occupied*(occupied+1)<=4096) {
            generator_groups([&](unsigned,unsigned r,unsigned,Occupations mid,Word holes) {
                while(holes) {
                    const unsigned q=pop(holes);
                    if(q==r)continue;
                    auto key=mid;
                    key.remove(bit(q));
                    probes.push_back({key,hash(key)&(slots_.size()-1),probes.size()});
                }
            });
            std::sort(probes.begin(),probes.end(),[](const Probe &a,const Probe &b) {return a.bucket<b.bucket;});
            resolved.resize(probes.size());
            for(const auto &p:probes)resolved[p.index]=probe_slot(slots_,p.key,p.bucket);
            scratch.swept+=probes.size();
        }
        size_t probe=0;
        generator_groups([&](unsigned s,unsigned r,unsigned a,Occupations intermediate,Word qs) {
                const unsigned rs_pair=s*k_+r;
                size_t header=words.size();
                words.push_back(0);
                while(qs) {
                    unsigned q=pop(qs);
                    auto residue=intermediate;
                    residue.remove(bit(q));
                    const auto &range=q==r ? *removed[s] :
                        slots_[probes.empty() ? slot_index(slots_,residue) : resolved[probe++]];
                    for(unsigned i=range.first;i<range.first+range.count;++i) {
                        auto [target,p]=parents_[i];
                        unsigned pq=q*k_+p;
                        if(target<g || p==q || pq>rs_pair || integral(rs_pair,pq)==0.)continue;
                        words.push_back(pack_term(target,
                            bank_.index(intermediate.occupied,intermediate.doubled,p,q),pq));
                    }
                }
                size_t links=words.size()-header-1;
                if(!links)words.pop_back();
                else {
                    if(links>no_plan)throw BudgetError("grouped UGA generator has too many links");
                    words[header]=pack_term(links,a,rs_pair);
                }
        });
    }

    void retain(unsigned g,const Plan &plan) {
        if(cache_.empty())return;
        const size_t bytes=sizeof(Plan)+plan.words.size()*sizeof(Word);
        auto used=cache_bytes_.load(std::memory_order_relaxed);
        while(bytes<=cache_limit_-used) {
            if(cache_bytes_.compare_exchange_weak(used,used+bytes,std::memory_order_relaxed)) {
                try {
                    cache_[g]=std::make_unique<Plan>(plan);
                } catch (...) {
                    cache_bytes_.fetch_sub(bytes,std::memory_order_relaxed);
                    throw;
                }
                cached_plans_.fetch_add(1,std::memory_order_relaxed);
                return;
            }
        }
    }

    template<size_t Inner>void apply_links_vector(unsigned source,const Word *begin,const Word *end,
        unsigned first_pair,const Matrix &a,const double *input,double *output,
        const double *forward,double *backward) const {
        const size_t inner=Inner ? Inner : a.rows;
        for(auto it=begin;it!=end;++it) {
            Word term=*it;
            unsigned target=address(term),second_pair=pair(term);
            const Matrix &b=bank_.at(matrix(term));
            const double scale=integral(first_pair,second_pair)*(first_pair==second_pair ? 0.5 : 1.);
            const size_t offset=first(target),n=count(target);
            const bool transpose=target!=source;
            for(size_t i=0;i<n;++i) {
                const double *row=b.values.data()+rank(target,i)*inner;
                const double x=scale*input[offset+i];
                double sum=0.;
                for(size_t j=0;j<inner;++j)sum+=row[j]*forward[j];
                if(transpose)for(size_t j=0;j<inner;++j)backward[j]+=row[j]*x;
                output[offset+i]+=scale*sum;
            }
        }
    }

    template<size_t Width>void apply_plan(unsigned g,const Plan &plan,const double *input,double *output,
                                           size_t cols,Scratch &scratch) const {
        const size_t width=Width ? Width : cols;
        const double *x=input+first(g)*width;
        double *y=output+first(g)*width;
        const size_t n=count(g);
        for(size_t i=0;i<n*width;++i)y[i]+=plan.constant*x[i];
        const auto &words=plan.words;
        for(size_t pos=0;pos<plan.singles_end;pos+=2) {
            Word term=words[pos];
            unsigned target=address(term);
            const Matrix &a=bank_.at(matrix(term));
            double scale=decode(words[pos+1]);
            const double *tx=input+first(target)*width;
            double *ty=output+first(target)*width;
            for(size_t i=0;i<count(target);++i) {
                const double *row=a.values.data()+rank(target,i)*a.cols;
                for(size_t j=0;j<n;++j) {
                    double w=scale*row[rank(g,j)];
                    for(size_t v=0;v<width;++v) {
                        ty[i*width+v]+=w*x[j*width+v];
                        y[j*width+v]+=w*tx[i*width+v];
                    }
                }
            }
        }

        for(size_t pos=plan.singles_end;pos<words.size();) {
            const Word header=words[pos++];
            const Matrix &a=bank_.at(matrix(header));
            const size_t length=a.rows*width;
            scratch.forward.assign(length,0.);
            scratch.backward.assign(length,0.);
            double *forward=scratch.forward.data(),*backward=scratch.backward.data();
            for(size_t i=0;i<a.rows;++i) {
                const double *row=a.values.data()+i*a.cols;
                for(size_t j=0;j<n;++j) {
                    double w=row[rank(g,j)];
                    for(size_t v=0;v<width;++v)forward[i*width+v]+=w*x[j*width+v];
                }
            }
            size_t end=pos+address(header);
            if constexpr(Width==1) {
                const Word *begin=words.data()+pos,*stop=words.data()+end;
                const unsigned pq=pair(header);
                switch(a.rows) {
                    case 1: apply_links_vector<1>(g,begin,stop,pq,a,input,output,forward,backward); break;
                    case 2: apply_links_vector<2>(g,begin,stop,pq,a,input,output,forward,backward); break;
                    case 5: apply_links_vector<5>(g,begin,stop,pq,a,input,output,forward,backward); break;
                    default: apply_links_vector<0>(g,begin,stop,pq,a,input,output,forward,backward);
                }
                pos=end;
            } else {
                for(;pos<end;++pos) {
                    Word term=words[pos];
                    unsigned target=address(term);
                    const Matrix &b=bank_.at(matrix(term));
                    const double scale=integral(pair(header),pair(term))*(pair(header)==pair(term) ? 0.5 : 1.);
                    const double *tx=input+first(target)*width;
                    double *ty=output+first(target)*width;
                    for(size_t i=0;i<count(target);++i) {
                        const double *row=b.values.data()+rank(target,i)*a.rows;
                        for(size_t j=0;j<a.rows;++j) {
                            double w=scale*row[j];
                            for(size_t v=0;v<width;++v)ty[i*width+v]+=w*forward[j*width+v];
                            if(target!=g)for(size_t v=0;v<width;++v)backward[j*width+v]+=w*tx[i*width+v];
                        }
                    }
                }
            }
            for(size_t i=0;i<a.rows;++i) {
                const double *row=a.values.data()+i*a.cols;
                for(size_t j=0;j<n;++j) {
                    double w=row[rank(g,j)];
                    for(size_t v=0;v<width;++v)y[j*width+v]+=w*backward[i*width+v];
                }
            }
        }
    }

public:
    GroupedUGA(const Primary &p,const double *h,const SpatialERI &integrals,Workers &w,
               size_t metadata,size_t coefficients,size_t plan_limit=std::numeric_limits<size_t>::max(),bool slot_sweep=false):
        primary_(p),workers_(w),bank_(v2::cih_template_electrons(p),p.s,coefficients,true),k_(p.k),k2_(k_*k_),
        orbitals_(k_==64 ? ~Word(0) : bit(k_)-1),integrals_(integrals),slot_sweep_(slot_sweep),scratch_(w.count()) {
        if(p.size()>=no_plan)throw BudgetError("grouped UGA basis exceeds index capacity");
        groups_.reserve(p.groups.size());
        for(const auto &g:p.groups) {
            const auto &block=*g.block;
            groups_.push_back({{block.key.first[0],block.key.first[1]},block.offset,block.complete()});
            complete_+=block.complete();
        }
        const bool dense_table=integrals.format==ERIFormat::dense;
        fixed_bytes_=groups_.capacity()*sizeof(Group)+(k2_+(dense_table ? k2_*k2_ : 0))*sizeof(double);
        if(fixed_bytes_>metadata)throw BudgetError("grouped UGA exceeds metadata budget");
        size_t records=0,keys=0;
        if(!groups_.empty()) {
            const size_t slot_limit=(metadata-fixed_bytes_)/sizeof(Slot);
            if(slot_limit<2)throw BudgetError("grouped UGA residue hash exceeds metadata budget");
            slots_.resize(2);
            removals([&](const Occupations &key,unsigned,unsigned) {
                if(!slots_[slot_index(slots_,key)].count) {
                    if(keys>=slots_.size()*2/3) {
                        if(slots_.size()>slot_limit/2)throw BudgetError("grouped UGA residue hash exceeds metadata budget");
                        std::vector<Slot> grown(slots_.size()*2);
                        for(const auto &entry:slots_)if(entry.count)grown[slot_index(grown,entry.key)]=entry;
                        slots_.swap(grown);
                    }
                    ++keys;
                }
                auto &entry=slots_[slot_index(slots_,key)];
                entry.key=key;
                ++entry.count;
                ++records;
            });
            fixed_bytes_+=slots_.capacity()*sizeof(Slot);
            if(records>=no_plan || records>(metadata-fixed_bytes_)/sizeof(Parent))
                throw BudgetError("grouped UGA residue links exceed metadata budget");
            parents_.resize(records);
            unsigned next=0;
            // Reserve each range, then fill backwards without a separate cursor array.
            for(auto &entry:slots_)if(entry.count) {next+=entry.count; entry.first=next;}
            removals([&](const Occupations &key,unsigned group,unsigned orbital) {
                auto &entry=slots_[slot_index(slots_,key)];
                parents_[--entry.first]={group,orbital};
            });
            fixed_bytes_+=parents_.capacity()*sizeof(Parent);
        }
        if(plan_limit && groups_.size() && groups_.size()<=(metadata-fixed_bytes_)/sizeof(cache_[0])) {
            cache_.resize(groups_.size());
            fixed_bytes_+=cache_.capacity()*sizeof(cache_[0]);
            cache_limit_=std::min(plan_limit,metadata-fixed_bytes_);
        }

        // H = K E + sum_(pq<rs) (pq|rs) E_pq E_rs + 1/2 sum_pq (pq|pq) E_pq^2.
        // K includes both normal ordering and the generator commutator.
        one_.resize(k2_);
        if(dense_table) {
            eri_.resize(k2_*k2_);
            workers_.run(k2_,[&](size_t rs,size_t) {
                for(size_t pq=0;pq<k2_;++pq)
                    eri_[rs*k2_+pq]=integrals(pq%k_,pq/k_,rs%k_,rs/k_);
            },8);
        }
        for(size_t q=0;q<k_;++q)for(size_t p0=0;p0<k_;++p0) {
            double value=h[p0*k_+q];
            for(size_t l=0;l<k_;++l) {
                const size_t left=l*k_+p0,right=q*k_+l;
                const double g=integral(right,left);
                value-=0.5*g;
                if(left<right)value-=0.5*g;
                if(left>right)value+=0.5*g;
            }
            one_[q*k_+p0]=value;
        }
    }

    void diagonal(size_t group,const double *h,const double *coulomb,const double *exchange,double *result) const {
        v2::cih_diagonal(primary_.groups[group],bank_,h,coulomb,exchange,k_,result);
    }
    size_t entries() const {return workers_.count()>1 ? primary_.size()*workers_.count() : 0;}
    Stats info() const {
        size_t cached=cache_bytes_.load(std::memory_order_relaxed);
        return {{"metadata_bytes",fixed_bytes_+cached},{"coefficient_bytes",bank_.bytes()},
            {"grouped_uga",1},{"slot_sweep",slot_sweep_},{"grouped_dense_eri_bytes",eri_.capacity()*sizeof(double)},{"complete_configurations",complete_},{"residue_links",parents_.size()},
            {"plan_cache_bytes",cached},{"plan_cache_limit",cache_limit_},
            {"cached_plans",cached_plans_.load(std::memory_order_relaxed)},{"workspace_entries_per_rhs",entries()}};
    }
    void apply(const double *x,size_t cols,double *y,Stats &stats,bool serial=false) {
        if(groups_.empty())return;
        auto start=Clock::now();
        size_t n=primary_.size(),stride=n*cols,nworkers=serial ? 1 : workers_.count();
        std::unique_ptr<double[]> output(nworkers>1 ? new double[nworkers*stride] : nullptr);
        if(nworkers>1)workers_.each_worker([&](size_t w) {
            std::fill_n(output.get()+w*stride,stride,0.);
        });
        for(auto &scratch:scratch_)scratch.built=scratch.hits=scratch.swept=0;
        auto run=[&](size_t g,size_t w) {
            auto &scratch=scratch_[w];
            const Plan *plan=cache_.empty() ? nullptr : cache_[g].get();
            if(!plan) {
                ++scratch.built;
                build_plan(g,scratch);
                plan=&scratch.plan;
                retain(g,*plan);
            } else ++scratch.hits;
            double *out=nworkers==1 ? y : output.get()+w*stride;
            if(cols==1)apply_plan<1>(g,*plan,x,out,cols,scratch);
            else apply_plan<0>(g,*plan,x,out,cols,scratch);
        };
        if(serial)for(size_t g=0;g<groups_.size();++g)run(g,0);
        else workers_.run(groups_.size(),run,4);
        stats["contraction_ns"]+=ns(start);
        start=Clock::now();
        if(nworkers>1)workers_.run(n,[&](size_t i,size_t) {
            for(size_t v=0;v<cols;++v) {
                double sum=0.;
                for(size_t w=0;w<nworkers;++w)sum+=output[w*stride+i*cols+v];
                y[i*cols+v]=sum;
            }
        },256);
        stats["reduction_ns"]+=ns(start);
        for(const auto &s:scratch_) {
            stats["plans_built"]+=s.built;
            stats["plan_hits"]+=s.hits;
            stats["slot_sweep_probes"]+=s.swept;
            stats["worker_temporary_bytes"]+=
                (s.forward.capacity()+s.backward.capacity())*sizeof(double)+s.plan.words.capacity()*sizeof(Word)+
                s.probes.capacity()*sizeof(Probe)+s.resolved.capacity()*sizeof(size_t);
        }
    }

};
}

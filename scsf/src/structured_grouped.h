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
    struct Group {Occupations occ;size_t first;unsigned count;bool complete;};
    struct Plan {
        double constant=0.;
        size_t singles_end=0;
        std::vector<Word> words;
    };
    struct Probe {Occupations key;Word hash;size_t index;};
    enum class PlanReuse {not_attempted, empty_or_shrunk, different_bank, suffix_limit,
                          changed_prefix, direct_scan, suffix_index};
    struct Scratch {
        Plan plan;
        std::vector<double> forward,backward;
        std::vector<Probe> probes;
        std::vector<size_t> resolved;
        size_t built=0,hits=0,swept=0,clear_ns=0,finish_ns=0,plan_ns=0;
    };

    const Primary &primary_;
    Workers &workers_;
    std::shared_ptr<const OneBank> bank_;
    bool reused_bank_;
    size_t k_,k2_;
    Word orbitals_;
    SpatialERI integrals_;
    bool slot_sweep_;
    std::vector<double> one_,eri_;
    std::vector<std::uint16_t> packed_pairs_;
    std::vector<Group> groups_;
    std::vector<Slot> slots_;
    std::vector<unsigned char> fingerprints_;
    std::vector<Parent> parents_;
    std::vector<std::shared_ptr<const Plan>> cache_;
    std::vector<Scratch> scratch_;
    std::unique_ptr<double[]> vector_workspace_;
    size_t fixed_bytes_=0,build_bytes_=0,complete_=0,reused_plans_=0;
    PlanReuse reuse_reason_=PlanReuse::not_attempted;
    size_t reuse_scan_ns_=0;
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
    static unsigned char fingerprint(Word value) {
        return 1+(value>>57);
    }
    size_t find_slot(const Occupations &key,Word value) const {
        const size_t mask=slots_.size()-1;
        size_t i=value&mask;
        // Most generator probes miss. Scan compact control bytes and load
        // the full occupation only after a fingerprint match.
        const auto tag=fingerprint(value);
        while(fingerprints_[i]) {
            if(fingerprints_[i]==tag && slots_[i].key==key)return i;
            i=(i+1)&mask;
        }
        return none;
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
    size_t first(unsigned group) const {return groups_[group].first;}
    size_t count(unsigned group) const {return groups_[group].count;}
    unsigned rank(unsigned group,size_t local) const {
        if(groups_[group].complete)return local;
        return primary_.groups[group].rank(local);
    }

    double integral(size_t row,size_t col) const {
        if(!eri_.empty())return eri_[row*k2_+col];
        if(!packed_pairs_.empty())return integrals_.packed(packed_pairs_[row],packed_pairs_[col]);
        return integrals_(row%k_,row/k_,col%k_,col/k_);
    }
    std::pair<const Parent *,const Parent *> matching(const Slot &slot,unsigned code) const {
        const auto *first=parents_.data()+slot.first,*last=first+slot.count;
        const auto &sym=primary_.native->symmetry();
        if(!sym.nontrivial())return {first,last};
        const auto *begin=std::lower_bound(first,last,code,
            [&](const Parent &parent,unsigned g){return sym[parent.orbital]<g;});
        const auto *end=std::upper_bound(begin,last,code,
            [&](unsigned g,const Parent &parent){return g<sym[parent.orbital];});
        return {begin,end};
    }

    void build_plan(unsigned g,Scratch &scratch) const {
        auto &plan=scratch.plan;
        const auto cfg=groups_[g].occ;
        const auto &sym=primary_.native->symmetry();
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
            auto [first,last]=matching(range,sym[q]);
            for(;first!=last;++first) {
                auto [target,p]=*first;
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
                words.push_back(pack_term(target,bank_->index(cfg.occupied,cfg.doubled,p,q)));
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
                    const unsigned a=bank_->index(cfg.occupied,cfg.doubled,r,s);
                    if(!bank_->at(a).rows)continue;
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
            generator_groups([&](unsigned s,unsigned r,unsigned,Occupations mid,Word holes) {
                while(holes) {
                    const unsigned q=pop(holes);
                    if(q==r)continue;
                    if(sym.nontrivial() && !sym.masks[sym[q]^sym.transfer(r,s)])continue;
                    auto key=mid;
                    key.remove(bit(q));
                    probes.push_back({key,hash(key),probes.size()});
                }
            });
            const size_t mask=slots_.size()-1;
            std::sort(probes.begin(),probes.end(),[mask](const Probe &a,const Probe &b) {
                return (a.hash&mask)<(b.hash&mask);
            });
            resolved.resize(probes.size());
            for(const auto &p:probes)resolved[p.index]=find_slot(p.key,p.hash);
            scratch.swept+=probes.size();
        }
        size_t probe=0;
        generator_groups([&](unsigned s,unsigned r,unsigned a,Occupations intermediate,Word qs) {
                const unsigned rs_pair=s*k_+r;
                size_t header=words.size();
                words.push_back(0);
                while(qs) {
                    unsigned q=pop(qs);
                    const auto creator=sym[q]^sym.transfer(r,s);
                    if(sym.nontrivial() && !sym.masks[creator])continue;
                    auto residue=intermediate;
                    residue.remove(bit(q));
                    const Slot *range=removed[s];
                    if(q!=r) {
                        const auto id=probes.empty() ? find_slot(residue,hash(residue)) : resolved[probe++];
                        if(id==none)continue;
                        range=&slots_[id];
                    }
                    auto [first,last]=matching(*range,creator);
                    for(;first!=last;++first) {
                        auto [target,p]=*first;
                        unsigned pq=q*k_+p;
                        if(target<g || p==q || pq>rs_pair || integral(rs_pair,pq)==0.)continue;
                        words.push_back(pack_term(target,
                            bank_->index(intermediate.occupied,intermediate.doubled,p,q),pq));
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

    static size_t plan_bytes(const Plan &plan) {
        return sizeof(Plan)+2*sizeof(void *)+plan.words.size()*sizeof(Word);
    }
    void retain(unsigned g,const Plan &plan,size_t limit) {
        const size_t bytes=plan_bytes(plan);
        auto used=cache_bytes_.load(std::memory_order_relaxed);
        do {
            memory_left(limit,checked_add(used,bytes),"CIH plan cache");
        } while(!cache_bytes_.compare_exchange_weak(used,used+bytes,std::memory_order_relaxed));
        try {
            cache_[g]=std::make_shared<const Plan>(plan);
        } catch (...) {
            cache_bytes_.fetch_sub(bytes,std::memory_order_relaxed);
            throw;
        }
        cached_plans_.fetch_add(1,std::memory_order_relaxed);
    }

    template<size_t Inner,bool Sparse=false>void apply_links_vector(unsigned source,const Word *begin,const Word *end,
        unsigned first_pair,const Matrix &a,const double *input,double *output,
        const double *forward,double *backward) const {
        const size_t inner=Inner ? Inner : a.rows;
        for(auto it=begin;it!=end;++it) {
            Word term=*it;
            unsigned target=address(term),second_pair=pair(term);
            const Matrix &b=bank_->at(matrix(term));
            const double scale=integral(first_pair,second_pair)*(first_pair==second_pair ? 0.5 : 1.);
            const size_t offset=first(target),n=count(target);
            const bool transpose=target!=source;
            if constexpr(Sparse)if(groups_[target].complete) {
                for(size_t i=0;i<n;++i) {
                    const double x=scale*input[offset+i];
                    double sum=0.;
                    for(size_t j=b.ptr[i];j<b.ptr[i+1];++j) {
                        const auto col=b.indices[j];
                        const double value=b.data[j];
                        sum+=value*forward[col];
                        if(transpose)backward[col]+=value*x;
                    }
                    output[offset+i]+=scale*sum;
                }
                continue;
            }
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
            const Matrix &a=bank_->at(matrix(term));
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

        std::array<double,16> short_forward,short_backward;
        for(size_t pos=plan.singles_end;pos<words.size();) {
            const Word header=words[pos++];
            const Matrix &a=bank_->at(matrix(header));
            const size_t length=a.rows*width;
            const bool sparse_a=Width==1 && groups_[g].complete && a.rows==14 && a.cols==14 &&
                a.data.size()<14*14/2;
            double *forward,*backward;
            if(Width==1 && length<=short_forward.size()) {
                forward=short_forward.data();
                backward=short_backward.data();
                std::fill_n(forward,length,0.);
                std::fill_n(backward,length,0.);
            } else {
                scratch.forward.assign(length,0.);
                scratch.backward.assign(length,0.);
                forward=scratch.forward.data();
                backward=scratch.backward.data();
            }
            if(sparse_a) {
                for(size_t i=0;i<14;++i) {
                    double sum=0.;
                    for(size_t j=a.ptr[i];j<a.ptr[i+1];++j)sum+=a.data[j]*x[a.indices[j]];
                    forward[i]=sum;
                }
            } else for(size_t i=0;i<a.rows;++i) {
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
                    // This intermediate rank has mostly structural zeros.
                    case 14:
                        if(groups_[g].complete)
                            apply_links_vector<14,true>(g,begin,stop,pq,a,input,output,forward,backward);
                        else apply_links_vector<0>(g,begin,stop,pq,a,input,output,forward,backward);
                        break;
                    default: apply_links_vector<0>(g,begin,stop,pq,a,input,output,forward,backward);
                }
                pos=end;
            } else {
                for(;pos<end;++pos) {
                    Word term=words[pos];
                    unsigned target=address(term);
                    const Matrix &b=bank_->at(matrix(term));
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
            if(sparse_a) {
                for(size_t i=0;i<14;++i)
                    for(size_t j=a.ptr[i];j<a.ptr[i+1];++j)y[a.indices[j]]+=a.data[j]*backward[i];
            } else for(size_t i=0;i<a.rows;++i) {
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
               size_t budget,bool slot_sweep=false,const GroupedUGA *previous=nullptr):
        primary_(p),workers_(w),bank_(previous ? previous->bank_ :
            std::make_shared<const OneBank>(cih_template_electrons(p),p.s,budget,true)),
        reused_bank_(previous!=nullptr),k_(p.k),k2_(k_*k_),
        orbitals_(k_==64 ? ~Word(0) : bit(k_)-1),integrals_(integrals),slot_sweep_(slot_sweep),scratch_(w.count()) {
        const size_t metadata=memory_left(budget,bank_->bytes(),"CIH spin coefficients");
        if(p.size()>=no_plan)throw BudgetError("grouped UGA basis exceeds index capacity");
        groups_.reserve(p.groups.size());
        for(const auto &g:p.groups) {
            const auto &block=*g.block;
            groups_.push_back({{block.key.first[0],block.key.first[1]},block.offset,
                static_cast<unsigned>(block.size()),block.complete()});
            complete_+=block.complete();
        }
        const auto &sym=primary_.native->symmetry();
        const bool dense_table=integrals.format==ERIFormat::dense && !sym.nontrivial();
        const bool packed=integrals.format!=ERIFormat::dense;
        fixed_bytes_=groups_.capacity()*sizeof(Group)+(k2_+(dense_table ? k2_*k2_ : 0))*sizeof(double)
            +(packed ? k2_*sizeof(packed_pairs_[0]) : 0);
        if(fixed_bytes_>metadata)throw BudgetError("grouped UGA exceeds metadata budget");
        size_t records=0,keys=0;
        if(!groups_.empty()) {
            const size_t slot_limit=(metadata-fixed_bytes_)/sizeof(Slot);
            if(slot_limit<2)throw BudgetError("grouped UGA residue hash exceeds metadata budget");
            slots_.resize(2);
            removals([&](const Occupations &key,unsigned,unsigned) {
                if(!slots_[slot_index(slots_,key)].count) {
                    if(keys>=slots_.size()*2/3) {
                        if(slots_.size()>slot_limit/3)throw BudgetError("CIH residue hash resize exceeds --max-mem");
                        build_bytes_=std::max(build_bytes_,fixed_bytes_+3*slots_.size()*sizeof(Slot));
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
            if(sym.nontrivial())workers_.run(slots_.size(),[&](size_t i,size_t) {
                const auto &entry=slots_[i];
                if(entry.count<2)return;
                auto first=parents_.begin()+entry.first;
                std::sort(first,first+entry.count,[&](const Parent &a,const Parent &b) {
                    const auto ag=sym[a.orbital],bg=sym[b.orbital];
                    return ag!=bg ? ag<bg : a.group>b.group;
                });
            },256);
            fixed_bytes_+=parents_.capacity()*sizeof(Parent);
            memory_left(metadata-fixed_bytes_,slots_.size(),"CIH hash fingerprints");
            fingerprints_.resize(slots_.size());
            workers_.run(slots_.size(),[&](size_t i,size_t) {
                if(slots_[i].count)fingerprints_[i]=fingerprint(hash(slots_[i].key));
            },512);
            fixed_bytes_+=fingerprints_.capacity();
        }
        memory_left(metadata-fixed_bytes_,groups_.size()*sizeof(cache_[0]),"CIH plan directory");
        cache_.resize(groups_.size());
        fixed_bytes_+=cache_.capacity()*sizeof(cache_[0]);
        build_bytes_=std::max(build_bytes_,fixed_bytes_);

        // H = K E + sum_(pq<rs) (pq|rs) E_pq E_rs + 1/2 sum_pq (pq|pq) E_pq^2.
        // K includes both normal ordering and the generator commutator.
        one_.resize(k2_);
        if(packed) {
            packed_pairs_.resize(k2_);
            for(size_t q=0;q<k_;++q)for(size_t p0=0;p0<k_;++p0)
                packed_pairs_[q*k_+p0]=orbital_pair(q,p0);
        }
        if(dense_table) {
            eri_.resize(k2_*k2_);
            workers_.run(k2_,[&](size_t rs,size_t) {
                for(size_t pq=0;pq<k2_;++pq)
                    eri_[rs*k2_+pq]=integrals(pq%k_,pq/k_,rs%k_,rs/k_);
            },8);
        }
        for(size_t q=0;q<k_;++q)for(size_t p0=0;p0<k_;++p0) {
            if(sym.transfer(p0,q))continue;
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

    bool can_share_bank(const Primary &p) const {
        return p.s==primary_.s && cih_template_electrons(p)==cih_template_electrons(primary_);
    }

    void reuse_plans(const GroupedUGA &old) {
        const auto start=Clock::now();
        const size_t prefix=old.groups_.size();
        if(cache_.empty() || old.cache_.empty() || prefix>groups_.size()) {
            reuse_reason_=PlanReuse::empty_or_shrunk;
            return;
        }
        if(bank_!=old.bank_) {
            reuse_reason_=PlanReuse::different_bank;
            return;
        }
        const size_t added=groups_.size()-prefix;
        const bool direct=added<=1024 && (added<=256 || prefix<=(size_t(1)<<27)/added);
        constexpr size_t tile=32;
        struct Summary {Occupations all,any;};
        std::array<Summary,128> summary;
        if(!direct && (added>tile*summary.size() || old.complete_!=prefix || complete_!=groups_.size())) {
            reuse_reason_=PlanReuse::suffix_limit;
            return;
        }
        for(size_t g=0;g<prefix;++g)if(!(groups_[g].occ==old.groups_[g].occ)) {
            reuse_reason_=PlanReuse::changed_prefix;
            return;
        }
        const auto distance=[](const Occupations &a,const Occupations &b) {
            return __builtin_popcountll(a.occupied^b.occupied)+__builtin_popcountll(a.doubled^b.doubled);
        };
        // Plans encode indices in the shared spin bank and occupation prefix.
        // A new partner within two electron transfers can change their links.
        if(!direct)for(size_t first=0;first<added;first+=tile) {
            auto &entry=summary[first/tile];
            entry.all=entry.any=groups_[prefix+first].occ;
            for(size_t j=first+1;j<std::min(first+tile,added);++j) {
                const auto occ=groups_[prefix+j].occ;
                entry.all.occupied&=occ.occupied;
                entry.all.doubled&=occ.doubled;
                entry.any.occupied|=occ.occupied;
                entry.any.doubled|=occ.doubled;
            }
        }
        workers_.run(prefix,[&](size_t g,size_t) {
            const auto &plan=old.cache_[g];
            if(!plan)return;
            const auto src=groups_[g].occ;
            if(direct) {
                for(size_t j=prefix;j<groups_.size();++j)
                    if(primary_.groups[g].block->irrep==primary_.groups[j].block->irrep &&
                        distance(src,groups_[j].occ)<=4)return;
            } else for(size_t first=0;first<added;first+=tile) {
                const auto &entry=summary[first/tile];
                // These bits differ from every occupation in the tile, so
                // their count is a lower bound on all exact distances.
                const auto occupied=(src.occupied&~entry.any.occupied)|(entry.all.occupied&~src.occupied);
                const auto doubled=(src.doubled&~entry.any.doubled)|(entry.all.doubled&~src.doubled);
                if(__builtin_popcountll(occupied)+__builtin_popcountll(doubled)>4)continue;
                for(size_t j=first;j<std::min(first+tile,added);++j)
                    if(primary_.groups[g].block->irrep==primary_.groups[prefix+j].block->irrep &&
                        distance(src,groups_[prefix+j].occ)<=4)return;
            }
            cache_[g]=plan;
        },64);
        reuse_reason_=direct ? PlanReuse::direct_scan : PlanReuse::suffix_index;
        reuse_scan_ns_=ns(start);
        size_t used=0;
        for(auto &plan:cache_)if(plan) {
            const size_t bytes=plan_bytes(*plan);
            used=checked_add(used,bytes);
            ++reused_plans_;
        }
        cache_bytes_.store(used,std::memory_order_relaxed);
        cached_plans_.store(reused_plans_,std::memory_order_relaxed);
    }

    void diagonal(size_t group,const double *h,const double *coulomb,const double *exchange,double *result) const {
        cih_diagonal(primary_.groups[group],*bank_,h,coulomb,exchange,k_,result);
    }
    size_t entries() const {return workers_.count()>1 ? primary_.size()*workers_.count() : 0;}
    size_t workspace_bytes() const {return vector_workspace_ ? entries()*sizeof(double) : 0;}
    void release_workspace() {vector_workspace_.reset();}
    size_t cached_bytes() const {return cache_bytes_.load(std::memory_order_relaxed);}
    Stats info() const {
        size_t cached=cached_bytes();
        return {{"metadata_bytes",fixed_bytes_+cached},{"metadata_build_bytes",build_bytes_},{"coefficient_bytes",bank_->bytes()},
            {"grouped_uga",1},{"reused_spin_bank",reused_bank_},{"slot_sweep",slot_sweep_},{"grouped_dense_eri_bytes",eri_.capacity()*sizeof(double)},{"complete_configurations",complete_},{"residue_links",parents_.size()},
            {"plan_cache_bytes",cached},
            {"reuse_reason",static_cast<size_t>(reuse_reason_)},{"reuse_scan_ns",reuse_scan_ns_},
            {"cached_plans",cached_plans_.load(std::memory_order_relaxed)},{"reused_plans",reused_plans_},
            {"workspace_entries_per_rhs",entries()}};
    }
    void apply(const double *x,size_t cols,double *y,Stats &stats,size_t plan_limit) {
        if(groups_.empty())return;
        auto start=Clock::now();
        size_t n=primary_.size(),stride=n*cols,nworkers=workers_.count();
        std::unique_ptr<double[]> block_workspace;
        double *output=nullptr;
        if(nworkers>1) {
            if(cols==1) {
                if(!vector_workspace_)vector_workspace_.reset(new double[nworkers*stride]);
                output=vector_workspace_.get();
            } else {
                block_workspace.reset(new double[nworkers*stride]);
                output=block_workspace.get();
            }
        }
        auto run=[&](size_t g,size_t w) {
            auto &scratch=scratch_[w];
            const Plan *plan=cache_[g].get();
            if(!plan) {
                const auto plan_start=Clock::now();
                ++scratch.built;
                build_plan(g,scratch);
                plan=&scratch.plan;
                retain(g,*plan,plan_limit);
                scratch.plan_ns+=ns(plan_start);
            } else ++scratch.hits;
            double *out=nworkers==1 ? y : output+w*stride;
            if(cols==1)apply_plan<1>(g,*plan,x,out,cols,scratch);
            else if(cols==2)apply_plan<2>(g,*plan,x,out,cols,scratch);
            else apply_plan<0>(g,*plan,x,out,cols,scratch);
        };
        const size_t groups=groups_.size();
        // Keep enough chunks for irregular plans while amortizing claims on
        // large spaces. First-touch each private output in the same job.
        const size_t chunk=std::clamp(groups/nworkers/32,size_t(4),size_t(32));
        std::atomic<size_t> next{0};
        const auto dispatch=Clock::now();
        workers_.each_worker([&](size_t w) {
            auto &scratch=scratch_[w];
            scratch.built=scratch.hits=scratch.swept=0;
            scratch.clear_ns=scratch.plan_ns=0;
            if(nworkers==1) {
                for(size_t g=0;g<groups;++g)run(g,w);
            } else {
                const auto clear_start=Clock::now();
                std::fill_n(output+w*stride,stride,0.);
                scratch.clear_ns=ns(clear_start);
                for(size_t first=next.fetch_add(chunk,std::memory_order_relaxed);first<groups;
                    first=next.fetch_add(chunk,std::memory_order_relaxed)) {
                    for(size_t g=first;g<std::min(first+chunk,groups);++g)run(g,w);
                }
            }
            scratch.finish_ns=ns(dispatch);
        });
        stats["contraction_ns"]+=ns(start);
        start=Clock::now();
        if(nworkers>1) {
            // Stream each worker's contiguous slice. Each tile owns its output
            // elements and still sums worker contributions in worker order.
            constexpr size_t tile=4096;
            workers_.run(stride/tile+(stride%tile!=0),[&](size_t t,size_t) {
                const size_t first=t*tile,last=std::min(first+tile,stride);
                for(size_t w=0;w<nworkers;++w) {
                    const double *local=output+w*stride;
                    for(size_t i=first;i<last;++i)y[i]+=local[i];
                }
            });
        }
        stats["reduction_ns"]+=ns(start);
        size_t finish_min=scratch_.front().finish_ns,finish_sum=0;
        for(const auto &s:scratch_) {
            finish_min=std::min(finish_min,s.finish_ns);
            finish_sum+=s.finish_ns;
            stats["worker_finish_max_ns"]=std::max(stats["worker_finish_max_ns"],s.finish_ns);
            stats["worker_plan_build_max_ns"]=std::max(stats["worker_plan_build_max_ns"],s.plan_ns);
            stats["worker_plan_build_sum_ns"]+=s.plan_ns;
            stats["workspace_clear_max_ns"]=std::max(stats["workspace_clear_max_ns"],s.clear_ns);
            stats["plans_built"]+=s.built;
            stats["plan_hits"]+=s.hits;
            stats["slot_sweep_probes"]+=s.swept;
            stats["worker_temporary_bytes"]+=
                (s.forward.capacity()+s.backward.capacity())*sizeof(double)+s.plan.words.capacity()*sizeof(Word)+
                s.probes.capacity()*sizeof(Probe)+s.resolved.capacity()*sizeof(size_t);
        }
        stats["worker_finish_min_ns"]=finish_min;
        stats["worker_finish_mean_ns"]=finish_sum/nworkers;
    }

};
}

/* SCSF; GPL-3.0-or-later. Deterministic CSF Epstein–Nesbet PT2. */
#pragma once
#include "selection.h"
#include <queue>
#include <sstream>

namespace scsf::structured {
namespace pt2 {

inline size_t bytes_for(size_t count,size_t width) {
    if(width && count>no_row/width)throw BudgetError("ENPT2 workspace size overflow");
    return count*width;
}

struct Transfer {
    std::array<int,2> holes,particles;
    int order;

    static constexpr int orbital_bits=7;
    static constexpr int packed_orbitals=1<<orbital_bits;
    std::uint32_t pack() const {
        return std::uint32_t(holes[0])|(std::uint32_t(holes[1])<<orbital_bits)|
            (std::uint32_t(particles[0])<<(2*orbital_bits))|(std::uint32_t(particles[1])<<(3*orbital_bits))|
            (std::uint32_t(order)<<(4*orbital_bits));
    }
    static Transfer unpack(std::uint32_t move) {
        const auto site=[=](int position) {return int((move>>(orbital_bits*position))&(packed_orbitals-1));};
        return {{site(0),site(1)},{site(2),site(3)},int(move>>(4*orbital_bits))};
    }
};

struct Link {
    Occupation::Limb low;
    size_t source;
    // Store either the short second limb or the index into the side arrays.
    std::uint32_t high,move;

    static bool inline_high(size_t orbitals) {return orbitals>64 && orbitals<=80;}
};

struct Sum {
    double value=0.,error=0.;
    void add(double term) {
        const double next=value+term;
        error+=std::abs(value)>=std::abs(term) ? (value-next)+term : (term-next)+value;
        value=next;
    }
    double total() const {return value+error;}
};

struct Root {
    Sum energy;
    double minimum=std::numeric_limits<double>::infinity();
    size_t positive=0,contributing=0;
    void merge(const Root &other) {
        energy.add(other.energy.total());
        minimum=std::min(minimum,other.minimum);
        positive+=other.positive;
        contributing+=other.contributing;
    }
};

struct Counts {
    size_t leaves=0,splits=0,singletons=0,visits=0,screened=0,connections=0;
    size_t external_occupations=0,external_ranks=0,pair_elements=0,diagonals=0,largest_leaf=0;
    size_t generate_ns=0,sort_ns=0,contract_ns=0,diagonal_prepare_ns=0,diagonal_apply_ns=0;
    size_t path_hits=0,path_misses=0,target_paths=0,block_contractions=0;
    size_t prefix_source_checks=0;

    void merge(const Counts &other) {
        leaves+=other.leaves;
        splits+=other.splits;
        singletons+=other.singletons;
        visits+=other.visits;
        screened+=other.screened;
        connections+=other.connections;
        external_occupations+=other.external_occupations;
        external_ranks+=other.external_ranks;
        pair_elements+=other.pair_elements;
        diagonals+=other.diagonals;
        largest_leaf=std::max(largest_leaf,other.largest_leaf);
        generate_ns+=other.generate_ns;
        sort_ns+=other.sort_ns;
        contract_ns+=other.contract_ns;
        diagonal_prepare_ns+=other.diagonal_prepare_ns;
        diagonal_apply_ns+=other.diagonal_apply_ns;
        path_hits+=other.path_hits;
        path_misses+=other.path_misses;
        target_paths+=other.target_paths;
        block_contractions+=other.block_contractions;
        prefix_source_checks+=other.prefix_source_checks;
    }
};

struct Ordered {
    HCITransferBounds *bounds;
    bool doubles=false,singles=false;
};

// All entries in a prefix, including empty sites, are fixed. A complete
// minimal transfer consists of mandatory differences followed by a
// multiset of holes and particles in the unfixed orbitals.
template<class Visit>
void constrained_transfers(const Config &source,const OrbitalSymmetry &sym,
    const signed char *fixed,const std::vector<int> &sites,int depth,
    const Ordered &ordered,double weight,double eps,Visit visit) {
    const int k=source.size();
    std::array<int,2> fixed_holes{},fixed_particles{};
    int nh=0,np=0;
    for(int i=0;i<depth;++i) {
        const int p=sites[i],diff=int(fixed[p])-source[p];
        if(nh-diff>2 || np+diff>2)return;
        for(int n=0;n<-diff;++n)fixed_holes[nh++]=p;
        for(int n=0;n<diff;++n)fixed_particles[np++]=p;
    }
    if(nh==2 && fixed_holes[0]>fixed_holes[1])std::swap(fixed_holes[0],fixed_holes[1]);
    if(np==2 && fixed_particles[0]>fixed_particles[1])std::swap(fixed_particles[0],fixed_particles[1]);
    const auto free=[&](int p) {return !fixed || fixed[p]<0;};
    if(!nh && !np)visit(Transfer{},false);
    const auto particles_for=[&](const std::array<int,2> &holes,int order) {
        const int q=holes[0],s=holes[1];
        auto emit=[&](int p,int r,bool bounded) {
            if(p==q || source[p]==2)return;
            if(order==2 && (p==s || r==q || r==s || source[r]>=(p==r ? 1 : 2)))return;
            visit(Transfer{holes,{p,r},order},bounded);
        };
        if(order==1) {
            if(np==1) {
                const int p=fixed_particles[0];
                if(sym[p]==sym[q])emit(p,0,false);
            } else if(ordered.singles) {
                auto [it,end]=ordered.bounds->ordered_singles(q,k);
                for(;it!=end;++it) {
                    if(screened(it->bound,weight,eps,k))break;
                    if(free(it->p))emit(it->p,0,true);
                }
            } else {
                same_irrep_orbitals(sym,k,q,[&](int p) {if(free(p))emit(p,0,false);});
            }
            return;
        }
        const unsigned irrep=sym.transfer(q,s);
        if(np==2) {
            const auto [p,r]=fixed_particles;
            if(sym.transfer(p,r)==irrep)emit(p,r,false);
        } else if(ordered.doubles) {
            auto [it,end]=ordered.bounds->ordered_transfers(q,s,k);
            for(;it!=end;++it) {
                if(screened(it->bound,weight,eps,k))break;
                const int a=fixed_particles[0];
                if(np==1 ? (it->p==a && free(it->r)) || (it->r==a && free(it->p))
                         : free(it->p) && free(it->r))
                    emit(it->p,it->r,true);
            }
        } else if(np==1) {
            const int r=fixed_particles[0];
            auto complete=[&](int p) {
                if(free(p))emit(std::min(p,r),std::max(p,r),false);
            };
            if(sym.nontrivial()) {
                for(int p:sym.orbitals[irrep^sym[r]])complete(p);
            } else for(int p=0;p<k;++p)complete(p);
        } else {
            particle_pairs(sym,k,q,s,[&](int p,int r) {
                if(free(p) && free(r))emit(p,r,false);
            },&source);
        }
    };
    for(int order=std::max({1,nh,np});order<=2;++order) {
        const int missing=order-nh;
        if(!missing)particles_for(fixed_holes,order);
        else for(int q=0;q<k;++q) {
            if(!source[q] || !free(q))continue;
            if(missing==1) {
                const int s=fixed_holes[0];
                particles_for(order==1 ? std::array<int,2>{q,0}
                                       : std::array<int,2>{std::min(q,s),std::max(q,s)},order);
            } else for(int s=q;s<k;++s)
                if(free(s) && source[s]>=(q==s ? 2 : 1))particles_for({q,s},order);
        }
    }
}

class PrefixSources {
    struct Bucket {
        unsigned occupied=0,doubled=0;
        size_t first=0,last=0;
    };
    int depth_;
    std::vector<Bucket> buckets_;
    std::vector<size_t> sources_;

    static unsigned bucket_count(int depth) {
        unsigned count=1;
        for(int i=0;i<depth;++i)count*=3;
        return count;
    }
public:
    static size_t reservation(int depth,size_t sources) {
        const size_t count=bucket_count(depth);
        return checked_add(bytes_for(count,sizeof(Bucket)+2*sizeof(size_t)),
                           checked_add(bytes_for(sources,sizeof(size_t)),sizeof(PrefixSources)));
    }

    PrefixSources(int depth,const Primary &primary,const std::vector<double> &weights,
                  const std::vector<int> &sites,size_t sources) : depth_(depth),sources_(sources) {
        const size_t count=bucket_count(depth_);
        std::vector<size_t> offsets(count),next(count);
        const auto code=[&](size_t source) {
            const auto &cfg=*primary.groups[source].cfg;
            unsigned value=0;
            for(int i=depth_-1;i>=0;--i)value=3*value+cfg[sites[i]];
            return value;
        };
        for(size_t source=0;source<weights.size();++source)
            if(weights[source])++offsets[code(source)];
        buckets_.reserve(count);
        size_t begin=0;
        for(size_t i=0;i<count;++i) {
            if(!offsets[i])continue;
            const size_t end=begin+offsets[i];
            Bucket bucket{0,0,begin,end};
            for(unsigned value=i,bit=1;value;value/=3,bit<<=1) {
                const unsigned digit=value%3;
                if(digit)bucket.occupied|=bit;
                if(digit==2)bucket.doubled|=bit;
            }
            buckets_.push_back(bucket);
            next[i]=begin;
            begin=end;
        }
        for(size_t source=0;source<weights.size();++source)
            if(weights[source])sources_[next[code(source)]++]=source;
    }

    template<class Visit>
    void compatible(const unsigned char *digits,int depth,Visit visit) const {
        const int fixed=std::min(depth_,depth);
        const unsigned mask=(1u<<fixed)-1;
        unsigned occupied=0,doubled=0;
        for(int i=0;i<fixed;++i) {
            occupied|=unsigned(digits[i]>0)<<i;
            doubled|=unsigned(digits[i]==2)<<i;
        }
        for(const auto &bucket:buckets_) {
            const auto present=bucket.occupied&mask,paired=bucket.doubled&mask;
            // Their common projected removal residue is the bitwise
            // intersection. Each side must reach it in at most two removals.
            if(__builtin_popcount(present&~occupied)+__builtin_popcount(paired&~doubled)>2 ||
               __builtin_popcount(occupied&~present)+__builtin_popcount(doubled&~paired)>2)continue;
            for(size_t i=bucket.first;i<bucket.last;++i)visit(sources_[i]);
        }
    }
};

struct Context {
    const Primary &primary;
    const double *one;
    const SpatialERI &eri;
    const double *coefficients,*energies;
    size_t roots;
    double eps,tolerance,diagonal_guard;
    Ordered ordered;
    const std::vector<double> &weights,&root_weights,&coulomb_exchange,&occupation_terms;
    const std::vector<std::unique_ptr<SpinRanks>> &spin;
    const std::vector<int> &sites;
    const std::vector<std::vector<double>> &diagonal_factors;
    const OneBank *bank;
    const PrefixSources *prefix_sources;
    int maximum_block_open,maximum_factor_input;
    bool block_screening;

    double connection_bound(const Config &source,const Occupation &key,const Transfer &move) const {
        const auto *bounds=ordered.bounds;
        return !bounds ? transfer_bound(source,move.holes,move.particles,move.order,one,eri,primary.k)
            : move.order==1 ? bounds->single_bound(source,move.particles[0],move.holes[0],primary.k,key)
            : bounds->selection_bound(source,move.holes,move.particles,move.order,primary.k);
    }
    bool reject(const Config &source,const Occupation &key,const Transfer &move,double maximum) const {
        return eps && move.order && screened(connection_bound(source,key,move),maximum,eps,primary.k);
    }
};

struct Prefix {
    size_t code;
    int depth;
    size_t cost=0,records=0;
};

constexpr int census_depth=13;
constexpr size_t large_census_sources=100000;

struct CensusSource { size_t id,weight; };

struct CensusStratum {
    size_t sources=0,samples=0,visited=0,selected=0;
    size_t boundary(size_t item) const {
        return item*(sources/samples)+item*(sources%samples)/samples;
    }
};

inline std::vector<CensusSource> census_sources(const Context &ctx,size_t sources,size_t active) {
    const bool large=sources>=large_census_sources;
    const size_t target=std::min(sources,large ? size_t(8192) : size_t(1536)+active*4);
    // Keep rare spin classes in the large census. A single high-spin
    // configuration can contribute substantially more work than most sources.
    std::vector<CensusStratum> strata(large ? ctx.spin.size() : 1);
    auto stratum=[&](size_t id) -> CensusStratum & {
        return strata[large ? ctx.primary.groups[id].block->open : 0];
    };
    if(large) {
        for(size_t id=0;id<ctx.weights.size();++id)
            if(ctx.weights[id])++stratum(id).sources;
    } else strata[0].sources=sources;
    size_t sample_count=0;
    for(auto &s:strata) {
        s.samples=std::min(s.sources,large ? std::max(size_t(32),size_t(double(target)*s.sources/sources)) : target);
        sample_count+=s.samples;
    }
    std::vector<CensusSource> sample;
    sample.reserve(sample_count);
    for(size_t id=0;id<ctx.weights.size() && sample.size()<sample_count;++id) {
        if(!ctx.weights[id])continue;
        auto &s=stratum(id);
        if(s.selected==s.samples)continue;
        const size_t first=s.boundary(s.selected),last=s.boundary(s.selected+1);
        if(s.visited++==first+(large ? (last-first)/2 : 0)) {
            sample.push_back({id,large ? last-first : 1});
            ++s.selected;
        }
    }
    return sample;
}

inline std::vector<Prefix> plan_prefixes(const Context &ctx,Workers &workers,size_t active,
        int depth,size_t tasks,size_t capacity,size_t sources,int sampled_depth,size_t record_capacity) {
    std::vector<Prefix> plan;
    plan.reserve(capacity);
    if(!sampled_depth) {
        for(size_t i=0;i<tasks;++i)plan.push_back({i,depth});
        return plan;
    }
    const size_t k=ctx.primary.k;
    const bool large=sources>=large_census_sources;
    const auto sample=census_sources(ctx,sources,active);
    std::array<size_t,census_depth+1> power{1};
    for(int j=1;j<=sampled_depth;++j)power[j]=3*power[j-1];
    const size_t buckets=power[sampled_depth];
    std::vector<std::atomic<size_t>> histogram(buckets),record_histogram(buckets);
    for(auto &value:histogram)value.store(0,std::memory_order_relaxed);
    for(auto &value:record_histogram)value.store(0,std::memory_order_relaxed);
    std::vector<size_t> stride(k);
    for(int i=0;i<sampled_depth;++i)stride[ctx.sites[i]]=power[sampled_depth-1-i];
    const auto &sym=ctx.primary.native->symmetry();
    workers.run(sample.size(),[&](size_t item,size_t) {
        const auto [id,weight]=sample[item];
        const auto &group=ctx.primary.groups[id];
        const auto &cfg=*group.cfg;
        size_t base=0;
        for(int p=0;p<sampled_depth;++p)base+=cfg[ctx.sites[p]]*stride[ctx.sites[p]];
        const size_t source_cost=std::min(group.size(),large ? size_t(1024) : size_t(64));
        constrained_transfers(cfg,sym,nullptr,ctx.sites,0,ctx.ordered,ctx.weights[id],ctx.eps,
            [&](const Transfer &move,bool bounded) {
                if(!move.order && group.block->complete())return;
                if((!bounded || (ctx.block_screening && move.order==1)) &&
                   ctx.reject(cfg,group.block->key,move,ctx.weights[id]))return;
                size_t code=base;
                int open=group.block->open;
                for(int j=0;j<move.order;++j) {
                    code-=stride[move.holes[j]];
                    code+=stride[move.particles[j]];
                    if(move.order==1 || move.holes[0]!=move.holes[1])
                        open+=cfg[move.holes[j]]==1 ? -1 : 1;
                    if(move.order==1 || move.particles[0]!=move.particles[1])
                        open+=cfg[move.particles[j]]==1 ? -1 : 1;
                }
                const auto &target=ctx.spin[open];
                if(!target || !target->size())return;
                const size_t target_cost=std::min(target->size(),large ? size_t(4096) : size_t(128));
                // Bank contractions do two rectangular products through an
                // intermediate spin class. Scalar contractions visit pairs.
                const bool blocked=ctx.bank && open<=ctx.maximum_block_open &&
                    group.block->open<=ctx.maximum_factor_input &&
                    (!large || group.block->open!=ctx.maximum_factor_input || open<ctx.maximum_block_open);
                const size_t work=blocked ?
                    8+3*source_cost+(2+std::min(source_cost,size_t(8)))*target_cost :
                    (large ? 8 : 1)*source_cost*target_cost;
                histogram[code].fetch_add(work*weight,std::memory_order_relaxed);
                record_histogram[code].fetch_add(weight,std::memory_order_relaxed);
            });
    },1,active);
    std::vector<size_t> cumulative(buckets+1),cumulative_records(buckets+1);
    for(size_t i=0;i<buckets;++i) {
        cumulative[i+1]=checked_add(cumulative[i],histogram[i].load(std::memory_order_relaxed));
        cumulative_records[i+1]=checked_add(cumulative_records[i],record_histogram[i].load(std::memory_order_relaxed));
    }
    if(!cumulative.back()) {
        for(size_t i=0;i<tasks;++i)plan.push_back({i,depth});
        return plan;
    }
    auto begin=[&](const Prefix &p){return p.code*power[sampled_depth-p.depth];};
    auto prefix=[&](size_t code,int level) {
        const size_t step=power[sampled_depth-level],first=code*step,last=first+step;
        return Prefix{code,level,cumulative[last]-cumulative[first],
                      cumulative_records[last]-cumulative_records[first]};
    };
    // Express the lane limit in sampled records and leave room for sampling
    // error. An overflowing leaf still splits against the exact arena size.
    const size_t record_target=large ? record_capacity-record_capacity/4 :
        size_t(0.75L*record_capacity*sample.size()/sources);
    // Cost-only splits revisit sources. Use them when the connection volume
    // is large enough to fill at least one arena per worker.
    const bool split_by_cost=large && cumulative_records.back()/record_capacity>=active;
    const size_t work_target=split_by_cost ? cumulative.back()/(8*active) : no_row;
    auto split_priority=[&](const Prefix &p) -> std::pair<bool,size_t> {
        if(p.depth==sampled_depth)return {false,0};
        return {p.records>record_target,p.cost};
    };
    const auto lower_priority=[&](size_t a,size_t b) {
        const auto x=split_priority(plan[a]),y=split_priority(plan[b]);
        return x==y ? a>b : x<y;
    };
    std::vector<size_t> pending;
    pending.reserve(capacity);
    std::priority_queue<size_t,std::vector<size_t>,decltype(lower_priority)> heap(lower_priority,std::move(pending));
    plan.push_back(prefix(0,0));
    heap.push(0);
    while(!heap.empty() && plan.size()+2<=capacity) {
        const size_t index=heap.top();
        const auto [over_capacity,work]=split_priority(plan[index]);
        if(!work || (!over_capacity && plan.size()>=tasks && work<=work_target))break;
        heap.pop();
        const size_t code=plan[index].code*3;
        const int depth=plan[index].depth+1;
        plan[index]=prefix(code,depth);
        plan.push_back(prefix(code+1,depth));
        plan.push_back(prefix(code+2,depth));
        if(depth<sampled_depth) {
            heap.push(index);
            heap.push(plan.size()-2);
            heap.push(plan.size()-1);
        }
    }
    std::sort(plan.begin(),plan.end(),[&](const Prefix &a,const Prefix &b){return begin(a)<begin(b);});
    return plan;
}

struct LeafFull {};
constexpr size_t radix_bins=256;

template<class Visit>
void for_occupied(const Occupation &key,Visit visit) {
    const auto limb_sites=[&](const Occupation::Limb &limb,int offset) {
        for(auto occupied=limb[0];occupied;occupied&=occupied-1) {
            const int bit=__builtin_ctzll(occupied);
            visit(offset+bit,1+int((limb[1]>>bit)&1));
        }
    };
    limb_sites(key.first,0);
    for(size_t i=0;i<key.high.size();++i)limb_sites(key.high[i],64*(i+1));
}

class Scratch {
    struct SourcePath {
        size_t row=no_row;
        Path path;
    };
    const Context &ctx_;
    const Basis &basis_;
    const size_t capacity_,limbs_,tile_,stored_limbs_;
    std::unique_ptr<Link[]> links_;
    std::unique_ptr<Occupation::Limb[]> high_;
    const bool packed_moves_,inline_high_;
    std::unique_ptr<Transfer[]> wide_moves_;
    size_t size_=0;
    Occupation key_,candidate_;
    Config target_;
    int target_open_=0;
    std::vector<signed char> fixed_;
    std::vector<size_t> target_ranks_;
    std::vector<Path> paths_;
    Path source_path_;
    std::vector<SourcePath> source_paths_;
    std::vector<double> amplitudes_;
    std::vector<int> occupied_,open_shells_;
    std::vector<double> exchange_weights_;
    const double *exchange_factors_=nullptr;
    double occupation_energy_=0.;
    Connection connection_,diagonal_;
    bool diagonal_ready_=false,same_occupation_ready_=false;
    bool paths_ready_=false,amplitudes_ready_=false;
    std::vector<double> gathered_,intermediate_;
    std::vector<unsigned char> root_active_;

    const Occupation::Limb *high(const Link &link) const {
        return high_.get()+link.high*stored_limbs_;
    }
    bool same_target(const Link &a,const Link &b) const {
        if(a.low!=b.low)return false;
        if(inline_high_)return a.high==b.high;
        return !stored_limbs_ || std::equal(high(a),high(a)+stored_limbs_,high(b));
    }
    void sort_links() {
        const auto less=[&](const Link &a,const Link &b) {
            if(a.low!=b.low)return a.low<b.low;
            if(inline_high_ && a.high!=b.high)return a.high<b.high;
            for(size_t l=0;l<stored_limbs_;++l)
                if(high(a)[l]!=high(b)[l])return high(a)[l]<high(b)[l];
            return a.source<b.source;
        };
        if(size_>=16384) {
            // Partition by the leading varying byte of the first key limb.
            // Sorting complete keys inside each bin keeps all parents for a
            // target together and preserves source order within that target.
            std::uint64_t variation=0;
            const auto first=links_[0].low[0];
            for(size_t i=1;i<size_;++i)variation|=links_[i].low[0]^first;
            if(variation) {
                const int shift=std::max(0,56-__builtin_clzll(variation));
                const auto bucket=[&](const Link &link){return (link.low[0]>>shift)&255;};
                std::array<size_t,radix_bins+1> offsets{};
                for(size_t i=0;i<size_;++i)++offsets[bucket(links_[i])+1];
                if(*std::max_element(offsets.begin(),offsets.end())<=size_-size_/8) {
                    std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());
                    std::array<size_t,radix_bins> next;
                    std::copy_n(offsets.begin(),radix_bins,next.begin());
                    for(size_t b=0;b<radix_bins;++b) {
                        while(next[b]<offsets[b+1]) {
                            const size_t dest=bucket(links_[next[b]]);
                            if(dest==b)++next[b];
                            else std::swap(links_[next[b]],links_[next[dest]++]);
                        }
                        std::sort(links_.get()+offsets[b],links_.get()+offsets[b+1],less);
                    }
                    return;
                }
            }
        }
        std::sort(links_.get(),links_.get()+size_,less);
    }
    void append(size_t source,const Transfer &move) {
        if(size_==capacity_)throw LeafFull{};
        if(!links_) {
            links_.reset(new Link[capacity_]);
            if(stored_limbs_)high_.reset(new Occupation::Limb[capacity_*stored_limbs_]);
            if(!packed_moves_)wide_moves_.reset(new Transfer[capacity_]);
        }
        std::uint32_t tail=size_;
        if(inline_high_) {
            // The two 16-bit fields follow the lexicographic limb order.
            const auto &limb=candidate_.high[0];
            tail=(std::uint32_t(limb[0])<<16)|std::uint32_t(limb[1]);
        }
        links_[size_]={candidate_.first,source,tail,packed_moves_ ? move.pack() : 0};
        if(!packed_moves_)wide_moves_[size_]=move;
        if(stored_limbs_)std::copy_n(candidate_.high.data(),stored_limbs_,high_.get()+size_*stored_limbs_);
        ++size_;
    }
    void set_target() {
        std::fill(target_.begin(),target_.end(),0);
        for_occupied(key_,[&](int p,int n) {target_[p]=n;});
        diagonal_ready_=same_occupation_ready_=false;
    }
    void make_path(const SpinRanks &spin,const Config &cfg,size_t rank,Path &path) const {
        spin.label(cfg,rank,path.label);
        path.prepare(ctx_.primary.k,ctx_.primary.s,2);
    }
    const Path &source_path(const Primary::Group &group,size_t local,const SpinRanks &spin,Counts &stats) {
        if(source_paths_.empty()) {
            make_path(spin,*group.cfg,group.rank(local),source_path_);
            return source_path_;
        }
        const size_t row=group.row(local);
        auto &entry=source_paths_[row%source_paths_.size()];
        if(entry.row==row)++stats.path_hits;
        else {
            make_path(spin,*group.cfg,group.rank(local),entry.path);
            entry.row=row;
            ++stats.path_misses;
        }
        return entry.path;
    }
    const Connection &same_occupation(Counts &stats) {
        if(!same_occupation_ready_) {
            const auto started=Clock::now();
            guga_diagonal_connection(target_,ctx_.one,ctx_.coulomb_exchange.data(),diagonal_);
            same_occupation_ready_=true;
            stats.diagonal_prepare_ns+=ns(started);
        }
        return diagonal_;
    }
    void prepare_diagonal(Counts &stats) {
        if(diagonal_ready_)return;
        const auto started=Clock::now();
        const auto &factors=ctx_.diagonal_factors[target_open_];
        exchange_factors_=factors.empty() ? nullptr : factors.data();
        exchange_weights_.clear();
        if(exchange_factors_ || target_open_<2) {
            occupied_.clear();
            open_shells_.clear();
            const size_t k=ctx_.primary.k;
            const auto *matrix=ctx_.occupation_terms.data();
            const auto *linear=matrix+k*k;
            occupation_energy_=0.;
            for_occupied(key_,[&](int p,int n) {
                const auto *row=matrix+size_t(p)*k;
                double field=linear[p]+0.5*n*row[p];
                for(int q:occupied_)field+=row[q]*target_[q];
                occupation_energy_+=n*field;
                occupied_.push_back(p);
                if(n==1)open_shells_.push_back(p);
            });
            const auto *exchange=ctx_.coulomb_exchange.data()+k*k;
            for(size_t p=1;p<open_shells_.size();++p) {
                const size_t i=open_shells_[p];
                for(size_t q=0;q<p;++q) {
                    const size_t j=open_shells_[q];
                    exchange_weights_.push_back(0.5*exchange[i*k+j]+0.5*exchange[j*k+i]);
                }
            }
        }
        diagonal_ready_=true;
        stats.diagonal_prepare_ns+=ns(started);
    }
    double reference_diagonal(size_t row,size_t rows,const SpinRanks &spin,Counts &stats) {
        const auto &connection=same_occupation(stats);
        if(connection.generators.empty())return connection.scalar;
        prepare_paths(rows,spin,stats);
        return guga_element(paths_[row],paths_[row],connection);
    }
    double diagonal(size_t row,size_t rows,const SpinRanks &spin,Counts &stats) {
        prepare_diagonal(stats);
        if(exchange_factors_) {
            const size_t columns=exchange_weights_.size();
            const auto *factors=exchange_factors_+target_ranks_[row]*columns;
            // Independent sums let the compiler vectorize the dot product.
            std::array<double,4> partial{};
            size_t column=0;
            for(;column+partial.size()<=columns;column+=partial.size())
                for(size_t lane=0;lane<partial.size();++lane)
                    partial[lane]+=exchange_weights_[column+lane]*factors[column+lane];
            double value=(partial[0]+partial[1])+(partial[2]+partial[3]);
            for(;column<columns;++column)value+=exchange_weights_[column]*factors[column];
            return occupation_energy_+value;
        }
        if(target_open_<2)return occupation_energy_;
        return reference_diagonal(row,rows,spin,stats);
    }
    void start_amplitudes(size_t rows) {
        if(amplitudes_ready_)return;
        std::fill_n(amplitudes_.data(),rows*ctx_.roots,0.);
        amplitudes_ready_=true;
    }
    void prepare_paths(size_t rows,const SpinRanks &spin,Counts &stats) {
        if(paths_ready_)return;
        for(size_t row=0;row<rows;++row)make_path(spin,target_,target_ranks_[row],paths_[row]);
        stats.target_paths+=rows;
        paths_ready_=true;
    }
    static int open_after(const Config &cfg,int open,int create,int destroy) {
        return open+(cfg[create]==0 ? 1 : -1)+(cfg[destroy]==2 ? 1 : -1);
    }
    static bool allowed(const Config &cfg,int create,int destroy) {
        return cfg[create]<2 && cfg[destroy]>0;
    }
    const Matrix &factor(const Primary::Group &group,int p,int q,int r=-1,int s=-1) const {
        const auto &key=group.block->key;
        const auto &cfg=*group.cfg;
        if(!limbs_) {
            auto [occupied,doubled]=key.first;
            if(r>=0) {
                const auto add=std::uint64_t(1)<<r,remove=std::uint64_t(1)<<s;
                occupied^=remove&~doubled;
                doubled=(doubled&~remove)|(add&occupied);
                occupied|=add;
            }
            return ctx_.bank->at(ctx_.bank->index(occupied,doubled,p,q));
        }
        int np=cfg[p],nq=cfg[q],open=group.block->open;
        int before_p=key.singles_before(p),before_q=key.singles_before(q);
        if(r>=0) {
            const int add=cfg[r]==0 ? 1 : -1,remove=cfg[s]==2 ? 1 : -1;
            np+=(p==r)-(p==s);
            nq+=(q==r)-(q==s);
            open+=add+remove;
            before_p+=(r<p ? add : 0)+(s<p ? remove : 0);
            before_q+=(r<q ? add : 0)+(s<q ? remove : 0);
        }
        return ctx_.bank->get(open,np,nq,before_p+(q<p && nq!=1),before_q+(p<q && np!=1));
    }
    template<bool Tiled>
    void apply_factor(const Matrix &factor,const double *input,double *output,size_t rows,double scale) {
        if(ctx_.roots==1) {
            for(size_t row=0;row<rows;++row) {
                const size_t rank=Tiled ? target_ranks_[row] : row;
                double value=0.;
                if(factor.cols<=4) {
                    const double *values=factor.values.data()+rank*factor.cols;
                    for(size_t i=0;i<factor.cols;++i)value+=values[i]*input[i];
                } else {
                    for(size_t j=factor.ptr[rank];j<factor.ptr[rank+1];++j)
                        value+=factor.data[j]*input[factor.indices[j]];
                }
                if constexpr(Tiled)output[row]+=scale*value;
                else output[row]=value;
            }
            return;
        }
        for(size_t row=0;row<rows;++row) {
            const size_t rank=Tiled ? target_ranks_[row] : row;
            const auto first=factor.ptr[rank],end=factor.ptr[rank+1];
            for(size_t root=0;root<ctx_.roots;++root) {
                if constexpr(Tiled)if(!root_active_.empty() && !root_active_[root])continue;
                double value=0.;
                for(size_t j=first;j<end;++j)
                    value+=factor.data[j]*input[factor.indices[j]*ctx_.roots+root];
                auto &slot=output[row*ctx_.roots+root];
                if constexpr(Tiled)slot+=scale*value;
                else slot=value;
            }
        }
    }
    bool contract_block(const Primary::Group &group,size_t rows,Counts &stats) {
        const int open=group.block->open;
        if(open>ctx_.maximum_factor_input || target_open_>ctx_.maximum_block_open)return false;
        const auto &cfg=*group.cfg;
        // Check the whole connection before touching its target tile. Any
        // unsupported spin class sends all its terms through the scalar path.
        for(const auto &term:connection_.generators)
            if(term.two_body && allowed(cfg,term.r,term.s) &&
               open_after(cfg,open,term.r,term.s)>ctx_.maximum_factor_input)return false;
        const double *input=ctx_.coefficients+group.block->offset*ctx_.roots;
        if(!group.block->complete()) {
            // The reduced matrices use full spin ranks, with zero input on
            // ranks absent from the variational source block.
            std::fill_n(gathered_.data(),basis_.spin(open).size()*ctx_.roots,0.);
            for(size_t local=0;local<group.size();++local)
                std::copy_n(ctx_.coefficients+group.row(local)*ctx_.roots,ctx_.roots,
                            gathered_.data()+group.rank(local)*ctx_.roots);
            input=gathered_.data();
        }
        start_amplitudes(rows);
        for(const auto &term:connection_.generators) {
            if(!term.two_body) {
                const auto &matrix=factor(group,term.p,term.q);
                apply_factor<true>(matrix,input,amplitudes_.data(),rows,term.coefficient);
                continue;
            }
            if(allowed(cfg,term.r,term.s) && open_after(cfg,open,term.r,term.s)>=ctx_.primary.s) {
                const auto &right=factor(group,term.r,term.s);
                const auto &left=factor(group,term.p,term.q,term.r,term.s);
                apply_factor<false>(right,input,intermediate_.data(),right.rows,1.);
                apply_factor<true>(left,intermediate_.data(),amplitudes_.data(),rows,term.coefficient);
            }
            // The stored two-electron terms are normal ordered.
            if(term.q==term.r) {
                const auto &matrix=factor(group,term.p,term.s);
                apply_factor<true>(matrix,input,amplitudes_.data(),rows,-term.coefficient);
            }
        }
        ++stats.block_contractions;
        return true;
    }
    void contract(size_t source,const Transfer &move,size_t rows,const SpinRanks &target_spin,Counts &stats) {
        const auto &group=ctx_.primary.groups[source];
        const auto &cfg=*group.cfg;
        const auto &spin=basis_.spin(group.block->open);
        if(!root_active_.empty()) {
            const double upper=move.order ? ctx_.connection_bound(cfg,group.block->key,move) : 0.;
            for(size_t r=0;r<ctx_.roots;++r)root_active_[r]=!move.order ||
                !screened(upper,ctx_.root_weights[source*ctx_.roots+r],ctx_.eps,ctx_.primary.k);
        }
        const bool bound=!ctx_.block_screening && ctx_.eps>0. && move.order && (group.size()>1 || rows>1);
        if(move.order) {
            ranked_connection(PairDescriptor(target_,cfg,PairDescriptor::no_id,source,move.holes,
                                             move.particles,move.order),ctx_.one,ctx_.eri,bound,connection_);
            if(connection_.generators.empty())return;
            if(bound && screened(connection_.term_bound,ctx_.weights[source],ctx_.eps,ctx_.primary.k))return;
            if(ctx_.bank && contract_block(group,rows,stats))return;
        }
        const auto &connection=move.order ? connection_ : same_occupation(stats);
        for(size_t local=0;local<group.size();++local) {
            const auto *c=ctx_.coefficients+group.row(local)*ctx_.roots;
            double maximum=0.;
            for(size_t r=0;r<ctx_.roots;++r)maximum=std::max(maximum,std::abs(c[r]));
            if(!maximum || (bound && screened(connection.term_bound,maximum,ctx_.eps,ctx_.primary.k)))continue;
            start_amplitudes(rows);
            prepare_paths(rows,target_spin,stats);
            const auto &path=source_path(group,local,spin,stats);
            for(size_t row=0;row<rows;++row) {
                const double element=guga_element(paths_[row],path,connection);
                ++stats.pair_elements;
                if(!std::isfinite(element))throw std::runtime_error("nonfinite ENPT2 coupling");
                for(size_t r=0;r<ctx_.roots;++r) {
                    const double product=element*c[r];
                    if(ctx_.block_screening ? (root_active_.empty() || root_active_[r]) : std::abs(product)>ctx_.eps)
                        amplitudes_[row*ctx_.roots+r]+=product;
                }
            }
        }
    }
    void sum(size_t rows,const SpinRanks &spin,Root *roots,Counts &stats) {
        const auto started=Clock::now();
        const size_t preparation_at_start=stats.diagonal_prepare_ns;
        for(size_t row=0;row<rows;++row) {
            const auto *t=amplitudes_.data()+row*ctx_.roots;
            if(std::all_of(t,t+ctx_.roots,[](double value){return value==0.;}))continue;
            const auto &path=paths_[row];
            const double h=diagonal(row,rows,spin,stats);
            ++stats.diagonals;
            for(size_t r=0;r<ctx_.roots;++r) {
                if(t[r]==0.)continue;
                double d=ctx_.energies[r]-h;
                // The specialized sum rearranges floating-point terms. Use
                // the reference expression at the denominator guard itself.
                if(std::abs(d)<=ctx_.diagonal_guard && (exchange_factors_ || target_open_<2))
                    d=ctx_.energies[r]-reference_diagonal(row,rows,spin,stats);
                if(!std::isfinite(t[r]) || !std::isfinite(d))
                    throw std::runtime_error("nonfinite ENPT2 amplitude or denominator");
                if(std::abs(d)<=ctx_.tolerance) {
                    prepare_paths(rows,spin,stats);
                    std::ostringstream msg;
                    msg.precision(17);
                    msg<<"ENPT2 denominator within tolerance for root "<<r<<", CSF "<<path.label
                       <<": numerator="<<t[r]<<", denominator="<<d<<" Eh";
                    throw std::domain_error(msg.str());
                }
                const double term=(t[r]/d)*t[r];
                if(!std::isfinite(term))throw std::runtime_error("nonfinite ENPT2 energy term");
                auto &root=roots[r];
                root.energy.add(term);
                root.minimum=std::min(root.minimum,std::abs(d));
                root.positive+=d>0.;
                ++root.contributing;
            }
        }
        stats.diagonal_apply_ns+=ns(started)-(stats.diagonal_prepare_ns-preparation_at_start);
    }
    template<class Parents>
    void evaluate(Root *roots,Counts &stats,Parents parents) {
        const size_t id=basis_.find(key_);
        const auto *internal=id==no_row ? nullptr : &basis_.blocks()[id];
        target_open_=key_.singles();
        const auto &spin=ctx_.spin[target_open_];
        if(!spin || !spin->size() || (internal && internal->complete()))return;
        ++stats.external_occupations;
        set_target();
        size_t rank=0;
        while(rank<spin->size()) {
            size_t rows=0;
            while(rows<tile_ && rank<spin->size()) {
                if(!internal || !internal->contains(rank))target_ranks_[rows++]=rank;
                ++rank;
            }
            if(!rows)break;
            stats.external_ranks+=rows;
            paths_ready_=amplitudes_ready_=false;
            parents([&](size_t source,const Transfer &move) {contract(source,move,rows,*spin,stats);});
            if(amplitudes_ready_)sum(rows,*spin,roots,stats);
        }
    }
    void singleton(const unsigned char *digits,Root *roots,Counts &stats) {
        for(int p=0;p<ctx_.primary.k;++p)target_[ctx_.sites[p]]=digits[p];
        key_=Occupation(target_);
        ++stats.singletons;
        const auto started=Clock::now();
        bool first_tile=true;
        evaluate(roots,stats,[&](auto visit) {
            transfers(target_,basis_.symmetry(),[&](const auto &holes,const auto &particles,int order) {
                ++stats.visits;
                candidate_=key_;
                candidate_.transfer(holes,particles,order);
                const auto id=basis_.find(candidate_);
                if(id==no_row || ctx_.weights[id]==0.)return;
                const Transfer move{particles,holes,order};
                if(ctx_.reject(*ctx_.primary.groups[id].cfg,candidate_,move,ctx_.weights[id])) {
                    ++stats.screened;
                    return;
                }
                if(first_tile)++stats.connections;
                visit(id,move);
            });
            first_tile=false;
        });
        stats.contract_ns+=ns(started);
    }
public:
    Scratch(const Context &ctx,size_t capacity,size_t limbs,size_t tile,size_t cached_paths,size_t block_rows,
            size_t occupied_sites,size_t exchange_terms)
        :ctx_(ctx),basis_(*ctx.primary.native),capacity_(capacity),limbs_(limbs),tile_(tile),
         stored_limbs_(Link::inline_high(ctx.primary.k) ? 0 : limbs),
         packed_moves_(ctx.primary.k<=Transfer::packed_orbitals),inline_high_(Link::inline_high(ctx.primary.k)),
         target_(ctx.primary.k),fixed_(ctx.primary.k),target_ranks_(tile),paths_(tile),source_paths_(cached_paths),
         amplitudes_(tile*ctx.roots),gathered_(block_rows*ctx.roots),intermediate_(block_rows*ctx.roots),
         root_active_(ctx.root_weights.empty() ? 0 : ctx.roots) {
        key_.high.resize(limbs);
        candidate_.high.resize(limbs);
        occupied_.reserve(occupied_sites);
        open_shells_.reserve(occupied_sites);
        exchange_weights_.reserve(exchange_terms);
    }
    bool process(const unsigned char *digits,int depth,Root *roots,Counts &stats) {
        size_=0;
        std::fill(fixed_.begin(),fixed_.end(),-1);
        for(int i=0;i<depth;++i)fixed_[ctx_.sites[i]]=digits[i];
        auto started=Clock::now();
        try {
            const auto generate=[&](size_t id) {
                ++stats.prefix_source_checks;
                const double maximum=ctx_.weights[id];
                if(!maximum)return;
                const auto &group=ctx_.primary.groups[id];
                const auto &source_key=group.block->key;
                const auto &cfg=*group.cfg;
                constrained_transfers(cfg,basis_.symmetry(),fixed_.data(),ctx_.sites,depth,
                    ctx_.ordered,maximum,ctx_.eps,
                    [&](const Transfer &move,bool bounded) {
                        ++stats.visits;
                        if(!move.order && group.block->complete())return;
                        if((!bounded || (ctx_.block_screening && move.order==1)) &&
                           ctx_.reject(cfg,source_key,move,maximum)) {
                            ++stats.screened;
                            return;
                        }
                        candidate_=source_key;
                        candidate_.transfer(move.holes,move.particles,move.order);
                        const auto &spin=ctx_.spin[candidate_.singles()];
                        if(spin && spin->size())append(id,move);
                    });
            };
            if(ctx_.prefix_sources)ctx_.prefix_sources->compatible(digits,depth,generate);
            else for(size_t id=0;id<ctx_.weights.size();++id)generate(id);
        } catch(const LeafFull &) {
            stats.generate_ns+=ns(started);
            if(depth!=ctx_.primary.k)return false;
            singleton(digits,roots,stats);
            ++stats.leaves;
            return true;
        }
        stats.generate_ns+=ns(started);
        ++stats.leaves;
        if(!size_)return true;
        stats.connections+=size_;
        stats.largest_leaf=std::max(stats.largest_leaf,size_);
        started=Clock::now();
        sort_links();
        stats.sort_ns+=ns(started);
        started=Clock::now();
        for(size_t begin=0;begin<size_;) {
            size_t end=begin+1;
            while(end<size_ && same_target(links_[begin],links_[end]))++end;
            key_.first=links_[begin].low;
            if(inline_high_) {
                const auto tail=links_[begin].high;
                key_.high[0]={tail>>16,tail&0xffffu};
            } else if(stored_limbs_)std::copy_n(high(links_[begin]),stored_limbs_,key_.high.data());
            evaluate(roots,stats,[&](auto visit) {
                for(size_t i=begin;i<end;++i) {
                    const auto &link=links_[i];
                    visit(link.source,packed_moves_ ? Transfer::unpack(link.move) : wide_moves_[link.high]);
                }
            });
            begin=end;
        }
        stats.contract_ns+=ns(started);
        return true;
    }
};

} // namespace pt2

struct ENPT2 {
    std::vector<pt2::Root> roots;
    Stats stats;
};

inline ENPT2 enpt2(const Primary &primary,Workers &workers,const double *one,const SpatialERI &eri,
    const double *coefficients,const double *energies,size_t roots,double offset,double eps,double tolerance,
    HCITransferBounds *bounds,size_t budget,size_t staging_bytes,bool block_screening) {
    using namespace pt2;
    const auto started=Clock::now();
    const size_t k=primary.k,groups=primary.groups.size(),limbs=k ? (k-1)/64 : 0;
    const auto &basis=*primary.native;
    const size_t open=basis.maximum_open()+1;
    const size_t matrix_bytes=bytes_for(k,bytes_for(k,sizeof(double)));
    size_t fixed=checked_add(staging_bytes,checked_add(bytes_for(3,matrix_bytes),bytes_for(k,sizeof(double))));
    fixed=checked_add(fixed,checked_add(bytes_for(groups,sizeof(double)),
        bytes_for(k,sizeof(int)+sizeof(std::array<size_t,3>))));
    fixed=checked_add(fixed,bytes_for(open,sizeof(std::unique_ptr<SpinRanks>)));
    fixed=checked_add(fixed,bytes_for(open,sizeof(std::vector<double>)));
    fixed=checked_add(fixed,bytes_for(roots,sizeof(Root)+sizeof(long double)+sizeof(double)));
    const size_t root_weight_count=block_screening && eps>0. && roots>1 ? bytes_for(groups,roots) : 0;
    fixed=checked_add(fixed,bytes_for(root_weight_count,sizeof(double)));
    memory_left(budget,fixed,"ENPT2 shared workspace");
    std::vector<double> weights(groups);
    std::vector<double> root_weights(root_weight_count);
    std::vector<std::unique_ptr<SpinRanks>> spin(open);
    size_t sources=0;
    {
        std::vector<long double> norms(roots);
        for(size_t id=0;id<groups;++id) {
            const auto &group=primary.groups[id];
            for(size_t row=group.block->offset;row<group.block->offset+group.size();++row) {
                for(size_t r=0;r<roots;++r) {
                    const double value=coefficients[row*roots+r];
                    if(!std::isfinite(value))throw std::invalid_argument("ENPT2 coefficients must be finite");
                    norms[r]+=static_cast<long double>(value)*value;
                    weights[id]=std::max(weights[id],std::abs(value));
                    if(root_weight_count)root_weights[id*roots+r]=std::max(root_weights[id*roots+r],std::abs(value));
                }
            }
            if(!weights[id])continue;
            ++sources;
            const int m=group.block->open;
            for(int n=std::max(primary.s,m-4);n<=std::min(basis.maximum_open(),m+4);n+=2) {
                if(spin[n])continue;
                fixed=checked_add(fixed,checked_add(sizeof(SpinRanks),bytes_for(n+1,bytes_for(n+2,sizeof(size_t)))));
                memory_left(budget,fixed,"ENPT2 spin ranks");
                spin[n]=std::make_unique<SpinRanks>(n,primary.s);
            }
        }
        for(auto norm:norms)
            if(std::abs(norm-1.L)>1.e-8L)throw std::invalid_argument("ENPT2 eigenvectors must be normalized");
    }
    std::vector<std::array<size_t,3>> occupations(k);
    for(size_t id=0;id<groups;++id)if(weights[id]) {
        const auto &cfg=*primary.groups[id].cfg;
        for(size_t p=0;p<k;++p)++occupations[p][cfg[p]];
    }
    std::vector<int> sites(k);
    std::iota(sites.begin(),sites.end(),0);
    auto most_common=[&](int p) {return *std::max_element(occupations[p].begin(),occupations[p].end());};
    // Use the most variable source sites first to avoid concentrating all
    // work in a single prefix of near-empty or near-doubly-occupied sites.
    std::sort(sites.begin(),sites.end(),[&](int p,int q) {
        const size_t a=most_common(p),b=most_common(q);
        return a==b ? p>q : a<b;
    });
    // Track per-lane payloads, including temporary capacity during generator
    // vector growth. A rank-zero connection retains at most one exchange
    // generator per pair of singly occupied orbitals.
    const size_t electrons=std::min(k,size_t(primary.n));
    const size_t open_max=std::min(size_t(primary.maximum_selected_open())+4,open-1);
    const size_t diagonal_terms=std::max(size_t(128),triangle_size(open_max));
    const size_t other_terms=std::max(size_t(128),2*electrons+1);
    const size_t path_bytes=checked_add(sizeof(Path),bytes_for(k+1,2+3*sizeof(int)));
    size_t lane_bytes=checked_add(sizeof(Scratch),checked_add(bytes_for(k,2+sizeof(int)),
        bytes_for(limbs,3*sizeof(Occupation::Limb))));
    lane_bytes=checked_add(lane_bytes,bytes_for(checked_add(diagonal_terms,other_terms),3*sizeof(GugaTerm)));
    lane_bytes=checked_add(lane_bytes,checked_add(bytes_for(diagonal_terms,sizeof(double)),
                                                bytes_for(electrons,2*sizeof(int))));
    lane_bytes=checked_add(lane_bytes,path_bytes);
    lane_bytes=checked_add(lane_bytes,(2*radix_bins+1)*sizeof(size_t));
    if(root_weight_count)lane_bytes=checked_add(lane_bytes,roots);
    const size_t rank_bytes=checked_add(path_bytes+sizeof(size_t),bytes_for(roots,sizeof(double)));
    const size_t move_bytes=k<=Transfer::packed_orbitals ? 0 : sizeof(Transfer);
    const size_t stored_limbs=Link::inline_high(k) ? 0 : limbs;
    const size_t record_bytes=checked_add(sizeof(Link)+move_bytes,bytes_for(stored_limbs,sizeof(Occupation::Limb)));
    const size_t task_bytes=checked_add(sizeof(Counts)+sizeof(Prefix)+sizeof(size_t),bytes_for(roots,sizeof(Root)));
    const size_t minimum_lane=checked_add(checked_add(lane_bytes,rank_bytes),checked_add(record_bytes,task_bytes));
    const size_t desired=std::min(workers.count(),std::max(size_t(1),sources/16));
    size_t available=memory_left(budget,fixed,"ENPT2 shared workspace");
    memory_left(available,minimum_lane,"ENPT2 minimum worker workspace");
    const size_t minimum_workspace=checked_add(fixed,minimum_lane);
    const size_t table_cap=std::min(size_t(16)*1024*1024,(available-minimum_lane)/32);
    std::vector<std::vector<double>> diagonal_factors(open);
    size_t factor_bytes=0;
    for(size_t m=2;m<open;++m) {
        if(!spin[m] || !spin[m]->size())continue;
        const size_t columns=triangle_size(m-1);
        const size_t row_bytes=columns*sizeof(double);
        if(spin[m]->size()>(table_cap-factor_bytes)/row_bytes)continue;
        auto &factors=diagonal_factors[m];
        factors.resize(spin[m]->size()*columns);
        factor_bytes+=factors.size()*sizeof(double);
        const Config singles(m,1);
        // In a diagonal exchange, the two generator intervals coincide.
        // Empty sites act as identities and the signs of closed sites
        // cancel. Thus only the relative positions of the open shells matter.
        workers.run(spin[m]->size(),[&](size_t rank,size_t) {
            const Path path(spin[m]->label(singles,rank),m,primary.s,2);
            for(size_t p=1;p<m;++p)
                for(size_t q=0;q<p;++q)
                    factors[rank*columns+p*(p-1)/2+q]=0.5+guga_two_body(path,path,q,p,p,q);
        },16,std::min(desired,(available-factor_bytes)/minimum_lane));
    }
    fixed+=factor_bytes;
    available-=factor_bytes;
    const bool large_census=sources>=large_census_sources;
    int sampled_depth=std::min(primary.k,large_census ? census_depth : census_depth-1);
    size_t census_buckets=1;
    for(int i=0;i<sampled_depth;++i)census_buckets*=3;
    const size_t sample_bytes=large_census ? bytes_for(8192+32*open,sizeof(CensusSource))+bytes_for(open,sizeof(CensusStratum)) :
        bytes_for(1536+4*desired,sizeof(CensusSource))+sizeof(CensusStratum);
    const auto census_reservation=[&](size_t buckets) {
        return checked_add(bytes_for(buckets,2*(sizeof(std::atomic<size_t>)+sizeof(size_t))),
            checked_add(sample_bytes,bytes_for(k+2,sizeof(size_t))));
    };
    size_t census_bytes=census_reservation(census_buckets);
    if(sampled_depth==census_depth && census_bytes>=available/32) {
        --sampled_depth;
        census_bytes=census_reservation(census_buckets/3);
    }
    const bool adaptive=desired>=4 && sources>=256 && census_bytes<available/32
        && census_bytes<=available-minimum_lane;
    if(adaptive) {
        fixed=checked_add(fixed,census_bytes);
        available-=census_bytes;
    }
    const auto bounds_started=Clock::now();
    Ordered ordered{bounds};
    size_t new_table_bytes=0;
    if(bounds && eps>0.) {
        const size_t base=bounds->bytes();
        const size_t lanes=std::min(desired,available/minimum_lane);
        // Prepare screening tables before sizing the record arenas. When
        // all lanes are active, sizing the arenas first can consume every
        // spare byte and force direct generation even for small tables.
        const size_t limit=checked_add(base,(available-lanes*minimum_lane)/8);
        const auto prepare=[](auto action) {
            try {
                return action();
            } catch(const BudgetError &) {
                return false;
            }
        };
        prepare([&]{bounds->prepare_transfer_bounds(primary.k,limit);return true;});
        ordered.doubles=prepare([&]{return bounds->prepare_ordered_transfers(primary.k,limit,workers);});
        ordered.singles=prepare([&]{return bounds->prepare_ordered_singles(primary.k,primary.n,limit);});
        new_table_bytes=bounds->bytes()-base;
        available-=new_table_bytes;
    }
    const size_t bounds_ns=ns(bounds_started);
    const auto block_started=Clock::now();
    std::unique_ptr<OneBank> bank;
    int maximum_block_open=0,maximum_factor_input=0;
    size_t block_rows=0,bank_bytes=0;
    if(block_screening) {
        const size_t lanes=std::min(desired,available/minimum_lane);
        const size_t spare=available-lanes*minimum_lane;
        const int regular_open=12+primary.s%2;
        const int max_output=regular_open+(large_census ? 4 : sources>=4096 ? 2 : 0);
        const int max_open=std::min({int(electrons),int(open_max),max_output});
        for(int m=max_open-(max_open-primary.s)%2;m>=primary.s;m-=2) {
            // The largest class only needs to be a destination. Extending
            // the sparse bank avoids a pairwise scalar loop over that class.
            const int max_input=std::min(m,std::max(regular_open,m-2));
            const size_t cap=std::min(size_t(m>regular_open+2 ? 384 : 128)*1024*1024,spare/8);
            if(cap<=sizeof(OneBank))break;
            const size_t rows=SpinRanks(m,primary.s).size();
            const size_t worker_bytes=bytes_for(rows,bytes_for(roots,2*sizeof(double)));
            if(worker_bytes>spare/lanes/2)continue;
            try {
                bank=std::make_unique<OneBank>(m,primary.s,cap-sizeof(OneBank),false,m<primary.n,max_input);
            } catch(const BudgetError &) {continue;}
            bank_bytes=sizeof(OneBank)+bank->bytes();
            block_rows=rows;
            maximum_block_open=m;
            maximum_factor_input=max_input;
            fixed+=bank_bytes;
            available-=bank_bytes;
            lane_bytes+=worker_bytes;
            break;
        }
    }
    const size_t block_setup_ns=ns(block_started);
    const size_t minimum_reserved_lane=checked_add(lane_bytes,checked_add(rank_bytes,
                                                 checked_add(record_bytes,task_bytes)));
    const auto source_index_started=Clock::now();
    std::unique_ptr<PrefixSources> prefix_sources;
    size_t source_index_bytes=0;
    if(adaptive && sources>=4096) {
        const int index_depth=std::min(primary.k,8);
        const size_t reservation=PrefixSources::reservation(index_depth,sources);
        const size_t lanes=std::min(desired,available/minimum_reserved_lane);
        if(reservation<=(available-lanes*minimum_reserved_lane)/16) {
            prefix_sources=std::make_unique<PrefixSources>(index_depth,primary,weights,sites,sources);
            source_index_bytes=reservation;
            fixed+=reservation;
            available-=reservation;
        }
    }
    const size_t source_index_ns=ns(source_index_started);
    const size_t active=std::min(desired,available/minimum_reserved_lane);
    const size_t to_schedule=std::min(4*active,std::max(size_t(1),sources/16));
    int depth=0;
    size_t tasks=1;
    // Keep the initial prefix tasks and their reductions bounded. A lane
    // splits a heavy prefix depth-first without adding scheduler metadata.
    while(tasks<to_schedule && size_t(depth)<k &&
          bytes_for(3*tasks,task_bytes)<=available-active*(minimum_reserved_lane-task_bytes)) {
        ++depth;
        tasks*=3;
    }
    size_t task_capacity=tasks;
    if(adaptive && sources>=4096) {
        const size_t task_space=available-active*(minimum_reserved_lane-task_bytes);
        task_capacity=std::min((large_census ? 64 : 16)*tasks,task_space/task_bytes);
    }
    const size_t partitions=bytes_for(task_capacity,task_bytes);
    const size_t lane_goal=std::max(size_t(64)*1024*1024,minimum_reserved_lane-task_bytes);
    const size_t lane_limit=std::min(lane_goal,(available-partitions)/active);
    const size_t cache_entry=path_bytes+sizeof(size_t);
    size_t cached_paths=std::min(size_t(256),(lane_limit-(minimum_reserved_lane-task_bytes))/32/cache_entry);
    if(cached_paths<16)cached_paths=0;
    const size_t cache_bytes=cached_paths*cache_entry;
    const size_t tile=std::min(std::max(size_t(16),block_rows),
                              (lane_limit-lane_bytes-cache_bytes-record_bytes)/rank_bytes);
    const size_t room=std::min((lane_limit-lane_bytes-cache_bytes-tile*rank_bytes)/record_bytes,
                              size_t(std::numeric_limits<std::uint32_t>::max()));
    // For small spaces reserve at most the combinatorial upper bound.
    const size_t holes=triangle_size(electrons);
    const size_t particles=triangle_size(k);
    const size_t per_source=checked_add(1,checked_add(bytes_for(electrons,k),bytes_for(holes,particles)));
    const size_t slots=sources>room/per_source ? room : sources*per_source;
    const size_t scratch=checked_add(lane_bytes,checked_add(cache_bytes,checked_add(tile*rank_bytes,slots*record_bytes)));
    const size_t workspace=checked_add(fixed,checked_add(partitions,active*scratch));
    const size_t peak=checked_add(workspace,new_table_bytes);
    const auto tables=coulomb_exchange_tables(eri,workers);
    // The occupation-only part is h' n + n^T M n / 2. For distinct
    // orbitals M=J-K/2. Its exchange contribution for two open shells is
    // restored by the +1/2 in the angular factors above.
    std::vector<double> occupation_terms(k*k+k);
    double largest_one=0.,largest_two=0.;
    for(double value:tables)largest_two=std::max(largest_two,std::abs(value));
    for(size_t p=0;p<k;++p) {
        const size_t diagonal=p*(k+1);
        largest_one=std::max(largest_one,std::abs(one[diagonal]));
        occupation_terms[k*k+p]=one[diagonal]-0.5*tables[diagonal];
        occupation_terms[diagonal]=tables[diagonal];
        for(size_t q=0;q<p;++q) {
            const size_t pq=p*k+q,qp=q*k+p;
            occupation_terms[pq]=occupation_terms[qp]=0.5*tables[pq]+0.5*tables[qp]
                -0.25*tables[k*k+pq]-0.25*tables[k*k+qp];
        }
    }
    const double n=primary.n;
    const double diagonal_guard=tolerance+32*std::numeric_limits<double>::epsilon()*(n+4)*
        (1+n*(largest_one+n*largest_two));
    std::vector<double> electronic(roots);
    for(size_t r=0;r<roots;++r)electronic[r]=energies[r]-offset;
    const Context ctx{primary,one,eri,coefficients,electronic.data(),roots,eps,tolerance,diagonal_guard,
                      ordered,weights,root_weights,tables,
                      occupation_terms,spin,sites,
                      diagonal_factors,bank.get(),prefix_sources.get(),maximum_block_open,maximum_factor_input,
                      block_screening};
    const auto planning_started=Clock::now();
    const auto plan=plan_prefixes(ctx,workers,active,depth,tasks,task_capacity,sources,
                                  adaptive ? sampled_depth : 0,slots);
    std::vector<size_t> dispatch(plan.size());
    std::iota(dispatch.begin(),dispatch.end(),0);
    std::sort(dispatch.begin(),dispatch.end(),[&](size_t a,size_t b) {
        return plan[a].cost==plan[b].cost ? a<b : plan[a].cost>plan[b].cost;
    });
    const size_t planning_ns=ns(planning_started);
    ENPT2 result{std::vector<Root>(roots),{}};
    std::vector<Root> partial(plan.size()*roots);
    std::vector<Counts> counts(plan.size());
    // Use separate lanes to retain one work buffer per participating worker.
    // The prefix number fixes the reduction order even with dynamic dispatch.
    std::atomic<size_t> next{0};
    workers.each_worker([&](size_t) {
        Scratch scratchpad(ctx,slots,limbs,tile,cached_paths,block_rows,electrons,diagonal_terms);
        std::vector<unsigned char> digits(k);
        for(;;) {
            const size_t item=next.fetch_add(1,std::memory_order_relaxed);
            if(item>=plan.size())break;
            const size_t task=dispatch[item];
            const int initial=plan[task].depth;
            size_t code=plan[task].code;
            for(int p=initial;p>0;--p) {digits[p-1]=code%3;code/=3;}
            int current=initial;
            auto &stats=counts[task];
            for(;;) {
                if(!scratchpad.process(digits.data(),current,partial.data()+task*roots,stats)) {
                    ++stats.splits;
                    digits[current++]=0;
                    continue;
                }
                while(current>initial && digits[current-1]==2)--current;
                if(current==initial)break;
                ++digits[current-1];
            }
        }
    },active);
    Counts totals;
    size_t largest_task=0;
    for(size_t task=0;task<plan.size();++task) {
        totals.merge(counts[task]);
        largest_task=std::max(largest_task,counts[task].contract_ns+counts[task].generate_ns+counts[task].sort_ns);
        for(size_t r=0;r<roots;++r)result.roots[r].merge(partial[task*roots+r]);
    }
    for(const auto &root:result.roots)
        if(!std::isfinite(root.energy.total()))throw std::runtime_error("nonfinite ENPT2 total correction");
    result.stats={{"source_configurations",sources},{"active_workers",active},{"initial_tasks",plan.size()},
        {"batches",totals.leaves},{"batch_splits",totals.splits},{"streamed_targets",totals.singletons},
        {"transfer_visits",totals.visits},{"screened_transfers",totals.screened},
        {"prefix_source_checks",totals.prefix_source_checks},{"source_index_bytes",source_index_bytes},
        {"source_index_ns",source_index_ns},
        {"connection_records",totals.connections},{"largest_batch_records",totals.largest_leaf},
        {"external_occupations",totals.external_occupations},{"external_ranks",totals.external_ranks},
        {"matrix_elements",totals.pair_elements},{"denominators",totals.diagonals},
        {"block_contractions",totals.block_contractions},{"block_factor_bytes",bank_bytes},{"block_setup_ns",block_setup_ns},
        {"maximum_block_open",maximum_block_open},
        {"maximum_factor_input_open",maximum_factor_input},
        {"bounds_ns",bounds_ns},{"planning_ns",planning_ns},{"adaptive_prefixes",adaptive},
        {"diagonal_factor_bytes",factor_bytes},
        {"ordered_doubles",ordered.doubles},{"ordered_singles",ordered.singles},
        {"generation_worker_ns",totals.generate_ns},{"sort_worker_ns",totals.sort_ns},
        {"contraction_worker_ns",totals.contract_ns},
        {"denominator_worker_ns",totals.diagonal_prepare_ns+totals.diagonal_apply_ns},
        {"diagonal_prepare_worker_ns",totals.diagonal_prepare_ns},
        {"diagonal_apply_worker_ns",totals.diagonal_apply_ns},
        {"largest_task_ns",largest_task},{"enpt2_ns",ns(started)},{"workspace_bytes",peak},
        {"minimum_workspace_bytes",minimum_workspace},{"records_per_worker",slots},
        {"rank_tile_rows",tile},{"target_paths_built",totals.target_paths},
        {"source_path_hits",totals.path_hits},{"source_path_misses",totals.path_misses}};
    return result;
}
} // namespace scsf::structured

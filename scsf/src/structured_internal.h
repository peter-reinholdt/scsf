/* SCSF; GPL-3.0-or-later. Shared exact operator-link infrastructure. */
#pragma once
#include "structured_spin.h"
#include "parallel.h"
#include "pair_descriptor.h"
#include "basis.h"
#include <chrono>
#include <set>
#include <mutex>
#include <unordered_map>

namespace scsf { namespace structured {
using Clock=std::chrono::steady_clock;
inline size_t ns(Clock::time_point start) { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count(); }
constexpr size_t none=std::numeric_limits<size_t>::max();

inline std::string pack(const Config &c) {
    std::string key((c.size()+3)/4,'\0');
    for(size_t p=0;p<c.size();++p)key[p/4]=unsigned(char(key[p/4]))|(c[p]<<(2*(p%4)));
    return key;
}
inline Config unpack(const std::string &key,int k) {
    Config c(k);for(int p=0;p<k;++p)c[p]=(unsigned(char(key[p/4]))>>(2*(p%4)))&3;return c;
}
inline std::string spin_label(const Config &c,const std::string &path) {
    std::string label;size_t n=0;for(int v:c)label+=v==1 ? path[n++] : v==2 ? '2' : '0';return label;
}
inline unsigned schedule_index(size_t value) {
    if(value>std::numeric_limits<unsigned>::max())throw BudgetError("removal schedule index exceeds encoding");
    return static_cast<unsigned>(value);
}
inline size_t pair_index(size_t p,size_t q,size_t modes) { return p*(2*modes-p-1)/2+q-p-1; }

class Workers {
    size_t count_;
    ThreadTeam team_;
public:
    Workers(size_t count,std::vector<int> cpus):count_(count),team_(count,std::move(cpus)){}
    template<class F>void run(size_t tasks,F callback,size_t chunk=1) {
        std::atomic<size_t> next{0};
        team_.apply_workers(count_,[&](size_t w) {
            for(;;) {
                size_t begin=next.fetch_add(chunk,std::memory_order_relaxed);if(begin>=tasks)break;
                for(size_t i=begin;i<std::min(tasks,begin+chunk);++i)callback(i,w);
            }
        });
    }
    size_t count() const { return count_; }
    template<class F>void each_worker(F callback) {
        team_.apply_workers(count_,std::move(callback));
    }
};

struct Primary {
    struct Group {
        const Config *cfg;
        const Block *block;
        size_t size() const {return block->size();}
        size_t row(size_t i) const {return block->offset+i;}
        size_t rank(size_t i) const {return block->rank(i);}
        bool canonical_full() const {return block->complete();}
    };
    std::shared_ptr<const Basis> native;
    std::vector<Config> configurations;
    int k,n,s;
    std::vector<Group> groups;

    explicit Primary(std::shared_ptr<const Basis> b)
        : native(std::move(b)),k(native->orbitals()),n(native->electrons()),s(native->two_s()) {
        configurations.reserve(native->blocks().size());
        groups.reserve(native->blocks().size());
        for(const auto &block:native->blocks()) {
            configurations.push_back(block.key.decode(k));
            groups.push_back({&configurations.back(),&block});
        }
    }
    size_t size() const {return native->size();}
    int maximum_selected_open() const {
        int open=s;
        for(const auto &group:groups)open=std::max(open,group.block->open);
        return open;
    }
    std::string label(size_t group,size_t local) const {
        const auto &g=groups[group];
        return native->spin(g.block->open).label(*g.cfg,g.rank(local));
    }
    size_t configuration_bytes() const {
        size_t bytes=configurations.capacity()*sizeof(Config);
        for(const auto &cfg:configurations)bytes+=cfg.capacity()*sizeof(int);
        return bytes;
    }
};

class RemovalBank {
    int spin_;
    size_t maximum_,bytes_=0;
    // Spin/projection, canonical occupation length, and (position, occupation
    // and spin) for each annihilator. All unlisted sites are singly occupied.
    using Key=std::array<int,7>;
    struct Hash {
        size_t operator()(const Key &key) const {
            size_t result=0;
            for(int value:key)result=result*1000003+unsigned(value);
            return result;
        }
    };
    std::unordered_map<Key,unsigned,Hash> directory_;
public:
    std::vector<Matrix> matrices;
    std::vector<unsigned> weights;
    RemovalBank(const Primary &primary,size_t maximum):spin_(primary.s),maximum_(maximum){}
    unsigned get(const Primary::Group &group,std::initializer_list<Operator> operators,int sb,int mb) {
        const auto &ket=*group.cfg;
        const int first=operators.begin()->orbital,last=(operators.end()-1)->orbital;
        const auto &occupation=group.block->key;
        auto position=[&](int orbital) {
            int rank=0;
            for(int base=0;base<=orbital/64;++base) {
                const auto &limb=base ? occupation.high[base-1] : occupation.first;
                auto open=limb[0]^limb[1];
                for(int site:{first,last})if(site/64==base)open|=std::uint64_t(1)<<(site%64);
                if(base==orbital/64)open&=(std::uint64_t(1)<<(orbital%64))-1;
                rank+=__builtin_popcountll(open);
            }
            return rank;
        };
        const int length=group.block->open+(ket[first]==2)+(last!=first && ket[last]==2);
        Key key{sb,mb,length,position(first),2*ket[first]+operators.begin()->spin,-1,0};
        if(operators.size()==2) {
            key[5]=first==last ? key[3] : position(last);
            key[6]=2*ket[last]+(operators.end()-1)->spin;
        }
        auto found=directory_.find(key);if(found!=directory_.end())return found->second;
        Config small(length,1);
        std::vector<Operator> ops;
        for(size_t j=3;j<key.size() && key[j]>=0;j+=2) {
            small[key[j]]=key[j+1]/2;
            ops.push_back({key[j],key[j+1]%2,false});
        }
        Config target=small;for(auto &op:ops)--target[op.orbital];
        int ma=std::count(target.begin(),target.end(),1),mk=std::count(small.begin(),small.end(),1);
        SpinPaths rows(ma,sb,maximum_/8),cols(mk,spin_,maximum_/8);
        if(rows.paths.size() && cols.paths.size()>maximum_/8/rows.paths.size())throw BudgetError("removal class exceeds coefficient budget");
        Matrix matrix(rows.paths.size(),cols.paths.size());
        for(size_t a=0;a<matrix.rows;++a)for(size_t b=0;b<matrix.cols;++b)
            matrix.at(a,b)=cg_operator(spin_label(target,rows.paths[a]),sb,mb,
                spin_label(small,cols.paths[b]),spin_,spin_,ops);
        matrix.sparse();size_t add=matrix.bytes()+sizeof(Key)+sizeof(unsigned)+64;
        if(add>maximum_-bytes_)throw BudgetError("removal matrices exceed coefficient budget");
        bytes_+=add;unsigned id=schedule_index(matrices.size());matrices.push_back(std::move(matrix));
        weights.push_back(spin_==0 ? sb+1 : 1);
        directory_.emplace(std::move(key),id);return id;
    }
    size_t bytes() const{return bytes_;}
};

struct Residues {
    struct Record {std::string key;unsigned parent=0;int p=0,q=-1;};
    std::vector<Record> records;
    std::vector<size_t> ptr;
    std::vector<Config> configs;
    template<class Getter>Residues(size_t count,int k,int remove,size_t maximum,Getter source) {
        size_t forecast=0;
        for(size_t g=0;g<count;++g) {
            const auto &c=source(g);size_t occupied=0,doubles=0;
            for(int n:c){occupied+=n>0;doubles+=n==2;}
            size_t add=remove==1 ? occupied : occupied*(occupied-1)/2+doubles;
            if(add>(maximum/sizeof(Record))-forecast)throw BudgetError("residue records exceed metadata budget");
            forecast+=add;
        }
        records.reserve(forecast);
        for(size_t g=0;g<count;++g) {
            Config c=source(g);
            for(int p=0;p<k;++p)if(c[p]) {
                --c[p];
                if(remove==1)records.push_back({pack(c),schedule_index(g),p,-1});
                else for(int q=p;q<k;++q)if(c[q]) {--c[q];records.push_back({pack(c),schedule_index(g),p,q});++c[q];}
                ++c[p];
            }
        }
        std::sort(records.begin(),records.end(),[](const auto &a,const auto &b){return a.key<b.key || (a.key==b.key && std::tie(a.parent,a.p,a.q)<std::tie(b.parent,b.p,b.q));});
        ptr.push_back(0);
        for(size_t i=0;i<records.size();++i)if(i==0 || records[i].key!=records[i-1].key) {
            if(i)ptr.push_back(i);
            configs.push_back(unpack(records[i].key,k));
        }
        if(!records.empty())ptr.push_back(records.size());
        if(bytes()>maximum)throw BudgetError("residue metadata exceeds metadata budget");
    }
    size_t bytes() const {
        size_t b=records.capacity()*sizeof(Record)+ptr.capacity()*sizeof(size_t)+configs.capacity()*sizeof(Config);
        for(auto &c:configs)b+=c.capacity()*sizeof(int);
        for(auto &r:records)b+=r.key.capacity()+1;
        return b;
    }
};

// Single-limb occupations sort directly, without string keys or decoded
// residue configurations. Preserve the narrower records below 33 orbitals.
struct CompactRemovalResidues {
    using Key=Occupation::Limb;
    template<class Word>struct Record {std::array<Word,2> key;unsigned parent;std::uint8_t p,q;};
    std::vector<Record<std::uint32_t>> narrow;
    std::vector<Record<std::uint64_t>> wide;
    std::vector<size_t> ptr;

    template<class Word,class Getter>
    void build(std::vector<Record<Word>> &records,size_t groups,int rank,size_t maximum,size_t begin,Getter key_at) {
        size_t count=0;
        for(size_t id=begin;id<groups;++id) {
            const auto [o,d]=key_at(id);
            const size_t occupied=__builtin_popcountll(o);
            const size_t add=rank==1 ? occupied : occupied*(occupied-1)/2+__builtin_popcountll(d);
            if(add>maximum/sizeof(Record<Word>)-count)throw BudgetError("compact removal records exceed metadata budget");
            count+=add;
        }
        records.reserve(count);
        auto remove=[](std::uint64_t &o,std::uint64_t &d,std::uint64_t bit) {
            o^=bit&~d;
            d&=~bit;
        };
        for(size_t id=begin;id<groups;++id) {
            const auto key=key_at(id);
            const auto parent=schedule_index(id);
            auto holes=key[0];
            while(holes) {
                const auto p=std::uint8_t(__builtin_ctzll(holes));
                const auto bit=std::uint64_t(1)<<p;
                auto o=key[0],d=key[1];
                remove(o,d,bit);
                if(rank==1)records.push_back({{Word(o),Word(d)},parent,p,0});
                else {
                    auto seconds=o&~(bit-1);
                    while(seconds) {
                        const auto q=std::uint8_t(__builtin_ctzll(seconds));
                        const auto second=std::uint64_t(1)<<q;
                        auto oo=o,dd=d;
                        remove(oo,dd,second);
                        records.push_back({{Word(oo),Word(dd)},parent,p,q});
                        seconds&=seconds-1;
                    }
                }
                holes&=holes-1;
            }
        }
        std::sort(records.begin(),records.end(),[](const auto &a,const auto &b) {
            return std::tie(a.key,a.parent,a.p,a.q)<std::tie(b.key,b.parent,b.p,b.q);
        });
        size_t residues=0;
        for(size_t i=0;i<records.size();++i)residues+=!i || records[i-1].key!=records[i].key;
        if(sizeof(size_t)>maximum-count*sizeof(Record<Word>) ||
            residues>(maximum-count*sizeof(Record<Word>)-sizeof(size_t))/sizeof(size_t))
            throw BudgetError("compact residue offsets exceed metadata budget");
        ptr.reserve(residues+1);
        for(size_t i=0;i<records.size();++i)
            if(!i || records[i-1].key!=records[i].key)ptr.push_back(i);
        ptr.push_back(records.size());
    }
    template<class Getter>
    CompactRemovalResidues(size_t groups,int k,int rank,size_t maximum,size_t begin,Getter key_at) {
        if(k<=32)build(narrow,groups,rank,maximum,begin,key_at);
        else build(wide,groups,rank,maximum,begin,key_at);
    }
    Record<std::uint64_t> at(size_t i) const {
        if(narrow.empty())return wide[i];
        const auto &r=narrow[i];
        return {{r.key[0],r.key[1]},r.parent,r.p,r.q};
    }
    size_t bytes() const {
        return narrow.capacity()*sizeof(narrow[0])+wide.capacity()*sizeof(wide[0])+ptr.capacity()*sizeof(size_t);
    }
};

// Bidirectional N-1 neighbor queries through common N-2 occupations.
struct RemovalQueries {
    struct Posting {unsigned config;int orbital;};
    std::vector<size_t> ptr,reverse_ptr;
    std::vector<Posting> parents,queries;
    size_t build_bytes=0;

    RemovalQueries(const std::vector<Config> &configs,int k,size_t maximum) {
        auto compile=[&](auto &index,auto record) {
            const size_t count=index.ptr.back();
            const size_t offsets=2*(configs.size()+1)*sizeof(size_t);
            if(offsets>maximum-index.bytes() || count>(maximum-index.bytes()-offsets)/(2*sizeof(Posting)))
                throw BudgetError("auxiliary queries exceed metadata budget");
            build_bytes=index.bytes()+offsets+2*count*sizeof(Posting);
            ptr=std::move(index.ptr);
            reverse_ptr.assign(configs.size()+1,0);
            parents.resize(count);
            queries.resize(count);
            for(size_t i=0;i<count;++i) {
                const auto [parent,orbital]=record(i);
                parents[i]={parent,orbital};
                ++reverse_ptr[parent+1];
            }
            std::partial_sum(reverse_ptr.begin(),reverse_ptr.end(),reverse_ptr.begin());
            auto cursor=reverse_ptr;
            for(size_t a=0;a+1<ptr.size();++a)
                for(size_t i=ptr[a];i<ptr[a+1];++i)
                    queries[cursor[parents[i].config]++]={schedule_index(a),parents[i].orbital};
        };
        if(k<=64) {
            CompactRemovalResidues index(configs.size(),k,1,maximum,0,
                [&](size_t i) {return Occupation(configs[i]).first;});
            compile(index,[&](size_t i) {const auto r=index.at(i);return Posting{r.parent,r.p};});
        } else {
            Residues index(configs.size(),k,1,maximum,[&](size_t i)->const Config &{return configs[i];});
            compile(index,[&](size_t i) {const auto &r=index.records[i];return Posting{r.parent,r.p};});
        }
    }
    size_t bytes() const {
        return (ptr.capacity()+reverse_ptr.capacity())*sizeof(size_t)+(parents.capacity()+queries.capacity())*sizeof(Posting);
    }
};

struct RemovalLinks {
    struct Family {int spin,m;int weight=1;};
    struct Parent {unsigned group,matrix,channel;};
    struct Block {unsigned config,family;size_t begin,end,dim;};
    std::unique_ptr<Residues> residues;
    std::vector<Family> families;
    std::vector<Block> blocks;
    std::vector<Parent> parents;
    std::vector<size_t> block_ids,reverse_ptr,offsets;
    std::vector<unsigned> reverse;
    std::vector<CompactRemovalResidues::Key> keys;
    size_t entries=0,weighted_links=0,residue_count=0,compact_builder_bytes=0;
    RemovalLinks(const Primary &primary,RemovalBank &bank,int rank,size_t maximum,
                 bool keep_block_ids,bool keep_reverse,size_t begin=0) {
        std::unique_ptr<CompactRemovalResidues> compact;
        if(primary.k<=64 && !keep_block_ids) {
            compact=std::make_unique<CompactRemovalResidues>(primary.groups.size(),primary.k,rank,maximum,begin,
                [&](size_t i) {return primary.native->blocks()[i].key.first;});
            residue_count=compact->ptr.size()-1;
            compact_builder_bytes=compact->bytes();
            if(residue_count>(maximum-compact_builder_bytes)/sizeof(keys[0]))
                throw BudgetError("incremental residue keys exceed metadata budget");
            keys.reserve(residue_count);
            for(size_t index=0;index<residue_count;++index)
                keys.push_back(compact->at(compact->ptr[index]).key);
        } else {
            residues=std::make_unique<Residues>(primary.groups.size(),primary.k,rank,maximum,
                [&](size_t g)->const Config &{return *primary.groups[g].cfg;});
            residue_count=residues->configs.size();
        }
        const auto &ptr=compact ? compact->ptr : residues->ptr;
        auto removal=[&](size_t i)->std::tuple<unsigned,int,int> {
            if(compact) {
                const auto r=compact->at(i);
                return {r.parent,r.p,r.q};
            }
            const auto &r=residues->records[i];
            return {r.parent,r.p,r.q};
        };
        if(rank==1) {
            for(int sigma=0;sigma<2;++sigma)for(int spin:{primary.s-1,primary.s+1}) {
                int m=primary.s-(sigma==0 ? 1 : -1);
                if(spin>=std::abs(m) && spin<=primary.n-1 && (primary.s!=0 || m==spin))
                    families.push_back({spin,m,primary.s==0 ? spin+1 : 1});
            }
        } else {
            for(int spin:{primary.s-2,primary.s,primary.s+2})for(int m:{primary.s-2,primary.s,primary.s+2})
                if(spin>=std::abs(m) && spin<=primary.n-2 && (primary.s!=0 || m==spin))
                    families.push_back({spin,m,primary.s==0 ? spin+1 : 1});
        }
        if(keep_block_ids)block_ids.assign(residue_count*families.size(),none);
        std::map<std::pair<int,int>,size_t> dimensions;
        for(size_t a=0;a<residue_count;++a) {
            int singles=compact ? __builtin_popcountll(keys[a][0]^keys[a][1])
                : std::count(residues->configs[a].begin(),residues->configs[a].end(),1);
            for(size_t f=0;f<families.size();++f) {
                auto family=families[f];
                auto [dimension,inserted]=dimensions.try_emplace({singles,family.spin},0);
                if(inserted)dimension->second=SpinPaths(singles,family.spin,maximum/8).paths.size();
                size_t dim=dimension->second;
                if(!dim)continue;
                size_t first=parents.size();
                for(size_t j=ptr[a];j<ptr[a+1];++j) {
                    const auto [parent,p,q]=removal(j);
                    const auto &g=primary.groups[parent];
                    auto add=[&](std::initializer_list<Operator> ops,size_t channel) {
                        int projection=primary.s;
                        for(auto &op:ops)projection-=op.spin==0 ? 1 : -1;
                        if(projection!=family.m)return;
                        unsigned matrix=bank.get(g,ops,family.spin,family.m);
                        const auto &c=bank.matrices[matrix];
                        bool used=false;
                        if(g.canonical_full() || !keep_block_ids) {
                            used=!c.data.empty();
                            weighted_links+=c.data.size();
                        } else {
                            for(size_t row=0;row<c.rows;++row)for(size_t i=0;i<g.size();++i)
                                if(c.at(row,g.rank(i))!=0.) {used=true;++weighted_links;}
                        }
                        if(!used)return;
                        if(keep_reverse)offsets.push_back(entries+(parents.size()-first)*dim);
                        parents.push_back({parent,matrix,schedule_index(channel)});
                    };
                    if(rank==1)for(int sigma=0;sigma<2;++sigma)add({{p,sigma,false}},p);
                    else if(primary.s==0) {
                        // The singlet spin rotation sign cancels fermionic
                        // reordering, so D(p alpha,q beta) = D(q alpha,p beta).
                        int x=p+(family.spin ? primary.k : 0),y=q+primary.k;
                        if(x!=y)add({{p,x/primary.k,false},{q,1,false}},pair_index(x,y,2*primary.k));
                    }
                    else for(int sigma=0;sigma<2;++sigma)for(int tau=0;tau<2;++tau) {
                        int x=p+sigma*primary.k,y=q+tau*primary.k;
                        if(x==y || (p==q && sigma>tau))continue;
                        if(x>y)std::swap(x,y);
                        add({{x%primary.k,x/primary.k,false},{y%primary.k,y/primary.k,false}},pair_index(x,y,2*primary.k));
                    }
                }
                if(parents.size()==first)continue;
                if(keep_block_ids)block_ids[a*families.size()+f]=blocks.size();
                blocks.push_back({schedule_index(a),schedule_index(f),first,parents.size(),dim});
                if(dim && (parents.size()-first)>(no_row-entries)/dim)
                    throw BudgetError("removal amplitude size overflow");
                entries+=(parents.size()-first)*dim;
            }
        }
        if(keep_reverse)build_reverse(primary.groups.size());
        if(bytes()>maximum-compact_builder_bytes)throw BudgetError("removal links exceed metadata budget");
    }

    // Merge the compiled suffix into a previous append-only N-2 schedule.
    // The same residue/spin family must mix all old and new channels together.
    RemovalLinks(const RemovalLinks &old,const RemovalLinks &delta,size_t group_count,
                 size_t maximum,bool keep_reverse,const std::vector<std::pair<int,int>> *old_pairs,int modes) {
        const size_t retained=delta.bytes();
        if(retained>maximum || old.bytes()>maximum-retained ||
            (keep_reverse && group_count-old.reverse_ptr.size()+1>
                (maximum-retained-old.bytes())/sizeof(size_t)))
            throw BudgetError("incremental removal merge exceeds metadata budget");
        families=old.families;
        parents.reserve(checked_add(old.parents.size(),delta.parents.size()));
        blocks.reserve(checked_add(old.blocks.size(),delta.blocks.size()));
        keys.reserve(checked_add(old.keys.size(),delta.keys.size()));
        std::set_union(old.keys.begin(),old.keys.end(),delta.keys.begin(),delta.keys.end(),std::back_inserter(keys));
        residue_count=keys.size();
        size_t a=0,b=0;
        for(size_t id=0;id<keys.size();++id) {
            for(size_t f=0;f<families.size();++f) {
                const size_t first=parents.size();
                size_t dim=0;
                auto append=[&](const RemovalLinks &source,size_t &cursor,bool decode) {
                    if(cursor==source.blocks.size())return;
                    const auto &block=source.blocks[cursor];
                    if(source.keys[block.config]!=keys[id] || block.family!=f)return;
                    ++cursor;
                    dim=block.dim;
                    for(size_t j=block.begin;j<block.end;++j) {
                        auto posting=source.parents[j];
                        if(decode) {
                            const auto [p,q]=(*old_pairs)[posting.channel];
                            posting.channel=schedule_index(pair_index(p,q,modes));
                        }
                        parents.push_back(posting);
                    }
                };
                append(old,a,old_pairs!=nullptr);
                append(delta,b,false);
                if(first==parents.size())continue;
                blocks.push_back({schedule_index(id),schedule_index(f),first,parents.size(),dim});
                if((parents.size()-first)>(no_row-entries)/dim)
                    throw BudgetError("removal amplitude size overflow");
                entries+=(parents.size()-first)*dim;
            }
        }
        weighted_links=checked_add(old.weighted_links,delta.weighted_links);
        if(keep_reverse) {
            offsets.resize(parents.size());
            size_t base=0;
            for(const auto &block:blocks)for(size_t j=block.begin;j<block.end;++j) {
                offsets[j]=base;
                base+=block.dim;
            }
            build_reverse(group_count);
        }
        if(bytes()>maximum-retained)throw BudgetError("incremental removal schedule exceeds metadata budget");
        compact_builder_bytes=retained;
    }

    // The forward array begins at the offset of the block's first parent.
    size_t offset(const Block &block) const {return offsets[block.begin];}
    void build_reverse(size_t group_count) {
            reverse_ptr.assign(group_count+1,0);
            for(auto &p:parents)++reverse_ptr[p.group+1];
            std::partial_sum(reverse_ptr.begin(),reverse_ptr.end(),reverse_ptr.begin());
            reverse.resize(parents.size());
            auto cursor=reverse_ptr;
            for(size_t p=0;p<parents.size();++p)reverse[cursor[parents[p].group]++]=schedule_index(p);
    }
        size_t bytes() const {
        return keys.capacity()*sizeof(keys[0])+(residues ? residues->bytes() : 0)+families.capacity()*sizeof(Family)+blocks.capacity()*sizeof(Block)
            +parents.capacity()*sizeof(Parent)+(block_ids.capacity()+reverse_ptr.capacity()+offsets.capacity())*sizeof(size_t)
            +reverse.capacity()*sizeof(unsigned);
    }
};

} }

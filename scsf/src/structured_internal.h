/* SCSF; GPL-3.0-or-later. Shared exact operator-link infrastructure. */
#pragma once
#include "structured_spin.h"
#include "parallel.h"
#include "pair_descriptor.h"
#include "basis.h"
#include <chrono>
#include <set>
#include <mutex>

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
        const v2::Block *block;
        size_t size() const {return block->size();}
        size_t row(size_t i) const {return block->offset+i;}
        size_t rank(size_t i) const {return block->rank(i);}
        bool canonical_full() const {return block->complete();}
    };
    std::shared_ptr<const v2::Basis> native;
    std::vector<Config> configurations;
    int k,n,s;
    std::vector<Group> groups;

    explicit Primary(std::shared_ptr<const v2::Basis> b)
        : native(std::move(b)),k(native->orbitals()),n(native->electrons()),s(native->two_s()) {
        configurations.reserve(native->blocks().size());
        groups.reserve(native->blocks().size());
        for(const auto &block:native->blocks()) {
            configurations.push_back(block.key.decode(k));
            groups.push_back({&configurations.back(),&block});
        }
    }
    size_t size() const {return native->size();}
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
    std::map<std::string,unsigned> directory_;
public:
    std::vector<Matrix> matrices;
    RemovalBank(const Primary &primary,size_t maximum):spin_(primary.s),maximum_(maximum){}
    unsigned get(const Config &ket,std::initializer_list<Operator> operators,int sb,int mb) {
        // Removal links supply one or two valid, ordered annihilation operators.
        // Build the canonical key on hits; expand the spin matrix inputs on misses.
        std::array<Operator,2> mapped;
        std::copy(operators.begin(),operators.end(),mapped.begin());
        const size_t count=operators.size();
        std::string occupations;
        for(size_t p=0;p<ket.size();++p) {
            bool keep=ket[p]==1;
            for(size_t i=0;i<count;++i)if(operators.begin()[i].orbital==int(p)) {
                mapped[i].orbital=occupations.size();
                keep=true;
            }
            if(keep)occupations+=char(ket[p]);
        }
        std::string key;
        key.reserve((3+2*count)*sizeof(int)+occupations.size());
        auto integer=[&](int v) { key.append(reinterpret_cast<const char *>(&v),sizeof(v)); };
        integer(sb);integer(mb);integer(count);
        for(size_t i=0;i<count;++i) {integer(mapped[i].orbital);integer(mapped[i].spin);}
        key+=occupations;
        auto found=directory_.find(key);if(found!=directory_.end())return found->second;
        Config small(occupations.begin(),occupations.end());
        std::vector<Operator> ops(mapped.begin(),mapped.begin()+count);
        Config target=small;for(auto &op:ops)--target[op.orbital];
        int ma=std::count(target.begin(),target.end(),1),mk=std::count(small.begin(),small.end(),1);
        SpinPaths rows(ma,sb,maximum_/8),cols(mk,spin_,maximum_/8);
        if(rows.paths.size() && cols.paths.size()>maximum_/8/rows.paths.size())throw BudgetError("removal class exceeds coefficient budget");
        Matrix matrix(rows.paths.size(),cols.paths.size());
        for(size_t a=0;a<matrix.rows;++a)for(size_t b=0;b<matrix.cols;++b)
            matrix.at(a,b)=cg_operator(spin_label(target,rows.paths[a]),sb,mb,
                spin_label(small,cols.paths[b]),spin_,spin_,ops);
        matrix.sparse();size_t add=matrix.bytes()+key.size()+64;
        if(add>maximum_-bytes_)throw BudgetError("removal matrices exceed coefficient budget");
        bytes_+=add;unsigned id=schedule_index(matrices.size());matrices.push_back(std::move(matrix));directory_.emplace(std::move(key),id);return id;
    }
    size_t bytes() const{return bytes_;}
};

struct Residues {
    struct Record {std::string key;unsigned parent=0;int p=0,q=-1;};
    std::vector<Record> records;
    std::vector<size_t> ptr;
    std::vector<Config> configs;
    std::vector<size_t> reverse_ptr;
    struct Query {unsigned residue;int orbital;};
    std::vector<Query> queries;
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
        if(remove==1) {
            reverse_ptr.assign(count+1,0);for(auto &r:records)++reverse_ptr[r.parent+1];
            std::partial_sum(reverse_ptr.begin(),reverse_ptr.end(),reverse_ptr.begin());queries.resize(records.size());auto cursor=reverse_ptr;
            for(size_t a=0;a<configs.size();++a)for(size_t r=ptr[a];r<ptr[a+1];++r)
                queries[cursor[records[r].parent]++]={schedule_index(a),records[r].p};
        }
        if(bytes()>maximum)throw BudgetError("residue metadata exceeds metadata budget");
    }
    size_t bytes() const {
        size_t b=records.capacity()*sizeof(Record)+ptr.capacity()*sizeof(size_t)+reverse_ptr.capacity()*sizeof(size_t)+queries.capacity()*sizeof(Query)+configs.capacity()*sizeof(Config);
        for(auto &c:configs)b+=c.capacity()*sizeof(int);
        for(auto &r:records)b+=r.key.capacity()+1;
        return b;
    }
};

// Up to 32 spatial sites fit in one pair of 32-bit masks. Use these
// records only during streamed N-2 construction. Wide occupations keep the
// existing string-key builder.
struct CompactRemovalResidues {
    struct Record {std::uint64_t key;unsigned parent;std::uint16_t p,q;};
    std::vector<Record> records;
    std::vector<size_t> ptr;
    std::vector<int> open;

    CompactRemovalResidues(const Primary &primary,int rank,size_t maximum,size_t begin) {
        size_t count=0;
        for(size_t id=begin;id<primary.groups.size();++id) {
            const auto &block=primary.native->blocks()[id];
            const auto o=block.key.first[0],d=block.key.first[1];
            const size_t occupied=__builtin_popcountll(o);
            const size_t add=rank==1 ? occupied : occupied*(occupied-1)/2+__builtin_popcountll(d);
            if(add>maximum/sizeof(Record)-count)throw BudgetError("compact removal records exceed metadata budget");
            count+=add;
        }
        records.reserve(count);
        auto remove=[](std::uint64_t &o,std::uint64_t &d,std::uint64_t bit) {
            o^=bit&~d;
            d&=~bit;
        };
        for(size_t id=begin;id<primary.groups.size();++id) {
            const auto &key=primary.native->blocks()[id].key.first;
            const auto parent=schedule_index(id);
            auto holes=key[0];
            while(holes) {
                const auto p=std::uint16_t(__builtin_ctzll(holes));
                const auto bit=std::uint64_t(1)<<p;
                auto o=key[0],d=key[1];
                remove(o,d,bit);
                if(rank==1)records.push_back({o|(d<<32),parent,p,0});
                else {
                    auto seconds=o&~(bit-1);
                    while(seconds) {
                        const auto q=std::uint16_t(__builtin_ctzll(seconds));
                        const auto second=std::uint64_t(1)<<q;
                        auto oo=o,dd=d;
                        remove(oo,dd,second);
                        records.push_back({oo|(dd<<32),parent,p,q});
                        seconds&=seconds-1;
                    }
                }
                holes&=holes-1;
            }
        }
        std::sort(records.begin(),records.end(),[](const Record &a,const Record &b) {
            return std::tie(a.key,a.parent,a.p,a.q)<std::tie(b.key,b.parent,b.p,b.q);
        });
        size_t residues=0;
        for(size_t i=0;i<records.size();++i)residues+=!i || records[i-1].key!=records[i].key;
        if(sizeof(size_t)>maximum-count*sizeof(Record) ||
            residues>(maximum-count*sizeof(Record)-sizeof(size_t))/(sizeof(size_t)+sizeof(int)))
            throw BudgetError("compact residue offsets exceed metadata budget");
        ptr.reserve(residues+1);
        open.reserve(residues);
        for(size_t i=0;i<records.size();++i)if(!i || records[i-1].key!=records[i].key) {
            auto key=records[i].key;
            ptr.push_back(i);
            open.push_back(__builtin_popcount(std::uint32_t(key)^std::uint32_t(key>>32)));
        }
        ptr.push_back(records.size());
    }
    size_t bytes() const {
        return records.capacity()*sizeof(Record)+ptr.capacity()*sizeof(size_t)+open.capacity()*sizeof(int);
    }
};

struct RemovalLinks {
    struct Family {int spin,m;int weight=1;};
    struct Parent {unsigned group,matrix,channel,weight;};
    struct Block {unsigned config,family;size_t begin,end,dim,offset;};
    std::unique_ptr<Residues> residues;
    std::vector<Family> families;
    std::vector<Block> blocks;
    std::vector<Parent> parents;
    std::vector<size_t> block_ids,reverse_ptr,offsets;
    std::vector<unsigned> reverse;
    std::vector<std::uint64_t> keys;
    size_t entries=0,weighted_links=0,residue_count=0,compact_builder_bytes=0;
    RemovalLinks(const Primary &primary,RemovalBank &bank,int rank,size_t maximum,
                 bool keep_block_ids,bool keep_reverse,size_t begin=0) {
        std::unique_ptr<CompactRemovalResidues> compact;
        if(primary.k<=32 && !keep_block_ids) {
            compact=std::make_unique<CompactRemovalResidues>(primary,rank,maximum,begin);
            residue_count=compact->open.size();
            compact_builder_bytes=compact->bytes();
            if(residue_count>(maximum-compact_builder_bytes)/sizeof(keys[0]))
                throw BudgetError("incremental residue keys exceed metadata budget");
            keys.reserve(residue_count);
            for(size_t index=0;index<residue_count;++index)
                keys.push_back(compact->records[compact->ptr[index]].key);
        } else {
            residues=std::make_unique<Residues>(primary.groups.size(),primary.k,rank,maximum,
                [&](size_t g)->const Config &{return *primary.groups[g].cfg;});
            residue_count=residues->configs.size();
        }
        const auto &ptr=compact ? compact->ptr : residues->ptr;
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
            int singles=compact ? compact->open[a] : std::count(residues->configs[a].begin(),residues->configs[a].end(),1);
            for(size_t f=0;f<families.size();++f) {
                auto family=families[f];
                auto [dimension,inserted]=dimensions.try_emplace({singles,family.spin},0);
                if(inserted)dimension->second=SpinPaths(singles,family.spin,maximum/8).paths.size();
                size_t dim=dimension->second;
                if(!dim)continue;
                size_t first=parents.size();
                for(size_t j=ptr[a];j<ptr[a+1];++j) {
                    const unsigned parent=compact ? compact->records[j].parent : residues->records[j].parent;
                    const int p=compact ? compact->records[j].p : residues->records[j].p;
                    const int q=compact ? compact->records[j].q : residues->records[j].q;
                    const auto &g=primary.groups[parent];
                    auto add=[&](std::initializer_list<Operator> ops,size_t channel) {
                        int projection=primary.s;
                        for(auto &op:ops)projection-=op.spin==0 ? 1 : -1;
                        if(projection!=family.m)return;
                        unsigned matrix=bank.get(*g.cfg,ops,family.spin,family.m);
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
                        parents.push_back({parent,matrix,schedule_index(channel),unsigned(family.weight)});
                    };
                    if(rank==1)for(int sigma=0;sigma<2;++sigma)add({{p,sigma,false}},p+sigma*primary.k);
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
                blocks.push_back({schedule_index(a),schedule_index(f),first,parents.size(),dim,entries});
                if(dim && (parents.size()-first)>(v2::no_row-entries)/dim)
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
        parents.reserve(v2::checked_add(old.parents.size(),delta.parents.size()));
        blocks.reserve(v2::checked_add(old.blocks.size(),delta.blocks.size()));
        keys.reserve(v2::checked_add(old.keys.size(),delta.keys.size()));
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
                blocks.push_back({schedule_index(id),schedule_index(f),first,parents.size(),dim,entries});
                if((parents.size()-first)>(v2::no_row-entries)/dim)
                    throw BudgetError("removal amplitude size overflow");
                entries+=(parents.size()-first)*dim;
            }
        }
        weighted_links=v2::checked_add(old.weighted_links,delta.weighted_links);
        if(keep_reverse) {
            offsets.resize(parents.size());
            for(const auto &block:blocks)for(size_t j=block.begin;j<block.end;++j)
                offsets[j]=block.offset+(j-block.begin)*block.dim;
            build_reverse(group_count);
        }
        if(bytes()>maximum-retained)throw BudgetError("incremental removal schedule exceeds metadata budget");
        compact_builder_bytes=retained;
    }

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

/* SCSF; GPL-3.0-or-later. Shared exact operator-link infrastructure. */
#pragma once
#include "structured_spin.h"
#include "structured_eri.h"
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
    template<class F>void run(size_t tasks,F callback,size_t chunk=1,size_t active=0) {
        std::atomic<size_t> next{0};
        team_.apply_workers(active ? active : count_,[&](size_t w) {
            for(;;) {
                size_t begin=next.fetch_add(chunk,std::memory_order_relaxed);if(begin>=tasks)break;
                for(size_t i=begin;i<std::min(tasks,begin+chunk);++i)callback(i,w);
            }
        });
    }
    size_t count() const { return count_; }
    template<class F>void each_worker(F callback,size_t active=0) {
        team_.apply_workers(active ? active : count_,std::move(callback));
    }
};

inline std::vector<double> coulomb_exchange_tables(const SpatialERI &eri,Workers &workers) {
    const size_t k=eri.k,kk=k*k;
    std::vector<double> tables(2*kk);
    workers.run(k,[&](size_t p,size_t) {
        const bool packed=eri.format!=ERIFormat::dense;
        for(size_t r=0;r<(packed ? p+1 : k);++r) {
            const double coulomb=eri(p,p,r,r),exchange=eri(p,r,p,r);
            tables[p*k+r]=coulomb;
            tables[kk+p*k+r]=exchange;
            if(packed && p!=r) {
                tables[r*k+p]=coulomb;
                tables[kk+r*k+p]=exchange;
            }
        }
    });
    return tables;
}

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
    size_t maximum_,bytes_=0,build_ns_=0,parallel_builds_=0;
public:
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
private:
    std::unordered_map<Key,unsigned,Hash> directory_;
public:
    std::vector<Matrix> matrices;
    std::vector<unsigned> weights;
    RemovalBank(const Primary &primary,size_t maximum):spin_(primary.s),maximum_(maximum){}
    void copy_classes(const RemovalBank &old) {
        memory_left(maximum_,old.bytes_,"removal coefficients");
        directory_=old.directory_;
        matrices=old.matrices;
        weights=old.weights;
        bytes_=old.bytes_;
    }
    static Key canonical_key(const Primary::Group &group,std::initializer_list<Operator> operators,
                             int sb,int mb) {
        const auto &ket=*group.cfg;
        const int first=operators.begin()->orbital,last=(operators.end()-1)->orbital;
        const auto &occupation=group.block->key;
        const bool single_limb=occupation.high.empty();
        const auto open=single_limb ? (occupation.first[0]^occupation.first[1])|
            (std::uint64_t(1)<<first)|(std::uint64_t(1)<<last) : 0;
        auto position=[&](int orbital) {
            if(single_limb)
                return __builtin_popcountll(open&((std::uint64_t(1)<<orbital)-1));
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
        return key;
    }
    size_t lookup(const Key &key) const {
        const auto found=directory_.find(key);
        return found==directory_.end() ? none : size_t(found->second);
    }
    unsigned get(const Primary::Group &group,std::initializer_list<Operator> operators,int sb,int mb,
                 Workers *workers) {
        return ensure(canonical_key(group,operators,sb,mb),workers);
    }
    unsigned ensure(const Key &key,Workers *workers,size_t allowance=no_row) {
        const auto found=lookup(key);
        if(found!=none)return static_cast<unsigned>(found);
        const auto start=Clock::now();
        const int sb=key[0],mb=key[1],length=key[2];
        Config small(length,1);
        std::vector<Operator> ops;
        for(size_t j=3;j<key.size() && key[j]>=0;j+=2) {
            small[key[j]]=key[j+1]/2;
            ops.push_back({key[j],key[j+1]%2,false});
        }
        Config target=small;for(auto &op:ops)--target[op.orbital];
        int ma=std::count(target.begin(),target.end(),1),mk=std::count(small.begin(),small.end(),1);
        const size_t cells=std::min(maximum_,allowance)/sizeof(double);
        SpinPaths rows(ma,sb,cells),cols(mk,spin_,cells);
        if(rows.paths.size() && cols.paths.size()>cells/rows.paths.size())
            throw BudgetError("removal class exceeds coefficient budget");
        Matrix matrix(rows.paths.size(),cols.paths.size());
        auto fill_row=[&](size_t a,size_t) {
            const auto bra=spin_label(target,rows.paths[a]);
            for(size_t b=0;b<matrix.cols;++b)
                matrix.at(a,b)=cg_operator(bra,sb,mb,spin_label(small,cols.paths[b]),spin_,spin_,ops);
        };
        // Matrix coefficients are independent; sparse packing and bank
        // publication happen only after all rows have been filled.
        if(workers && workers->count()>1 && matrix.rows>=16 && matrix.rows*matrix.cols>=4096) {
            workers->run(matrix.rows,fill_row);
            ++parallel_builds_;
        } else for(size_t a=0;a<matrix.rows;++a)fill_row(a,0);
        matrix.sparse();size_t add=matrix.bytes()+sizeof(Key)+sizeof(unsigned)+64;
        if(add>std::min(maximum_-bytes_,allowance))throw BudgetError("removal matrices exceed coefficient budget");
        bytes_+=add;unsigned id=schedule_index(matrices.size());matrices.push_back(std::move(matrix));
        weights.push_back(spin_==0 ? sb+1 : 1);
        directory_.emplace(key,id);
        build_ns_+=ns(start);
        return id;
    }
    size_t bytes() const{return bytes_;}
    size_t build_ns() const{return build_ns_;}
    size_t parallel_builds() const{return parallel_builds_;}
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

// Sort occupation bitsets directly, without allocating keys or decoding residues.
struct CompactRemovalResidues {
    using Key=Occupation::Limb;
    struct DoubleKey {
        std::array<Key,2> limbs;
        // Construction and merge write each key in the worker that owns it.
        DoubleKey() noexcept {}
        DoubleKey(const std::array<Key,2> &key) noexcept : limbs(key) {}
        bool operator<(const DoubleKey &other) const {return limbs<other.limbs;}
        bool operator==(const DoubleKey &other) const {return limbs==other.limbs;}
        bool operator!=(const DoubleKey &other) const {return !(*this==other);}
        const Key &operator[](size_t i) const {return limbs[i];}
    };
    template<class Word>struct Record {std::array<Word,2> key;unsigned parent;std::uint8_t p,q;};
    template<size_t Limbs>struct MultiRecord {
        std::array<Key,Limbs> key;
        unsigned parent;
        std::uint8_t p,q;
    };
    std::vector<Record<std::uint32_t>> narrow;
    std::vector<Record<std::uint64_t>> wide;
    std::unique_ptr<MultiRecord<2>[]> double_limb;
    std::unique_ptr<MultiRecord<3>[]> triple_limb;
    std::unique_ptr<MultiRecord<4>[]> quadruple_limb;
    size_t multi_bytes=0;
    std::vector<size_t> ptr;
    size_t generate_ns=0,sort_ns=0;

    template<class Word,class Getter>
    void build(std::vector<Record<Word>> &records,size_t groups,int rank,size_t maximum,size_t begin,
               Getter key_at,Workers *workers) {
        auto start=Clock::now();
        size_t count=0;
        for(size_t id=begin;id<groups;++id) {
            const auto [o,d]=key_at(id);
            const size_t occupied=__builtin_popcountll(o);
            const size_t add=rank==1 ? occupied : occupied*(occupied-1)/2+__builtin_popcountll(d);
            if(add>maximum/sizeof(Record<Word>)-count)
                throw BudgetError("compact removal records exceed metadata budget");
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
        generate_ns=ns(start);
        sort_and_index(records.data(),count,maximum,workers);
    }

    template<class RecordType,class Less>
    static size_t partition_parallel(RecordType *records,size_t count,Workers &workers,Less less) {
        const size_t tasks=workers.count();
        const auto boundary=[&](size_t t) {return count/tasks*t+count%tasks*t/tasks;};
        std::array<RecordType,257> sample;
        for(size_t i=0;i<sample.size();++i)sample[i]=records[count/sample.size()*i];
        auto median=sample.begin()+sample.size()/2;
        std::nth_element(sample.begin(),median,sample.end(),less);
        std::array<size_t,256> cuts;
        workers.run(tasks,[&](size_t t,size_t) {
            cuts[t]=std::partition(records+boundary(t),records+boundary(t+1),
                [&](const auto &record){return less(record,*median);})-records;
        });
        size_t split=0;
        for(size_t t=0;t<tasks;++t)split+=cuts[t]-boundary(t);
        // Each tile is partitioned. Exchange disjoint runs on the wrong
        // side of the global split. Only the run descriptors need storage.
        struct Exchange {size_t left,right,count;};
        std::array<Exchange,512> exchanges;
        size_t exchanges_count=0,right=0,position=split;
        for(size_t t=0;boundary(t)<split;++t) {
            const size_t last=std::min(split,boundary(t+1));
            for(size_t i=cuts[t];i<last;) {
                while(position>=cuts[right]) {
                    ++right;
                    position=std::max(split,boundary(right));
                }
                const size_t length=std::min(last-i,cuts[right]-position);
                exchanges[exchanges_count++]={i,position,length};
                i+=length;
                position+=length;
            }
        }
        workers.run(exchanges_count,[&](size_t t,size_t) {
            const auto [left,right,length]=exchanges[t];
            std::swap_ranges(records+left,records+left+length,records+right);
        });
        return split;
    }

    template<class RecordType>
    void sort_and_index(RecordType *records,size_t count,size_t maximum,Workers *workers) {
        if(!count) {
            memory_left(maximum,sizeof(size_t),"compact residue offsets");
            ptr.push_back(0);
            return;
        }
        const auto start=Clock::now();
        const auto less=[](const auto &a,const auto &b) {
            return std::tie(a.key,a.parent,a.p,a.q)<std::tie(b.key,b.parent,b.p,b.q);
        };
        // Partition at medians, then sort disjoint ranges in place. The
        // concatenated ranges preserve the original total ordering.
        size_t partitions=1;
        constexpr size_t min_partition=262144;
        size_t split=count/2;
        if(workers && workers->count()>=8 && count/workers->count()>=min_partition) {
            const size_t cut=partition_parallel(records,count,*workers,less);
            if(cut>=count/4 && cut<=count-count/4) {
                split=cut;
                partitions=2;
            }
        }
        const auto boundary=[&](size_t i,size_t ranges) {
            if(ranges==1)return records+i*count;
            const size_t half=ranges/2;
            const size_t first=i<half ? 0 : split;
            const size_t size=i<half ? split : count-split;
            const size_t offset=i<half ? i : i-half;
            return records+first+size/half*offset+(size%half)*offset/half;
        };
        while(workers && 2*partitions<=workers->count() && count/(2*partitions)>=min_partition) {
            const size_t ranges=2*partitions;
            workers->run(partitions,[&](size_t i,size_t) {
                std::nth_element(boundary(2*i,ranges),boundary(2*i+1,ranges),boundary(2*i+2,ranges),less);
            });
            partitions=ranges;
        }
        if(partitions==1)std::sort(records,records+count,less);
        else workers->run(partitions,[&](size_t i,size_t) {
            std::sort(boundary(i,partitions),boundary(i+1,partitions),less);
        });
        sort_ns=ns(start);
        auto starts_residue=[&](size_t i) {return !i || records[i-1].key!=records[i].key;};
        std::array<size_t,257> offsets{};
        const bool parallel_scan=partitions>1 && partitions<offsets.size();
        size_t residues=0;
        if(parallel_scan) {
            workers->run(partitions,[&](size_t task,size_t) {
                const size_t first=boundary(task,partitions)-records;
                const size_t last=boundary(task+1,partitions)-records;
                for(size_t i=first;i<last;++i)offsets[task+1]+=starts_residue(i);
            });
            std::partial_sum(offsets.begin(),offsets.begin()+partitions+1,offsets.begin());
            residues=offsets[partitions];
        } else {
            for(size_t i=0;i<count;++i)residues+=starts_residue(i);
        }
        if(sizeof(size_t)>maximum-count*sizeof(RecordType) ||
            residues>(maximum-count*sizeof(RecordType)-sizeof(size_t))/sizeof(size_t))
            throw BudgetError("compact residue offsets exceed metadata budget");
        if(parallel_scan) {
            ptr.resize(residues+1);
            workers->run(partitions,[&](size_t task,size_t) {
                const size_t first=boundary(task,partitions)-records;
                const size_t last=boundary(task+1,partitions)-records;
                size_t out=offsets[task];
                for(size_t i=first;i<last;++i)if(starts_residue(i))ptr[out++]=i;
            });
            ptr.back()=count;
        } else {
            ptr.reserve(residues+1);
            for(size_t i=0;i<count;++i)if(starts_residue(i))ptr.push_back(i);
            ptr.push_back(count);
        }
    }

    template<size_t Limbs>
    void build_multi(std::unique_ptr<MultiRecord<Limbs>[]> &records,const Primary &primary,
                     int rank,size_t maximum,size_t begin,Workers *workers) {
        using Posting=MultiRecord<Limbs>;
        const auto start=Clock::now();
        const size_t groups=primary.groups.size()-begin;
        const size_t tasks=workers ? std::min(workers->count(),std::max(size_t(1),groups/256)) : 1;
        const auto boundary=[&](size_t t) {return begin+groups/tasks*t+(groups%tasks)*t/tasks;};
        const auto run=[&](auto body) {
            if(tasks==1)body(0,0);
            else workers->run(tasks,body);
        };
        std::array<size_t,257> offsets{};
        run([&](size_t t,size_t) {
            size_t count=0;
            for(size_t g=boundary(t);g<boundary(t+1);++g) {
                const auto &key=primary.groups[g].block->key;
                size_t occupied=__builtin_popcountll(key.first[0]),doubles=__builtin_popcountll(key.first[1]);
                for(const auto &limb:key.high) {
                    occupied+=__builtin_popcountll(limb[0]);
                    doubles+=__builtin_popcountll(limb[1]);
                }
                const size_t add=rank==1 ? occupied : occupied*(occupied-1)/2+doubles;
                if(add>maximum/sizeof(Posting)-count)
                    throw BudgetError("compact removal records exceed metadata budget");
                count+=add;
            }
            offsets[t+1]=count;
        });
        for(size_t t=0;t<tasks;++t) {
            if(offsets[t+1]>maximum/sizeof(Posting)-offsets[t])
                throw BudgetError("compact removal records exceed metadata budget");
            offsets[t+1]+=offsets[t];
        }
        const size_t count=offsets[tasks];
        // Every worker writes its own range. Avoid clearing the whole array
        // on the caller before those writes.
        records.reset(new Posting[count]);
        multi_bytes=count*sizeof(Posting);
        const auto remove=[](auto &key,size_t p) {
            auto &[occupied,doubled]=key[p/64];
            const auto bit=std::uint64_t(1)<<(p%64);
            occupied^=bit&~doubled;
            doubled&=~bit;
        };
        const auto visit_occupied=[](const auto &key,size_t from,auto visit) {
            for(size_t limb=from/64;limb<Limbs;++limb) {
                auto bits=key[limb][0];
                if(limb==from/64)bits&=UINT64_MAX<<(from%64);
                for(;bits;bits&=bits-1)visit(limb*64+__builtin_ctzll(bits));
            }
        };
        run([&](size_t t,size_t) {
            size_t out=offsets[t];
            for(size_t g=boundary(t);g<boundary(t+1);++g) {
                const auto &occupation=primary.groups[g].block->key;
                std::array<Key,Limbs> key;
                key[0]=occupation.first;
                for(size_t limb=1;limb<Limbs;++limb)key[limb]=occupation.high[limb-1];
                const auto parent=schedule_index(g);
                visit_occupied(key,0,[&](size_t p) {
                    auto reduced=key;
                    remove(reduced,p);
                    if(rank==1)records[out++]={reduced,parent,std::uint8_t(p),0};
                    else visit_occupied(reduced,p,[&](size_t q) {
                        auto pair=reduced;
                        remove(pair,q);
                        records[out++]={pair,parent,std::uint8_t(p),std::uint8_t(q)};
                    });
                });
            }
        });
        generate_ns=ns(start);
        sort_and_index(records.get(),count,maximum,workers);
    }

    CompactRemovalResidues(const Primary &primary,int rank,size_t maximum,size_t begin,Workers *workers) {
        if(primary.k<=128)build_multi(double_limb,primary,rank,maximum,begin,workers);
        else if(primary.k<=192)build_multi(triple_limb,primary,rank,maximum,begin,workers);
        else build_multi(quadruple_limb,primary,rank,maximum,begin,workers);
    }
    template<class Getter>
    CompactRemovalResidues(size_t groups,int k,int rank,size_t maximum,size_t begin,Getter key_at,
                           Workers *workers=nullptr) {
        if(k<=32)build(narrow,groups,rank,maximum,begin,key_at,workers);
        else build(wide,groups,rank,maximum,begin,key_at,workers);
    }
    Record<std::uint64_t> at(size_t i) const {
        if(narrow.empty())return wide[i];
        const auto &r=narrow[i];
        return {{r.key[0],r.key[1]},r.parent,r.p,r.q};
    }
    std::tuple<unsigned,int,int> posting(size_t i) const {
        const auto read=[](const auto &r) {return std::tuple<unsigned,int,int>{r.parent,r.p,r.q};};
        if(!narrow.empty())return read(narrow[i]);
        if(!wide.empty())return read(wide[i]);
        if(double_limb)return read(double_limb[i]);
        if(triple_limb)return read(triple_limb[i]);
        return read(quadruple_limb[i]);
    }
    int singles(size_t residue) const {
        const size_t i=ptr[residue];
        const auto count=[](const auto &r) {
            int singles=0;
            for(const auto &limb:r.key)singles+=__builtin_popcountll(limb[0]^limb[1]);
            return singles;
        };
        if(double_limb)return count(double_limb[i]);
        if(triple_limb)return count(triple_limb[i]);
        return count(quadruple_limb[i]);
    }
    size_t bytes() const {
        return narrow.capacity()*sizeof(narrow[0])+wide.capacity()*sizeof(wide[0])+ptr.capacity()*sizeof(size_t)+multi_bytes;
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
    // The parallel builders size these arrays before dispatch and assign
    // every record. Let the worker writing a record first touch its pages.
    struct Parent {
        unsigned group,matrix,channel;
        Parent() noexcept {}
        Parent(unsigned group,unsigned matrix,unsigned channel) noexcept
            :group(group),matrix(matrix),channel(channel) {}
    };
    struct Block {
        unsigned config,family;
        size_t begin,end,dim;
        Block() noexcept {}
        Block(unsigned config,unsigned family,size_t begin,size_t end,size_t dim) noexcept
            :config(config),family(family),begin(begin),end(end),dim(dim) {}
    };
    struct CompactBlock {
        std::uint32_t config,begin,end,shape;
        static constexpr std::uint32_t max_dim=UINT32_MAX>>4;
        CompactBlock() noexcept {}
        CompactBlock(const Block &b) noexcept
            :config(b.config),begin(b.begin),end(b.end),shape((b.dim<<4)|b.family) {}
        Block expand() const {return {config,shape&15,begin,end,shape>>4};}
    };
    std::unique_ptr<Residues> residues;
    std::vector<Family> families;
    std::vector<Block> blocks;
    std::vector<CompactBlock> compact_blocks;
    std::vector<Parent> parents;
    std::vector<size_t> block_ids,reverse_ptr,offsets;
    std::vector<unsigned> reverse;
    std::vector<CompactRemovalResidues::Key> keys;
    std::vector<CompactRemovalResidues::DoubleKey> double_keys;
    size_t entries=0,weighted_links=0,residue_count=0,compact_builder_bytes=0,singletons=0,max_dim=0;
    Stats construction;
    size_t block_count() const {return blocks.size()+compact_blocks.size();}
    Block block_at(size_t i) const {return compact_blocks.empty() ? blocks[i] : compact_blocks[i].expand();}
    unsigned block_config(size_t i) const {return compact_blocks.empty() ? blocks[i].config : compact_blocks[i].config;}
    bool can_compact(size_t parent_count) const {
        return parent_count<=UINT32_MAX && max_dim<=CompactBlock::max_dim && families.size()<=16;
    }
    void resize_blocks(size_t count,bool compact) {
        if(compact)compact_blocks.resize(count);
        else blocks.resize(count);
    }
    void set_block(size_t i,const Block &block) {
        if(compact_blocks.empty())blocks[i]=block;
        else compact_blocks[i]=block;
    }
    int compact_singles(const CompactRemovalResidues &compact,size_t residue) const {
        if(!double_keys.empty()) {
            const auto &key=double_keys[residue];
            return __builtin_popcountll(key[0][0]^key[0][1])+__builtin_popcountll(key[1][0]^key[1][1]);
        }
        if(keys.empty())return compact.singles(residue);
        const auto &key=keys[residue];
        return __builtin_popcountll(key[0]^key[1]);
    }
    bool compile_private_singlet(const Primary &primary,RemovalBank &bank,int rank,size_t maximum,
                                 const CompactRemovalResidues &compact,Workers &workers,size_t reused_groups) {
        using Key=RemovalBank::Key;
        constexpr size_t slots=4096,max_tasks=256;
        const size_t tasks=std::min({max_tasks,workers.count(),residue_count/16384});
        if(tasks<2)return false;
        const size_t discovery_bytes=tasks*slots*2*sizeof(Key);
        const size_t retained=bytes(),occupied=checked_add(retained,compact_builder_bytes);
        if(occupied>maximum || discovery_bytes>maximum-occupied)return false;
        const size_t initial_coefficients=bank.bytes();
        // Split between residues so all families and parents for a residue
        // remain in one task. Prefixes preserve native order across tasks.
        const auto boundary=[&](size_t i) {
            return residue_count/tasks*i+residue_count%tasks*i/tasks;
        };
        std::array<std::array<size_t,2>,257> dims{};
        std::array<bool,257> seen{};
        for(size_t a=0;a<residue_count;++a) {
            const size_t singles=compact_singles(compact,a);
            if(seen[singles])continue;
            seen[singles]=true;
            for(size_t f=0;f<families.size();++f)
                dims[singles][f]=SpinPaths(singles,families[f].spin,maximum/8).paths.size();
        }
        const auto visit=[&](size_t t,size_t first_parent,auto block) {
            for(size_t a=boundary(t);a<boundary(t+1);++a) {
                const size_t end=compact.ptr[a+1];
                // Parents within a residue are sorted by group. Old groups
                // only use classes already in the bank.
                if(first_parent && std::get<0>(compact.posting(end-1))<first_parent)continue;
                const size_t singles=compact_singles(compact,a);
                for(size_t f=0;f<families.size();++f) {
                    const auto family=families[f];
                    const size_t dim=dims[singles][f];
                    if(!dim)continue;
                    block(a,f,dim,[&](auto add) {
                        for(size_t j=compact.ptr[a];j<end;++j) {
                            const auto [parent,p,q]=compact.posting(j);
                            if(parent<first_parent)continue;
                            const auto &g=primary.groups[parent];
                            const int k=primary.k;
                            if(rank==1) {
                                const auto key=RemovalBank::canonical_key(g,{{p,1,false}},family.spin,family.m);
                                add(key,parent,size_t(p));
                            } else {
                                const int x=p+(family.spin ? k : 0),y=q+k;
                                if(x!=y) {
                                    const auto key=RemovalBank::canonical_key(g,{{p,x/k,false},{q,1,false}},
                                                                                family.spin,family.m);
                                    add(key,parent,pair_index(x,y,2*k));
                                }
                            }
                        }
                    });
                }
            }
        };

        // Discover only missing classes against the read-only old bank. A
        // bounded local hash table avoids per-posting staging allocations.
        const auto discovery=Clock::now();
        std::vector<Key> table(tasks*slots);
        std::array<size_t,max_tasks> missing_counts{};
        workers.run(tasks,[&](size_t t,size_t) {
            auto *local=table.data()+t*slots;
            Key empty{};
            empty[0]=-1;
            std::fill_n(local,slots,empty);
            visit(t,reused_groups,[&](size_t,size_t,size_t,auto records) {
                records([&](const Key &key,unsigned,size_t) {
                    if(missing_counts[t]>slots/2 || bank.lookup(key)!=none)return;
                    size_t i=RemovalBank::Hash{}(key)&(slots-1);
                    while(local[i][0]>=0 && local[i]!=key)i=(i+1)&(slots-1);
                    if(local[i][0]<0) {local[i]=key;++missing_counts[t];}
                });
            });
        });
        size_t total_missing=0;
        for(size_t t=0;t<tasks;++t) {
            if(missing_counts[t]>slots/2)return false;
            total_missing+=missing_counts[t];
        }
        std::vector<Key> missing;
        missing.reserve(total_missing);
        for(const auto &key:table)if(key[0]>=0)missing.push_back(key);
        const size_t discovery_scratch=(table.capacity()+missing.capacity())*sizeof(Key);
        std::vector<Key>().swap(table);
        std::sort(missing.begin(),missing.end());
        missing.erase(std::unique(missing.begin(),missing.end()),missing.end());
        construction["parallel_class_discovery_ns"]=ns(discovery);
        const size_t created=missing.size();
        // No worker calls the mutable bank. A class may dispatch its own
        // parallel matrix-row fill from this controlling thread.
        const size_t class_allowance=maximum-occupied-missing.capacity()*sizeof(Key);
        for(const auto &key:missing) {
            const size_t remaining=memory_left(class_allowance,bank.bytes()-initial_coefficients,
                                               "removal coefficients");
            bank.ensure(key,&workers,remaining);
        }
        const size_t class_scratch=missing.capacity()*sizeof(Key);
        std::vector<Key>().swap(missing);

        struct Counts {size_t parents=0,blocks=0,entries=0,weights=0,lookups=0,singletons=0,max_dim=0;};
        std::array<Counts,max_tasks> counts{},offsets{};
        const auto counted=Clock::now();
        workers.run(tasks,[&](size_t t,size_t) {
            auto &c=counts[t];
            visit(t,0,[&](size_t,size_t,size_t dim,auto records) {
                const size_t before=c.parents;
                records([&](const Key &key,unsigned,size_t) {
                    ++c.lookups;
                    const size_t matrix=bank.lookup(key);
                    if(matrix==none)throw std::logic_error("missing frozen removal class");
                    const size_t nnz=bank.matrices[matrix].data.size();
                    c.weights=checked_add(c.weights,nnz);
                    c.parents+=nnz!=0;
                });
                if(c.parents==before)return;
                ++c.blocks;
                c.singletons+=c.parents-before==1;
                c.max_dim=std::max(c.max_dim,dim);
                if(c.parents-before>(no_row-c.entries)/dim)
                    throw BudgetError("removal amplitude size overflow");
                c.entries+=(c.parents-before)*dim;
            });
        });
        Counts total;
        for(size_t t=0;t<tasks;++t) {
            offsets[t]=total;
            const auto &c=counts[t];
            total.parents=checked_add(total.parents,c.parents);
            total.blocks=checked_add(total.blocks,c.blocks);
            total.entries=checked_add(total.entries,c.entries);
            total.weights=checked_add(total.weights,c.weights);
            total.lookups=checked_add(total.lookups,c.lookups);
            singletons+=c.singletons;
            max_dim=std::max(max_dim,c.max_dim);
        }
        construction["parallel_posting_count_ns"]=ns(counted);
        size_t available=memory_left(maximum-occupied,bank.bytes()-initial_coefficients,
                                     "removal coefficients");
        if(total.parents>available/sizeof(Parent))
            throw BudgetError("removal links exceed metadata budget");
        available-=total.parents*sizeof(Parent);
        const bool compact_layout=can_compact(total.parents);
        if(total.blocks>available/(compact_layout ? sizeof(CompactBlock) : sizeof(Block)))
            throw BudgetError("removal links exceed metadata budget");
        const auto filling=Clock::now();
        parents.resize(total.parents);
        resize_blocks(total.blocks,compact_layout);
        workers.run(tasks,[&](size_t t,size_t) {
            size_t parent=offsets[t].parents,block=offsets[t].blocks;
            visit(t,0,[&](size_t a,size_t f,size_t dim,auto records) {
                const size_t before=parent;
                records([&](const Key &key,unsigned g,size_t channel) {
                    const size_t matrix=bank.lookup(key);
                    if(matrix==none)throw std::logic_error("missing frozen removal class");
                    if(!bank.matrices[matrix].data.empty())
                        parents[parent++]={g,schedule_index(matrix),schedule_index(channel)};
                });
                if(parent!=before)set_block(block++,{schedule_index(a),schedule_index(f),before,parent,dim});
            });
        });
        construction["parallel_posting_fill_ns"]=ns(filling);
        construction["removal_lookup_hits"]=total.lookups-created;
        construction["parallel_posting_tasks"]=tasks;
        // Keep the peak of the three live buffer sets, accounting separately
        // for the coefficient bytes that the caller tracks in the bank.
        const size_t posting_bytes=bytes()-retained;
        const size_t scratch=std::max(discovery_scratch,class_scratch);
        if(scratch>posting_bytes)compact_builder_bytes+=scratch-posting_bytes;
        entries=total.entries;
        weighted_links=total.weights;
        return true;
    }

    RemovalLinks(const Primary &primary,RemovalBank &bank,int rank,size_t maximum,
                 bool keep_block_ids,bool keep_reverse,size_t begin=0,Workers *workers=nullptr,
                 bool parallel_postings=false,size_t reused_groups=0,bool retain_double_keys=false) {
        auto start=Clock::now();
        std::unique_ptr<CompactRemovalResidues> compact;
        if(primary.k<=64 && !keep_block_ids) {
            compact=std::make_unique<CompactRemovalResidues>(primary.groups.size(),primary.k,rank,maximum,begin,
                [&](size_t i) {return primary.native->blocks()[i].key.first;},workers);
            residue_count=compact->ptr.size()-1;
            compact_builder_bytes=compact->bytes();
            if(residue_count>(maximum-compact_builder_bytes)/sizeof(keys[0]))
                throw BudgetError("incremental residue keys exceed metadata budget");
            keys.reserve(residue_count);
            for(size_t index=0;index<residue_count;++index)
                keys.push_back(compact->at(compact->ptr[index]).key);
        } else if(primary.k<=256 && !keep_block_ids) {
            compact=std::make_unique<CompactRemovalResidues>(primary,rank,maximum,begin,workers);
            residue_count=compact->ptr.size()-1;
            compact_builder_bytes=compact->bytes();
            if(retain_double_keys && primary.k<=128 &&
                residue_count<=(maximum-compact_builder_bytes)/sizeof(double_keys[0])) {
                double_keys.resize(residue_count);
                auto fill=[&](size_t i,size_t) {double_keys[i]=compact->double_limb[compact->ptr[i]].key;};
                if(workers)workers->run(residue_count,fill,2048);
                else for(size_t i=0;i<residue_count;++i)fill(i,0);
            }
        } else {
            residues=std::make_unique<Residues>(primary.groups.size(),primary.k,rank,maximum,
                [&](size_t g)->const Config &{return *primary.groups[g].cfg;});
            residue_count=residues->configs.size();
        }
        const size_t generate=compact ? compact->generate_ns : 0,sort=compact ? compact->sort_ns : 0;
        construction["residue_generate_ns"]=generate;
        construction["residue_sort_ns"]=sort;
        construction["residue_index_ns"]=ns(start)-generate-sort;
        start=Clock::now();
        const size_t initial_classes=bank.matrices.size(),initial_build_ns=bank.build_ns();
        const size_t initial_parallel_builds=bank.parallel_builds();
        size_t lookups=0,sampled_hits=0,sampled_hit_ns=0;
        const auto &ptr=compact ? compact->ptr : residues->ptr;
        auto removal=[&](size_t i)->std::tuple<unsigned,int,int> {
            if(compact)return compact->posting(i);
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
        if(parallel_postings && compact && !keep_reverse && primary.s==0 && workers &&
            compile_private_singlet(primary,bank,rank,maximum,*compact,*workers,reused_groups)) {
            if(bytes()>maximum-compact_builder_bytes)throw BudgetError("removal links exceed metadata budget");
            construction["posting_compile_ns"]=ns(start);
            construction["removal_matrix_build_ns"]=bank.build_ns()-initial_build_ns;
            construction["parallel_removal_matrices"]=bank.parallel_builds()-initial_parallel_builds;
            return;
        }
        if(keep_block_ids)block_ids.assign(residue_count*families.size(),none);
        std::vector<std::array<size_t,5>> dimensions(residue_count ? size_t(primary.n-rank)+1 : 0);
        for(auto &row:dimensions)row.fill(no_row);
        for(size_t a=0;a<residue_count;++a) {
            int singles=compact ? compact_singles(*compact,a)
                : std::count(residues->configs[a].begin(),residues->configs[a].end(),1);
            for(size_t f=0;f<families.size();++f) {
                auto family=families[f];
                auto &dim=dimensions[singles][family.spin-primary.s+rank];
                if(dim==no_row)dim=SpinPaths(singles,family.spin,maximum/8).paths.size();
                if(!dim)continue;
                size_t first=parents.size();
                for(size_t j=ptr[a];j<ptr[a+1];++j) {
                    const auto [parent,p,q]=removal(j);
                    const auto &g=primary.groups[parent];
                    auto add=[&](std::initializer_list<Operator> ops,size_t channel) {
                        int projection=primary.s;
                        for(auto &op:ops)projection-=op.spin==0 ? 1 : -1;
                        if(projection!=family.m)return;
                        unsigned matrix;
                        // Sampling avoids calling the clock for every hit.
                        if((lookups++ & 2047)==0) {
                            const size_t classes=bank.matrices.size();
                            const auto lookup_start=Clock::now();
                            matrix=bank.get(g,ops,family.spin,family.m,workers);
                            if(bank.matrices.size()==classes) {
                                sampled_hit_ns+=ns(lookup_start);
                                ++sampled_hits;
                            }
                        } else matrix=bank.get(g,ops,family.spin,family.m,workers);
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
                singletons+=parents.size()-first==1;
                max_dim=std::max(max_dim,dim);
                if(keep_block_ids)block_ids[a*families.size()+f]=blocks.size();
                blocks.push_back({schedule_index(a),schedule_index(f),first,parents.size(),dim});
                if(dim && (parents.size()-first)>(no_row-entries)/dim)
                    throw BudgetError("removal amplitude size overflow");
                entries+=(parents.size()-first)*dim;
            }
        }
        if(keep_reverse)build_reverse(primary.groups.size());
        if(bytes()>maximum-compact_builder_bytes)throw BudgetError("removal links exceed metadata budget");
        construction["posting_compile_ns"]=ns(start);
        construction["removal_matrix_build_ns"]=bank.build_ns()-initial_build_ns;
        construction["parallel_removal_matrices"]=bank.parallel_builds()-initial_parallel_builds;
        construction["removal_lookup_hits"]=lookups-(bank.matrices.size()-initial_classes);
        construction["removal_lookup_samples"]=sampled_hits;
        construction["removal_lookup_sample_ns"]=sampled_hit_ns;
    }

    template<class Key>
    void merge_keys(const RemovalLinks &old,const RemovalLinks &delta,size_t maximum,
                    size_t reverse_bytes,const std::vector<std::pair<int,int>> *old_pairs,int modes,
                    Workers &workers,std::atomic<bool> *dirty,std::vector<Key> &merged,
                    const std::vector<Key> &old_keys,const std::vector<Key> &delta_keys) {
        struct Range {
            size_t old_key,delta_key,old_block,delta_block;
            size_t keys=0,blocks=0,joined_singletons=0,key_offset=0,block_offset=0;
        };
        const size_t upper=checked_add(old_keys.size(),delta_keys.size());
        if(!upper)return;
        const size_t tasks=std::max(size_t(1),std::min({workers.count(),upper/65536,size_t(256)}));
        std::array<Range,257> ranges{};
        auto lower_block=[](const RemovalLinks &links,size_t key) {
            size_t first=0,last=links.block_count();
            while(first<last) {
                const size_t middle=first+(last-first)/2;
                if(links.block_config(middle)<key)first=middle+1;
                else last=middle;
            }
            return first;
        };
        const bool split_old=old_keys.size()>=delta_keys.size();
        const auto &split_keys=split_old ? old_keys : delta_keys;
        for(size_t t=1;t<tasks;++t) {
            const size_t split=split_keys.size()/tasks*t+split_keys.size()%tasks*t/tasks;
            auto &r=ranges[t];
            r.old_key=split_old ? split :
                std::lower_bound(old_keys.begin(),old_keys.end(),split_keys[split])-old_keys.begin();
            r.delta_key=split_old ?
                std::lower_bound(delta_keys.begin(),delta_keys.end(),split_keys[split])-delta_keys.begin() : split;
            r.old_block=lower_block(old,r.old_key);
            r.delta_block=lower_block(delta,r.delta_key);
        }
        ranges[tasks]={old_keys.size(),delta_keys.size(),old.block_count(),delta.block_count()};
        auto order=[](const std::vector<Key> &keys,const Block &block) {
            return std::tie(keys[block.config],block.family);
        };
        auto run=[&](auto body) {
            if(tasks==1)body(0,0);
            else workers.run(tasks,body);
        };
        run([&](size_t t,size_t) {
            auto &r=ranges[t];
            const auto &last=ranges[t+1];
            size_t a=r.old_key,b=r.delta_key;
            r.keys=last.old_key-a+last.delta_key-b;
            while(a<last.old_key && b<last.delta_key) {
                if(old_keys[a]<delta_keys[b])++a;
                else if(delta_keys[b]<old_keys[a])++b;
                else {++a;++b;--r.keys;}
            }
            a=r.old_block;
            b=r.delta_block;
            r.blocks=last.old_block-a+last.delta_block-b;
            while(a<last.old_block && b<last.delta_block) {
                const auto x=old.block_at(a),y=delta.block_at(b);
                if(order(old_keys,x)<order(delta_keys,y))++a;
                else if(order(delta_keys,y)<order(old_keys,x))++b;
                else {
                    ++a;
                    ++b;
                    --r.blocks;
                    r.joined_singletons+=(x.end-x.begin==1)+(y.end-y.begin==1);
                    if(dirty && x.end-x.begin==1)
                        dirty[old.parents[x.begin].group].store(true,std::memory_order_relaxed);
                }
            }
        });
        size_t count=0;
        singletons=checked_add(old.singletons,delta.singletons);
        for(size_t t=0;t<tasks;++t) {
            auto &r=ranges[t];
            r.key_offset=residue_count;
            r.block_offset=count;
            residue_count=checked_add(residue_count,r.keys);
            count=checked_add(count,r.blocks);
            singletons-=r.joined_singletons;
        }
        schedule_index(residue_count-1);
        const size_t parent_count=checked_add(old.parents.size(),delta.parents.size());
        const bool compact=!reverse_bytes && can_compact(parent_count);
        size_t available=memory_left(maximum,bytes(),"incremental removal families");
        available=memory_left(available,reverse_bytes,"incremental reverse removal schedule");
        const size_t pair_count=old_pairs ? old_pairs->size() : 0;
        if(pair_count>available/sizeof(unsigned))
            throw BudgetError("incremental pair channels exceed metadata budget");
        available-=pair_count*sizeof(unsigned);
        if(residue_count>available/sizeof(Key))
            throw BudgetError("incremental residue keys exceed metadata budget");
        available-=residue_count*sizeof(Key);
        if(parent_count>available/sizeof(Parent))
            throw BudgetError("incremental removal parents exceed metadata budget");
        available-=parent_count*sizeof(Parent);
        if(count>available/(compact ? sizeof(CompactBlock) : sizeof(Block)))
            throw BudgetError("incremental removal blocks exceed metadata budget");
        std::vector<unsigned> old_channels;
        old_channels.reserve(pair_count);
        if(old_pairs)for(const auto &[p,q]:*old_pairs)
            old_channels.push_back(schedule_index(pair_index(p,q,modes)));
        merged.resize(residue_count);
        resize_blocks(count,compact);
        parents.resize(parent_count);
        const auto parent_start=[](const RemovalLinks &source,size_t block) {
            return block==source.block_count() ? source.parents.size() : source.block_at(block).begin;
        };
        run([&](size_t t,size_t) {
            auto &r=ranges[t];
            const auto &last=ranges[t+1];
            std::set_union(old_keys.begin()+r.old_key,old_keys.begin()+last.old_key,
                delta_keys.begin()+r.delta_key,delta_keys.begin()+last.delta_key,merged.begin()+r.key_offset);
            size_t a=r.old_block,b=r.delta_block,key=r.key_offset,block=r.block_offset;
            size_t old_begin=parent_start(old,a),old_end=old_begin;
            size_t next=old_begin+parent_start(delta,b),old_out=next;
            const auto copy_old=[&] {
                if(old_begin==old_end)return;
                const auto begin=old.parents.begin()+old_begin,end=old.parents.begin()+old_end;
                if(old_channels.empty())std::copy(begin,end,parents.begin()+old_out);
                else std::transform(begin,end,parents.begin()+old_out,[&](Parent parent) {
                    parent.channel=old_channels[parent.channel];
                    return parent;
                });
                old_begin=old_end;
            };
            while(a<last.old_block || b<last.delta_block) {
                const auto x=a<last.old_block ? old.block_at(a) : Block{0,0,0,0,0};
                const auto y=b<last.delta_block ? delta.block_at(b) : Block{0,0,0,0,0};
                const bool take_old=x.dim && (!y.dim || !(order(delta_keys,y)<order(old_keys,x)));
                const bool take_delta=y.dim && (!x.dim || !(order(old_keys,x)<order(delta_keys,y)));
                const auto &keys=take_old ? old_keys : delta_keys;
                const auto &item=take_old ? x : y;
                while(merged[key]!=keys[item.config])++key;
                const size_t first=next;
                if(take_old) {
                    old_end=x.end;
                    next+=x.end-x.begin;
                    ++a;
                }
                if(take_delta) {
                    copy_old();
                    std::copy(delta.parents.begin()+y.begin,delta.parents.begin()+y.end,parents.begin()+next);
                    next+=y.end-y.begin;
                    old_out=next;
                    ++b;
                }
                set_block(block++,{static_cast<unsigned>(key),item.family,first,next,item.dim});
            }
            copy_old();
        });
        compact_builder_bytes=old_channels.capacity()*sizeof(unsigned);
    }

    // Merge the compiled suffix into a previous append-only N-2 schedule.
    // The same residue/spin family must mix all old and new channels together.
    RemovalLinks(const RemovalLinks &old,const RemovalLinks &delta,size_t group_count,
                 size_t maximum,bool keep_reverse,const std::vector<std::pair<int,int>> *old_pairs,int modes,
                 Workers &workers,std::atomic<bool> *dirty=nullptr) {
        const size_t retained=delta.bytes();
        const size_t available=memory_left(maximum,retained,"incremental removal suffix");
        const size_t parent_count=checked_add(old.parents.size(),delta.parents.size());
        size_t reverse_bytes=0;
        if(keep_reverse) {
            const size_t directory=checked_add(group_count,1);
            if(directory>available/(2*sizeof(size_t)))throw BudgetError("incremental reverse removal directory");
            reverse_bytes=directory*2*sizeof(size_t);
            if(parent_count>(available-reverse_bytes)/(sizeof(size_t)+sizeof(unsigned)))
                throw BudgetError("incremental reverse removal postings");
            reverse_bytes+=parent_count*(sizeof(size_t)+sizeof(unsigned));
        }
        families=old.families;
        max_dim=std::max(old.max_dim,delta.max_dim);
        // A joined residue has the same dimension in both schedules.
        entries=checked_add(old.entries,delta.entries);
        if(!old.double_keys.empty() || !delta.double_keys.empty()) {
            if(old.double_keys.size()!=old.residue_count || delta.double_keys.size()!=delta.residue_count)
                throw BudgetError("wide removal keys exceed metadata budget");
            merge_keys(old,delta,available,reverse_bytes,old_pairs,modes,workers,dirty,double_keys,
                old.double_keys,delta.double_keys);
        } else {
            if(old.keys.size()!=old.residue_count || delta.keys.size()!=delta.residue_count)
                throw BudgetError("removal keys exceed metadata budget");
            merge_keys(old,delta,available,reverse_bytes,old_pairs,modes,workers,dirty,keys,old.keys,delta.keys);
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
            compact_builder_bytes=std::max(compact_builder_bytes,(group_count+1)*sizeof(size_t));
        }
        if(bytes()>available)throw BudgetError("incremental removal schedule exceeds metadata budget");
        compact_builder_bytes=checked_add(retained,compact_builder_bytes);
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
        return keys.capacity()*sizeof(keys[0])+double_keys.capacity()*sizeof(double_keys[0])
            +(residues ? residues->bytes() : 0)+families.capacity()*sizeof(Family)+blocks.capacity()*sizeof(Block)
            +compact_blocks.capacity()*sizeof(CompactBlock)
            +parents.capacity()*sizeof(Parent)+(block_ids.capacity()+reverse_ptr.capacity()+offsets.capacity())*sizeof(size_t)
            +reverse.capacity()*sizeof(unsigned);
    }
};

} }

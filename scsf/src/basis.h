/* SCSF; GPL-3.0-or-later. Native occupation blocks and genealogical spin ranks. */
#pragma once
#include "csf.h"
#include <algorithm>
#include <numeric>

namespace scsf {

inline constexpr size_t no_row = std::numeric_limits<size_t>::max();

inline size_t checked_add(size_t a, size_t b) {
    if (b > no_row-a) throw BudgetError("native basis index overflow");
    return a+b;
}

// Suffix counts for nonnegative walks. The up branch precedes the down
// branch, independently of the character codes and external bit-key order.
class SpinRanks {
    int m_, s_;
    size_t stride_;
    std::vector<size_t> counts_;
    size_t count(int left, int spin) const {
        return spin >= 0 && size_t(spin) < stride_ ? counts_[left*stride_+spin] : 0;
    }
public:
    SpinRanks(int m, int s) : m_(m), s_(s), stride_(size_t(m)+2) {
        if (s>m || (m-s)%2) return;
        counts_.resize((size_t(m)+1)*stride_);
        counts_[s]=1;
        for (int left=1; left<=m; ++left) {
            for (int prefix=(m-left)%2; prefix<=m-left; prefix+=2) {
                counts_[left*stride_+prefix]=checked_add(count(left-1,prefix+1),count(left-1,prefix-1));
            }
        }
    }
    size_t size() const { return counts_.empty() ? 0 : count(m_,0); }
    size_t bytes() const { return counts_.capacity()*sizeof(size_t); }
    size_t rank(const std::string &label) const {
        if (!size()) throw std::invalid_argument("invalid spin class");
        size_t result=0;
        int spin=0,left=m_;
        for (char step:label) {
            if (step!='u' && step!='d') continue;
            if (--left<0) throw std::invalid_argument("too many spin steps");
            if (step=='d') {
                result+=count(left,spin+1);
                --spin;
            } else ++spin;
            if (spin<0) throw std::invalid_argument("negative intermediate spin");
        }
        if (left || spin!=s_ || result>=size()) throw std::invalid_argument("spin path sector mismatch");
        return result;
    }
    std::string label(const Config &occupation, size_t rank) const {
        if (rank>=size()) throw std::out_of_range("spin rank exceeds class dimension");
        std::string result(occupation.size(),'0');
        int left=m_,spin=0;
        for (size_t p=0; p<occupation.size(); ++p) {
            if (occupation[p]==2) result[p]='2';
            if (occupation[p]!=1) continue;
            size_t up=count(--left,spin+1);
            if (rank<up) {
                result[p]='u';
                ++spin;
            } else {
                result[p]='d';
                rank-=up;
                --spin;
            }
        }
        return result;
    }
};

struct Occupation {
    using Limb=std::array<std::uint64_t,2>;
    Limb first{};
    std::vector<Limb> high;

    Occupation()=default;
    explicit Occupation(const Config &cfg) : high(cfg.empty() ? 0 : (cfg.size()-1)/64) {
        for (size_t p=0; p<cfg.size(); ++p) {
            auto &limb=p<64 ? first : high[p/64-1];
            const auto bit=std::uint64_t(1)<<(p%64);
            if (cfg[p]) limb[0]|=bit;
            if (cfg[p]==2) limb[1]|=bit;
        }
    }
    void transfer(const std::array<int,2> &holes,const std::array<int,2> &particles,int order) {
        for(int i=0;i<order;++i) {
            const int q=holes[i],p=particles[i];
            auto &src=q<64 ? first : high[q/64-1];
            auto &dst=p<64 ? first : high[p/64-1];
            const auto remove=std::uint64_t(1)<<(q%64),add=std::uint64_t(1)<<(p%64);
            src[0]^=remove&~src[1];
            src[1]&=~remove;
            dst[1]|=add&dst[0];
            dst[0]|=add;
        }
    }
    int singles() const {
        int open=__builtin_popcountll(first[0]^first[1]);
        for(const auto &limb:high)open+=__builtin_popcountll(limb[0]^limb[1]);
        return open;
    }
    bool operator==(const Occupation &other) const { return first==other.first && high==other.high; }
    Config decode(int k) const {
        Config cfg(k,0);
        auto unpack=[&](const Limb &limb, size_t start) {
            auto occupied=limb[0];
            while (occupied) {
                int bit=__builtin_ctzll(occupied);
                cfg[start+bit]=1+bool(limb[1]&(std::uint64_t(1)<<bit));
                occupied&=occupied-1;
            }
        };
        unpack(first,0);
        for (size_t i=0; i<high.size(); ++i) unpack(high[i],(i+1)*64);
        return cfg;
    }
    size_t hash() const {
        size_t value=0;
        auto mix=[&](const Limb &limb) {
            for (auto word:limb) {
                word=(word^(word>>30))*UINT64_C(0xbf58476d1ce4e5b9);
                word=(word^(word>>27))*UINT64_C(0x94d049bb133111eb);
                value^=(word^(word>>31))+UINT64_C(0x9e3779b97f4a7c15)+(value<<6)+(value>>2);
            }
        };
        mix(first);
        for (const auto &limb:high) mix(limb);
        return value;
    }
};

struct Block {
    Occupation key;
    size_t offset=0, dimension=0;
    int open;
    // Ranks are only present for partial occupations.
    std::vector<size_t> ranks;
    Block(Occupation occupation,int m,size_t dim) : key(std::move(occupation)),dimension(dim),open(m) {}
    bool complete() const { return ranks.empty(); }
    size_t size() const { return complete() ? dimension : ranks.size(); }
    size_t rank(size_t local) const { return complete() ? local : ranks[local]; }
    bool contains(size_t r) const {
        return complete() || std::binary_search(ranks.begin(),ranks.end(),r);
    }
};

class Basis {
public:
    using Additions=std::map<Config,std::vector<size_t>>;
private:
    int k_, n_, s_;
    size_t size_=0, generation_=0;
    std::vector<std::shared_ptr<const SpinRanks>> spins_;
    std::vector<Block> blocks_;
    std::unordered_multimap<size_t,size_t> directory_;
    std::vector<size_t> public_rows_;
    static int max_open(int k,int n,int s) {
        if (k<0 || k>std::numeric_limits<int>::max()/2 || n<0 || n>2*k ||
            s<0 || s>std::min(n,2*k-n) || (n-s)%2)
            throw std::invalid_argument("invalid native CSF sector");
        return std::min(n,2*k-n);
    }
    void offsets() {
        size_=0;
        for (auto &block:blocks_) {
            block.offset=size_;
            size_=checked_add(size_,block.size());
        }
        if (size_>size_t(std::numeric_limits<pybind11::ssize_t>::max()))
            throw BudgetError("native basis exceeds array index capacity");
    }
    size_t insert(Occupation key,int open) {
        const auto id=blocks_.size();
        directory_.emplace(key.hash(),id);
        blocks_.emplace_back(std::move(key),open,spin(open).size());
        return id;
    }
    void ensure_spin(int open) {
        auto &spin=spins_.at(open);
        if (!spin) spin=std::make_shared<SpinRanks>(open,s_);
    }
    void compact(Block &block) {
        std::sort(block.ranks.begin(),block.ranks.end());
        if (std::adjacent_find(block.ranks.begin(),block.ranks.end())!=block.ranks.end())
            throw std::invalid_argument("duplicate native CSF label");
        if (!block.ranks.empty() && block.ranks.back()>=block.dimension)
            throw std::invalid_argument("rank exceeds native spin class");
        if (block.ranks.size()==block.dimension) std::vector<size_t>().swap(block.ranks);
    }
public:
    Basis(int k,int n,int s) : k_(k),n_(n),s_(s),spins_(max_open(k,n,s)+1) {}
    Basis(int k,int n,int s,const std::vector<std::string> &labels) : Basis(k,n,s) {
        std::vector<std::pair<size_t,size_t>> order;
        order.reserve(labels.size());
        for (const auto &label:labels) {
            if (label.size()!=size_t(k)) throw std::invalid_argument("wrong native label width");
            const auto cfg=configuration(label,s);
            if (std::accumulate(cfg.begin(),cfg.end(),0)!=n)
                throw std::invalid_argument("wrong native label electron number");
            Occupation key(cfg);
            const int open=std::count(cfg.begin(),cfg.end(),1);
            auto id=find(key);
            if (id==no_row) {
                ensure_spin(open);
                id=insert(std::move(key),open);
            }
            const size_t rank=spin(open).rank(label);
            blocks_[id].ranks.push_back(rank);
            order.emplace_back(id,rank);
        }
        for (auto &block:blocks_) compact(block);
        offsets();
        size_t i=0;
        for (auto [id,rank]:order) {
            const auto &block=blocks_[id];
            const auto local=block.complete() ? rank : std::lower_bound(block.ranks.begin(),block.ranks.end(),rank)-block.ranks.begin();
            const size_t row=block.offset+local;
            if (row!=i && public_rows_.empty()) {
                public_rows_.reserve(size_);
                public_rows_.resize(i);
                std::iota(public_rows_.begin(),public_rows_.end(),0);
            }
            if (public_rows_.capacity()) public_rows_.push_back(row);
            ++i;
        }
    }
    int orbitals() const { return k_; }
    int electrons() const { return n_; }
    int two_s() const { return s_; }
    size_t size() const { return size_; }
    size_t generation() const { return generation_; }
    int maximum_open() const { return int(spins_.size())-1; }
    const SpinRanks &spin(int open) const { return *spins_[open]; }
    const auto &blocks() const { return blocks_; }
    bool public_is_native() const { return public_rows_.empty(); }
    const auto &public_rows() const { return public_rows_; }
    size_t find(const Occupation &key) const { return find(key,key.hash()); }
    size_t find(const Occupation &key,size_t hash) const {
        auto [first,last]=directory_.equal_range(hash);
        for (;first!=last;++first) if (blocks_[first->second].key==key) return first->second;
        return no_row;
    }
    std::string native_label(size_t row) const {
        if (row>=size_) throw std::out_of_range("native CSF row exceeds basis");
        auto block=std::upper_bound(blocks_.begin(),blocks_.end(),row,
            [](size_t i,const Block &b) { return i<b.offset; });
        --block;
        return spin(block->open).label(block->key.decode(k_),block->rank(row-block->offset));
    }
    std::string public_label(size_t row) const {
        return native_label(public_is_native() ? row : public_rows_.at(row));
    }
    std::pair<std::shared_ptr<Basis>,std::vector<size_t>> extend(const Additions &additions) const {
        auto next=std::make_shared<Basis>(*this);
        next->generation_=checked_add(generation_,1);
        for (const auto &[cfg,ranks]:additions) {
            Occupation key(cfg);
            size_t id=find(key);
            if (id==no_row) {
                const int open=std::count(cfg.begin(),cfg.end(),1);
                next->ensure_spin(open);
                id=next->insert(std::move(key),open);
                next->blocks_[id].ranks=ranks;
            } else if (!blocks_[id].complete()) {
                auto &block=next->blocks_[id];
                if (ranks.empty()) block.ranks.clear();
                else block.ranks.insert(block.ranks.end(),ranks.begin(),ranks.end());
            }
            next->compact(next->blocks_[id]);
        }
        next->offsets();
        std::vector<size_t> remap;
        for (size_t id=0; id<blocks_.size(); ++id) {
            const auto &old=blocks_[id],&block=next->blocks_[id];
            if (remap.empty() && old.offset==block.offset && old.size()==block.size()) continue;
            size_t local=0;
            for (size_t i=0; i<old.size(); ++i) {
                const auto rank=old.rank(i);
                if (block.complete()) local=rank;
                else while (block.rank(local)!=rank) ++local;
                const size_t before=old.offset+i,after=block.offset+local;
                if (after!=before && remap.empty()) {
                    remap.reserve(size_);
                    remap.resize(before);
                    std::iota(remap.begin(),remap.end(),0);
                }
                if (remap.capacity()) remap.push_back(after);
            }
        }
        if (!remap.empty() || !public_is_native()) {
            next->public_rows_.resize(next->size_);
            for (size_t i=0; i<size_; ++i) {
                const size_t row=public_is_native() ? i : public_rows_[i];
                next->public_rows_[i]=remap.empty() ? row : remap[row];
            }
            size_t at=size_,old_row=0;
            for (size_t row=0; row<next->size_; ++row) {
                if (old_row<size_ && row==(remap.empty() ? old_row : remap[old_row])) ++old_row;
                else next->public_rows_[at++]=row;
            }
        }
        return {std::move(next),std::move(remap)};
    }
    Stats info() const {
        size_t ranks=0,wide=0,spin_bytes=spins_.capacity()*sizeof(spins_[0]),complete=0;
        for (const auto &block:blocks_) {
            ranks+=block.ranks.capacity()*sizeof(size_t);
            wide+=block.key.high.capacity()*sizeof(Occupation::Limb);
            complete+=block.complete();
        }
        for (const auto &spin:spins_) if (spin) spin_bytes+=sizeof(SpinRanks)+spin->bytes();
        const size_t directory_bytes=directory_.bucket_count()*sizeof(void *)+
            directory_.size()*(sizeof(std::pair<const size_t,size_t>)+2*sizeof(void *));
        return {{"basis",size_},{"configurations",blocks_.size()},{"complete_configurations",complete},
                {"rank_map_bytes",ranks},{"public_permutation_bytes",public_rows_.capacity()*sizeof(size_t)},
                {"occupation_bytes",blocks_.capacity()*sizeof(Block)+wide+directory_bytes},
                {"wide_mask_bytes",wide},{"spin_rank_bytes",spin_bytes},{"generation",generation_}};
    }
};
} // namespace scsf

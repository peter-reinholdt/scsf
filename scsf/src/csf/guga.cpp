/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#include "csf.h"
#include "selected_spin_trie.h"
#include <pybind11/stl.h>
#include <algorithm>
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#include <immintrin.h>
#define SCSF_GUGA_BMI2 1
#endif

// One-electron segment factors: Dobrautz, Smart & Alavi, JCP 151, 094104
// (2019), Table 3, doi:10.1063/1.5108908. Implemented from the equations.
// Two-electron products sum complete intermediate walks by a two-state dynamic
// program over their spins, not over magnetic projections or selected CSFs.
namespace scsf {
namespace {
int step(char c) { return c == 'u' ? 1 : c == 'd' ? 2 : c == '2' ? 3 : 0; }
int occupation(char c) { return c == '2' ? 2 : c != '0'; }
int occupation(int d) { return d == 3 ? 2 : d != 0; }

// Small, immutable, integral-independent tables. Higher intermediate spins
// fall back to the analytic equations, so table size never grows with input.
constexpr int table_max_spin = 256;
using FactorTable = std::array<std::array<double,8>,table_max_spin+1>;
const FactorTable &factor_table() {
    static const FactorTable values = [] {
        FactorTable result{};
        for (int b = 0; b <= table_max_spin; ++b) result[b] = {
            b > 0 ? std::sqrt((b+1.0)/b) : 0.0,
            std::sqrt((b+1.0)/(b+2.0)),
            std::sqrt(b/(b+1.0)), std::sqrt((b+2.0)/(b+1.0)),
            b > 0 ? std::sqrt((b-1.0)*(b+1.0))/b : 0.0,
            std::sqrt((b+1.0)*(b+3.0))/(b+2.0),
            b > 0 ? 1.0/b : 0.0, 1.0/(b+2.0)};
        return result;
    }();
    return values;
}

double segment(int da, int db, int a, int b, int p, int q, int site, int mode = 0) {
    if (p > q) { std::swap(p, q); std::swap(da, db); std::swap(a, b); }
    const double *f = mode && b >= 0 && b <= table_max_spin ? factor_table()[b].data() : nullptr;
    if (p == q) {
        if (da != db || a != b) return 0.0;
        return site == p ? occupation(db) : 1.0;
    }
    if (site < p || site > q) return da == db && a == b ? 1.0 : 0.0;
    if (site == p) {
        if (occupation(da) != occupation(db)+1 || std::abs(a-b) != 1) return 0.0;
        if (db == 0) return 1.0;
        if (db == 1) return f ? f[0] : b > 0 ? std::sqrt((b+1.0)/b) : 0.0;
        return f ? f[1] : std::sqrt((b+1.0)/(b+2.0));
    }
    if (site == q) {
        if (occupation(da) != occupation(db)-1 || a != b) return 0.0;
        if (da == 0) return 1.0;
        return f ? f[da == 1 ? 2 : 3] : std::sqrt((da == 1 ? b : b+2.0)/(b+1.0));
    }
    int delta = b-a;
    if (occupation(da) != occupation(db) || std::abs(delta) != 1) return 0.0;
    if (da == db) {
        if (da == 0) return 1.0;
        if (da == 3) return -1.0;
        if (da == 1) return delta == -1 ? -1.0 :
            (f ? f[4] : b > 0 ? std::sqrt((b-1.0)*(b+1.0))/b : 0.0);
        return delta == 1 ? -1.0 : f ? f[5] : std::sqrt((b+1.0)*(b+3.0))/(b+2.0);
    }
    if (da == 1 && db == 2 && delta == -1) return f ? -f[7] : -1.0/(b+2.0);
    if (da == 2 && db == 1 && delta == 1 && b > 0) return f ? f[6] : 1.0/b;
    return 0.0;
}

// Segment-table convention; the outer CSF phase conversion is applied once.
template<bool Checkpoint>
double generator(const Path &bra, const Path &ket, int p, int q) {
    if (p > q) return generator<Checkpoint>(ket, bra, q, p);
    if (p == q) return bra.label == ket.label ? occupation(ket.label[p]) : 0.0;
    if (bra.label.compare(0, p, ket.label, 0, p) ||
        bra.label.compare(q+1, std::string::npos, ket.label, q+1, std::string::npos) ||
        bra.spins.get<Checkpoint>(q+1) != ket.spins.get<Checkpoint>(q+1)) return 0.0;
    double value = 1.0;
    for (int i = p; i <= q; ++i) {
        if (bra.guga_mode == 2 && i > p && i < q &&
            bra.next_single[i] > i && ket.next_single[i] > i) {
            int stop = std::min({q,bra.next_single[i],ket.next_single[i]});
            if (bra.label.compare(i,stop-i,ket.label,i,stop-i) ||
                std::abs(bra.spins.get<Checkpoint>(i+1)-ket.spins.get<Checkpoint>(i+1)) != 1) return 0.0;
            if ((ket.closed_prefix[stop]-ket.closed_prefix[i])%2) value = -value;
            i = stop-1;
            continue;
        }
        value *= segment(step(bra.label[i]), step(ket.label[i]), bra.spins.get<Checkpoint>(i+1), ket.spins.get<Checkpoint>(i+1), p, q, i,bra.guga_mode);
        if (value == 0.0) break;
    }
    return value;
}

template<bool Checkpoint>
double product(const Path &bra, const Path &ket, int p, int q, int r, int s) {
    if (r == s) return occupation(ket.label[s])*generator<Checkpoint>(bra, ket, p, q);
    if (p == q) return occupation(bra.label[p])*generator<Checkpoint>(bra, ket, r, s);
    if (!occupation(ket.label[s]) || occupation(ket.label[r]) == 2) return 0.0;
    int begin = std::min({p,q,r,s}), end = std::max({p,q,r,s});
    // Outside both generators the two outer walks and the intermediate must
    // coincide. Reject once, then omit their exact identity transfers.
    if (bra.label.compare(0, begin, ket.label, 0, begin) ||
        bra.label.compare(end+1, std::string::npos, ket.label, end+1, std::string::npos) ||
        bra.spins.get<Checkpoint>(end+1) != ket.spins.get<Checkpoint>(end+1)) return 0.0;
    std::array<int, 2> spins{ket.spins.get<Checkpoint>(begin), -1};
    std::array<double, 2> weights{1.0, 0.0};
    int count = 1;
    int lo = std::min(r, s), hi = std::max(r, s);
    for (int i = begin; i <= end; ++i) {
        int b = ket.spins.get<Checkpoint>(i+1);
        bool inside = static_cast<int>(i) >= lo && static_cast<int>(i) < hi;
        std::array<int, 2> after{inside ? b-1 : b, inside ? b+1 : -1};
        std::array<double, 2> next{0.0, 0.0};
        int n = occupation(ket.label[i])+(static_cast<int>(i) == r)-(static_cast<int>(i) == s);
        for (int j = 0; j < (inside ? 2 : 1); ++j) {
            if (after[j] < 0) continue;
            for (int t = 0; t < count; ++t) {
                if (weights[t] == 0.0) continue;
                int change = after[j]-spins[t], d;
                if (n == 1 && std::abs(change) == 1) d = change == 1 ? 1 : 2;
                else if (n != 1 && change == 0) d = n == 2 ? 3 : 0;
                else continue;
                double right = segment(d, step(ket.label[i]), after[j], b, r, s, i,bra.guga_mode);
                if (right == 0.0) continue;
                double left = segment(step(bra.label[i]), d, bra.spins.get<Checkpoint>(i+1), after[j], p, q, i,bra.guga_mode);
                next[j] += weights[t]*right*left;
            }
        }
        count = inside ? 2 : 1;
        spins = after;
        weights = next;
        if (weights[0] == 0.0 && weights[1] == 0.0) return 0.0;
        if (bra.guga_mode == 2 && i != p && i != q && i != r && i != s &&
            bra.next_single[i] > i && ket.next_single[i] > i) {
            int stop = std::min({end+1,bra.next_single[i],ket.next_single[i]});
            for (int endpoint : {p,q,r,s}) if (endpoint > i) stop = std::min(stop,endpoint);
            if (bra.label.compare(i,stop-i,ket.label,i,stop-i)) return 0.0;
            int active = (std::min(p,q) < i && i < std::max(p,q))+
                         (std::min(r,s) < i && i < std::max(r,s));
            if (active%2 && (ket.closed_prefix[stop]-ket.closed_prefix[i+1])%2)
                for (double &weight : weights) weight = -weight;
            i = stop-1;
        }
    }
    return weights[0];
}
} // namespace

template<bool Checkpoint>
double guga_generator_impl(const Path &bra, const Path &ket, int p, int q) {
    return bra.phase*ket.phase*generator<Checkpoint>(bra, ket, p, q);
}

double guga_segment(int da, int db, int a, int b, int p, int q, int site, int mode) {
    return segment(da,db,a,b,p,q,site,mode);
}

template<bool Checkpoint>
double guga_two_body_impl(const Path &bra, const Path &ket, int p, int q, int r, int s) {
    return bra.phase*ket.phase*(product<Checkpoint>(bra, ket, p, q, r, s)-
                               (q == r ? generator<Checkpoint>(bra, ket, p, s) : 0.0));
}

template<bool Checkpoint>
double guga_element_impl(const Path &bra, const Path &ket, const Connection &connection) {
    double value = bra.label == ket.label ? connection.scalar : 0.0;
    for (const auto &term : connection.generators) {
        double coupling = term.two_body ? guga_two_body_impl<Checkpoint>(bra, ket, term.p, term.q, term.r, term.s) :
                                         guga_generator_impl<Checkpoint>(bra, ket, term.p, term.q);
        value += term.coefficient*coupling;
    }
    return value;
}

double guga_generator(const Path &bra,const Path &ket,int p,int q) {
    return bra.spins.is_compressed() || ket.spins.is_compressed() ?
        guga_generator_impl<true>(bra,ket,p,q) : guga_generator_impl<false>(bra,ket,p,q);
}
double guga_two_body(const Path &bra,const Path &ket,int p,int q,int r,int s) {
    return bra.spins.is_compressed() || ket.spins.is_compressed() ?
        guga_two_body_impl<true>(bra,ket,p,q,r,s) : guga_two_body_impl<false>(bra,ket,p,q,r,s);
}
double guga_element(const Path &bra,const Path &ket,const Connection &c) {
    return bra.spins.is_compressed() || ket.spins.is_compressed() ?
        guga_element_impl<true>(bra,ket,c) : guga_element_impl<false>(bra,ket,c);
}

GugaMemo::GugaMemo(size_t maximum, bool reuse, bool packed,Format format,bool collect_spin,
                   bool align_tiny,bool pair_encode)
    : maximum_(maximum), reuse_(reuse), packed_(packed),
      align_tiny_(align_tiny),pair_encode_(pair_encode),format_(format),collect_spin_(collect_spin) {
    if (format_!=Generic && packed_) {
        static_assert(sizeof(LeanSlot)==24 && sizeof(TinySlot)==16,"narrow cache layout");
        size_t count=maximum/((format_==Tiny) ? sizeof(TinySlot) : sizeof(LeanSlot));
        if (count>1) count-=count%2;
        if (format_==Tiny) {
            if (align_tiny_ && count) {
                auto *slots=static_cast<TinySlot *>(::operator new(count*sizeof(TinySlot),std::align_val_t(64)));
                aligned_tiny_.reset(slots);
                std::uninitialized_value_construct_n(slots,count);
            } else tiny_slots_.resize(count);
            tiny_count_=count;tiny_data_=aligned_tiny_ ? aligned_tiny_.get() : tiny_slots_.data();
            bytes_=count*sizeof(TinySlot);
        }
        else { lean_slots_.resize(count); bytes_=count*sizeof(LeanSlot); }
    } else if (packed) {
        size_t count = maximum/sizeof(Slot), power = count ? 1 : 0;
        while (power && power <= count/2) power *= 2;
        slots_.resize(power);
        bytes_ = slots_.size()*sizeof(Slot);
    }
}

namespace {
std::uint64_t plan_mix(std::uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x^(x >> 31);
}
}

void prepare_guga_plan(const Config &bra,const Config &ket,const Connection &c,GugaPlan &plan) {
    plan.resize(c.generators.size());
    for (size_t t=0;t<c.generators.size();++t) {
        const auto &term=c.generators[t];
        auto &p=plan[t]; p=GugaPreparedTerm{};
        p.begin=std::min(term.p,term.q); p.end=std::max(term.p,term.q);
        if (term.two_body) {
            p.begin=std::min({p.begin,term.r,term.s});
            p.end=std::max({p.end,term.r,term.s});
        }
        p.meta[0]=term.two_body;
        int kept=0;
        for (int i=p.begin;i<=p.end;++i) {
            bool endpoint=i==term.p || i==term.q || (term.two_body && (i==term.r || i==term.s));
            if (!endpoint && bra[i]==ket[i] && bra[i]!=1) {
                int active=std::min(term.p,term.q)<i && i<std::max(term.p,term.q);
                if (term.two_body) active+=std::min(term.r,term.s)<i && i<std::max(term.r,term.s);
                if (ket[i]==2 && active%2) p.parity=-p.parity;
                continue;
            }
            if (kept==32) { p.fits=false; break; }
            if (i==term.p) p.meta[4]=kept;
            if (i==term.q) p.meta[5]=kept;
            if (term.two_body && i==term.r) p.meta[6]=kept;
            if (term.two_body && i==term.s) p.meta[7]=kept;
            p.sites[kept++]=i;
        }
        p.meta[3]=kept;
        // Boundary spins and step words vary between path pairs; all other
        // key metadata is fixed. Hash equality is never used as identity.
        p.hash=0x9e3779b97f4a7c15ULL;
        for (int i : {0,3,4,5,6,7}) p.hash=plan_mix(p.hash^std::uint32_t(p.meta[i]));
    }
}

template<bool Checkpoint>
double GugaMemo::planned_element_impl(const Path &bra,const Path &ket,const Connection &c,const GugaPlan &plan) {
    if (!packed_) return element_impl<Checkpoint>(bra,ket,c);
    double value=bra.label==ket.label ? c.scalar : 0.;
    size_t first=0,last=bra.label.size();
    while (first<last && bra.label[first]==ket.label[first]) ++first;
    while (last>first && bra.label[last-1]==ket.label[last-1]) --last;
    for (size_t t=0;t<c.generators.size();++t) {
        const auto &term=c.generators[t]; const auto &p=plan[t];
        if ((first!=last && (size_t(p.begin)>first || size_t(p.end+1)<last)) ||
            bra.spins.get<Checkpoint>(p.end+1)!=ket.spins.get<Checkpoint>(p.end+1)) {
            ++orthogonal_; value+=term.coefficient*(bra.phase*ket.phase*0.); continue;
        }
        PackedKey key; key.meta=p.meta;
        key.meta[1]=bra.spins.get<Checkpoint>(p.begin); key.meta[2]=ket.spins.get<Checkpoint>(p.begin);
        Slot *slot=nullptr;
        if (p.fits && !slots_.empty()) {
            for (int at=0;at<p.meta[3];++at) {
                int i=p.sites[at];
                key.steps[at/16]|=std::uint64_t(4*step(bra.label[i])+step(ket.label[i]))<<(4*(at%16));
            }
            key_sites_+=p.meta[3];
            auto hash=plan_mix(p.hash^key.steps[0]);
            hash=plan_mix(hash^key.steps[1]);
            hash=plan_mix(hash^(std::uint64_t(std::uint32_t(key.meta[1]))<<32)^std::uint32_t(key.meta[2]));
            slot=&slots_[hash&(slots_.size()-1)];
        }
        int phase=bra.phase*ket.phase*p.parity;
        bool hit=slot && slot->uses && slot->key==key;
        if (hit) { ++hits_; ++slot->uses; } else ++misses_;
        double coupling;
        if (hit && reuse_) coupling=phase*slot->value;
        else {
            coupling=term.two_body ? guga_two_body_impl<Checkpoint>(bra,ket,term.p,term.q,term.r,term.s) :
                                     guga_generator_impl<Checkpoint>(bra,ket,term.p,term.q);
            if (slot && !hit) {
                replacements_+=slot->uses!=0;
                slot->key=key; slot->value=phase*coupling; slot->uses=1;
            } else if (!slot) ++bypass_;
        }
        value+=term.coefficient*coupling;
    }
    return value;
}

namespace {
using BitWords=std::array<std::uint64_t,2>;
std::uint64_t spread_nibbles(std::uint64_t x) {
    x=(x|(x<<24))&0x000000ff000000ffULL;
    x=(x|(x<<12))&0x000f000f000f000fULL;
    x=(x|(x<<6))&0x0303030303030303ULL;
    return (x|(x<<3))&0x1111111111111111ULL;
}
BitWords encode_portable(std::uint64_t down,const GugaBitTerm &p,bool bra) {
    std::uint64_t compressed=0,mask=p.keep;
    int bit=0;
    while (mask) {
        auto lowest=mask&(~mask+1);
        if (down&lowest) compressed|=std::uint64_t(1)<<bit;
        mask^=lowest; ++bit;
    }
    const auto &base=bra ? p.bra : p.ket;
    unsigned factor=bra ? 12 : 3;
    return {base[0]^(spread_nibbles(compressed&0xffff)*factor),
            base[1]^(spread_nibbles((compressed>>16)&0xffff)*factor)};
}
#ifdef SCSF_GUGA_BMI2
__attribute__((target("bmi2"))) BitWords encode_bmi2(std::uint64_t down,const GugaBitTerm &p,bool bra) {
    auto compressed=_pext_u64(down,p.keep);
    const auto &base=bra ? p.bra : p.ket;
    unsigned factor=bra ? 12 : 3;
    return {base[0]^(_pdep_u64(compressed&0xffff,0x1111111111111111ULL)*factor),
            base[1]^(_pdep_u64(compressed>>16,0x1111111111111111ULL)*factor)};
}
#endif
auto bit_encoder(bool accelerated) -> BitWords (*)(std::uint64_t,const GugaBitTerm &,bool) {
#ifdef SCSF_GUGA_BMI2
    if (accelerated && guga_bmi2_available()) return &encode_bmi2;
#else
    (void)accelerated;
#endif
    return &encode_portable;
}
}

bool guga_bmi2_available() {
#ifdef SCSF_GUGA_BMI2
    static const bool available=__builtin_cpu_supports("bmi2");
    return available;
#else
    return false;
#endif
}

void prepare_guga_bits(const Config &bra,const Config &ket,const GugaPlan &plan,GugaBitPlan &bits) {
    bits.resize(plan.size());
    std::uint64_t different=0;
    if (bra.size()<=64) for (int i=0;i<int(bra.size());++i)
        if (bra[i]!=ket[i]) different|=std::uint64_t(1)<<i;
    for (size_t t=0;t<plan.size();++t) {
        const auto &p=plan[t]; auto &b=bits[t]; b=GugaBitTerm{};
        b.compact_meta=std::uint64_t(p.meta[0])|(std::uint64_t(p.meta[3])<<13)|
            (std::uint64_t(p.meta[4])<<18)|(std::uint64_t(p.meta[5])<<22)|
            (std::uint64_t(p.meta[6])<<26)|(std::uint64_t(p.meta[7])<<30);
        if (!p.fits || bra.size()>64) { b.compatible=false; continue; }
        auto before=p.begin ? (std::uint64_t(1)<<p.begin)-1 : 0;
        auto after=p.end==63 ? 0 : ~((std::uint64_t(1)<<(p.end+1))-1);
        b.outside=before|after;
        b.compatible=(different&b.outside)==0;
        for (int at=0;at<p.meta[3];++at) {
            int i=p.sites[at]; b.keep|=std::uint64_t(1)<<i;
            unsigned da=bra[i]==2 ? 3 : bra[i], db=ket[i]==2 ? 3 : ket[i];
            b.bra[at/16]|=std::uint64_t(da)<<(4*(at%16)+2);
            b.ket[at/16]|=std::uint64_t(db)<<(4*(at%16));
        }
    }
}

void prepare_guga_bit_connection(const Config &bra,const Config &ket,const Connection &c,
                                 GugaPlan &plan,GugaBitPlan &bits,bool lean) {
    if (bra.size()>64) { prepare_guga_plan(bra,ket,c,plan); bits.clear(); return; }
    plan.resize(c.generators.size()); bits.resize(c.generators.size());
    std::uint64_t different=0;
    for (int i=0;i<int(bra.size());++i) if (bra[i]!=ket[i]) different|=std::uint64_t(1)<<i;
    for (size_t t=0;t<c.generators.size();++t) {
        const auto &term=c.generators[t]; auto &p=plan[t]; auto &b=bits[t];
        p.meta={}; p.parity=1; p.fits=true; b=GugaBitTerm{};
        p.begin=std::min(term.p,term.q); p.end=std::max(term.p,term.q);
        if (term.two_body) { p.begin=std::min({p.begin,term.r,term.s}); p.end=std::max({p.end,term.r,term.s}); }
        p.meta[0]=term.two_body;
        b.outside=(p.begin ? (std::uint64_t(1)<<p.begin)-1 : 0)|
            (p.end==63 ? 0 : ~((std::uint64_t(1)<<(p.end+1))-1));
        b.compatible=(different&b.outside)==0;
        int kept=0;
        for (int i=p.begin;i<=p.end;++i) {
            bool endpoint=i==term.p || i==term.q || (term.two_body && (i==term.r || i==term.s));
            if (!endpoint && bra[i]==ket[i] && bra[i]!=1) {
                int active=std::min(term.p,term.q)<i && i<std::max(term.p,term.q);
                if (term.two_body) active+=std::min(term.r,term.s)<i && i<std::max(term.r,term.s);
                if (ket[i]==2 && active%2) p.parity=-p.parity;
                continue;
            }
            if (kept==32) { p.fits=false; break; }
            p.sites[kept]=i; b.keep|=std::uint64_t(1)<<i;
            if (i==term.p) p.meta[4]=kept;
            if (i==term.q) p.meta[5]=kept;
            if (term.two_body && i==term.r) p.meta[6]=kept;
            if (term.two_body && i==term.s) p.meta[7]=kept;
            unsigned da=bra[i]==2 ? 3 : bra[i],db=ket[i]==2 ? 3 : ket[i];
            b.bra[kept/16]|=std::uint64_t(da)<<(4*(kept%16)+2);
            b.ket[kept/16]|=std::uint64_t(db)<<(4*(kept%16)); ++kept;
        }
        p.meta[3]=kept; p.hash=0x9e3779b97f4a7c15ULL;
        for (int i:{0,3,4,5,6,7}) p.hash=plan_mix(p.hash^std::uint32_t(p.meta[i]));
        b.compact_meta=std::uint64_t(p.meta[0])|(std::uint64_t(kept)<<13)|
            (std::uint64_t(p.meta[4])<<18)|(std::uint64_t(p.meta[5])<<22)|
            (std::uint64_t(p.meta[6])<<26)|(std::uint64_t(p.meta[7])<<30);
        if (lean) b.lean_meta=std::uint32_t(p.meta[0])|(std::uint32_t(kept)<<9)|
            (std::uint32_t(p.meta[4])<<14)|(std::uint32_t(p.meta[5])<<18)|
            (std::uint32_t(p.meta[6])<<22)|(std::uint32_t(p.meta[7])<<26);
    }
}

namespace {
// Operator endpoints include both ends of the retained interval. Store the
// first minimum/maximum positions plus the remaining two endpoint values.
std::uint32_t endpoint_code(const std::array<int,4> &ends,int kept) {
    if (kept==1) return 0;
    int lo=0,hi=0;while (ends[lo]!=0) ++lo;while (ends[hi]!=kept-1) ++hi;
    std::uint32_t code=lo|(hi<<2);int shift=4;
    for (int i=0;i<4;++i) if (i!=lo && i!=hi) { code|=ends[i]<<shift;shift+=4; }
    return code;
}
std::uint64_t wide_tiny_key(std::uint64_t steps,int kept,std::uint32_t static_meta,int bra,int ket) {
    int width=kept==10 ? 2 : 4;
    auto payload=std::uint64_t(static_meta)|(std::uint64_t(bra)<<13)|(std::uint64_t(ket)<<(13+width));
    return std::uint64_t(kept)|(steps<<4)|(payload<<(4+4*kept));
}
}

void prepare_guga_lean_connection(const Config &bra,const Config &ket,const Connection &c,
                                  GugaLeanPlan &plan,bool tiny) {
    static_assert(sizeof(GugaLeanTerm)==48,"lean preparation layout");
    if (bra.size()>64) { plan.clear();return; }
    plan.resize(c.generators.size());std::uint64_t different=0;
    for (int i=0;i<int(bra.size());++i) if (bra[i]!=ket[i]) different|=std::uint64_t(1)<<i;
    for (size_t t=0;t<c.generators.size();++t) {
        const auto &term=c.generators[t];auto &p=plan[t];p=GugaLeanTerm{};
        p.begin=std::min(term.p,term.q);p.end=std::max(term.p,term.q);
        if (term.two_body) { p.begin=std::min({int(p.begin),term.r,term.s});p.end=std::max({int(p.end),term.r,term.s}); }
        p.outside=(p.begin ? (std::uint64_t(1)<<p.begin)-1 : 0)|
            (p.end==63 ? 0 : ~((std::uint64_t(1)<<(p.end+1))-1));
        p.compatible=(different&p.outside)==0;
        std::array<int,4> ends{};int kept=0;
        for (int i=p.begin;i<=p.end;++i) {
            bool endpoint=i==term.p || i==term.q || (term.two_body && (i==term.r || i==term.s));
            if (!endpoint && bra[i]==ket[i] && bra[i]!=1) {
                int active=std::min(term.p,term.q)<i && i<std::max(term.p,term.q);
                if (term.two_body) active+=std::min(term.r,term.s)<i && i<std::max(term.r,term.s);
                if (ket[i]==2 && active%2) p.parity=-p.parity;
                continue;
            }
            if (kept==16) { p.fits=false;break; }
            p.keep|=std::uint64_t(1)<<i;
            if (i==term.p) ends[0]=kept;
            if (i==term.q) ends[1]=kept;
            if (term.two_body && i==term.r) ends[2]=kept;
            if (term.two_body && i==term.s) ends[3]=kept;
            unsigned da=bra[i]==2 ? 3 : bra[i],db=ket[i]==2 ? 3 : ket[i];
            p.bra|=std::uint64_t(da)<<(4*kept+2);p.ket|=std::uint64_t(db)<<(4*kept);++kept;
        }
        p.kept=kept;
        if (!p.fits) continue;
        p.meta=std::uint32_t(term.two_body)|(std::uint32_t(kept)<<9)|
            (std::uint32_t(ends[0])<<14)|(std::uint32_t(ends[1])<<18)|
            (std::uint32_t(ends[2])<<22)|(std::uint32_t(ends[3])<<26);
        if (tiny && kept<=10) p.tiny_meta=std::uint32_t(term.two_body)|(endpoint_code(ends,kept)<<1);
    }
}

// Immutable occupation masks remove repeated spectator scans. Physical site
// order, closed-spectator phase and compact endpoint encoding are unchanged.
void prepare_guga_mask_impl(const Connection &c,GugaLeanPlan &plan,bool tiny,
    std::uint64_t a1,std::uint64_t a2,std::uint64_t b1,std::uint64_t b2) {
    auto below=[](int p) { return p==64 ? ~std::uint64_t(0) : (std::uint64_t(1)<<p)-1; };
    auto inside=[&](int p,int q) {
        int low=std::min(p,q),high=std::max(p,q);
        return high>low ? below(high)&~below(low+1) : std::uint64_t(0);
    };
    const auto different=(a1^b1)|(a2^b2);
    plan.resize(c.generators.size());
    for (size_t t=0;t<c.generators.size();++t) {
        const auto &term=c.generators[t];auto &p=plan[t];p=GugaLeanTerm{};
        p.begin=std::min(term.p,term.q);p.end=std::max(term.p,term.q);
        auto endpoints=(std::uint64_t(1)<<term.p)|(std::uint64_t(1)<<term.q);
        auto odd=inside(term.p,term.q);
        if (term.two_body) {
            p.begin=std::min({int(p.begin),term.r,term.s});p.end=std::max({int(p.end),term.r,term.s});
            endpoints|=(std::uint64_t(1)<<term.r)|(std::uint64_t(1)<<term.s);
            odd^=inside(term.r,term.s);
        }

        auto interval=below(p.end+1)&~below(p.begin);
        p.outside=~interval;p.compatible=(different&p.outside)==0;
        auto kept=(a1|b1|different|endpoints)&interval;
        auto closed=a2&b2&~endpoints&odd;
        // Parity needs only odd/even, avoiding an ISA-specific population count.
        p.parity=__builtin_parityll(closed) ? -1 : 1;
        std::array<int,4> ends{};int n=0;
        while (kept) {
            if (n==16) {
                p.fits=false;
                p.parity=__builtin_parityll(closed&below(__builtin_ctzll(kept))) ? -1 : 1;
                break;
            }
            int site=__builtin_ctzll(kept);auto bit=std::uint64_t(1)<<site;kept&=kept-1;
            p.keep|=bit;
            if (site==term.p) ends[0]=n;
            if (site==term.q) ends[1]=n;
            if (term.two_body && site==term.r) ends[2]=n;
            if (term.two_body && site==term.s) ends[3]=n;
            unsigned da=(a2&bit) ? 3 : (a1&bit) ? 1 : 0;
            unsigned db=(b2&bit) ? 3 : (b1&bit) ? 1 : 0;
            p.bra|=std::uint64_t(da)<<(4*n+2);p.ket|=std::uint64_t(db)<<(4*n);++n;
        }
        p.kept=n;
        if (!p.fits) continue;
        p.meta=std::uint32_t(term.two_body)|(std::uint32_t(n)<<9)|
            (std::uint32_t(ends[0])<<14)|(std::uint32_t(ends[1])<<18)|
            (std::uint32_t(ends[2])<<22)|(std::uint32_t(ends[3])<<26);
        if (tiny && n<=10) p.tiny_meta=std::uint32_t(term.two_body)|(endpoint_code(ends,n)<<1);
    }
}

void prepare_guga_mask_connection(const Connection &c,GugaLeanPlan &plan,bool tiny,
    std::uint64_t a1,std::uint64_t a2,std::uint64_t b1,std::uint64_t b2) {
    prepare_guga_mask_impl(c,plan,tiny,a1,a2,b1,b2);
}
// Adjacent permutations share occupations and retained sites. Only endpoint
// roles and closed-spectator phase change; no integral data is retained.


namespace {
using LeanEncoder=std::uint64_t (*)(std::uint64_t,const GugaLeanTerm &,bool);
std::uint64_t lean_portable(std::uint64_t down,const GugaLeanTerm &p,bool bra) {
    std::uint64_t x=0,mask=p.keep;int bit=0;
    while (mask) { auto low=mask&(~mask+1);if (down&low) x|=std::uint64_t(1)<<bit;mask^=low;++bit; }
    return (bra ? p.bra : p.ket)^(spread_nibbles(x&0xffff)*(bra ? 12 : 3));
}
#ifdef SCSF_GUGA_BMI2
__attribute__((target("bmi2"))) std::uint64_t lean_bmi2(std::uint64_t down,const GugaLeanTerm &p,bool bra) {
    return (bra ? p.bra : p.ket)^(_pdep_u64(_pext_u64(down,p.keep)&0xffff,0x1111111111111111ULL)*(bra ? 12 : 3));
}
#endif
LeanEncoder lean_encoder(bool accelerated) {
#ifdef SCSF_GUGA_BMI2
    if (accelerated && guga_bmi2_available()) return &lean_bmi2;
#else
    (void)accelerated;
#endif
    return &lean_portable;
}
using PairEncoder=std::uint64_t (*)(std::uint64_t,std::uint64_t,const GugaLeanTerm &);
std::uint64_t lean_pair_portable(std::uint64_t a,std::uint64_t b,const GugaLeanTerm &p) {
    std::uint64_t x=0,y=0,mask=p.keep;int bit=0;
    while (mask) {
        auto low=mask&(~mask+1);
        if (a&low) x|=std::uint64_t(1)<<bit;
        if (b&low) y|=std::uint64_t(1)<<bit;
        mask^=low;++bit;
    }
    return (p.bra|p.ket)^((spread_nibbles(x&0xffff)*12)|(spread_nibbles(y&0xffff)*3));
}
#ifdef SCSF_GUGA_BMI2
__attribute__((target("bmi2"))) std::uint64_t lean_pair_bmi2(std::uint64_t a,std::uint64_t b,const GugaLeanTerm &p) {
    auto x=_pdep_u64(_pext_u64(a,p.keep)&0xffff,0x1111111111111111ULL);
    auto y=_pdep_u64(_pext_u64(b,p.keep)&0xffff,0x1111111111111111ULL);
    return (p.bra|p.ket)^((x*12)|(y*3));
}
#endif
PairEncoder lean_pair_encoder(bool accelerated) {
#ifdef SCSF_GUGA_BMI2
    if (accelerated && guga_bmi2_available()) return &lean_pair_bmi2;
#else
    (void)accelerated;
#endif
    return &lean_pair_portable;
}
}

namespace {
using ShortEncoder=std::uint64_t (*)(std::uint64_t,const GugaBitTerm &,bool);
std::uint64_t short_portable(std::uint64_t down,const GugaBitTerm &p,bool bra) {
    std::uint64_t x=0,mask=p.keep;int bit=0;
    while (mask) { auto low=mask&(~mask+1); if (down&low) x|=std::uint64_t(1)<<bit;mask^=low;++bit; }
    return (bra ? p.bra[0] : p.ket[0])^(spread_nibbles(x&0xffff)*(bra ? 12 : 3));
}
#ifdef SCSF_GUGA_BMI2
__attribute__((target("bmi2"))) std::uint64_t short_bmi2(std::uint64_t down,const GugaBitTerm &p,bool bra) {
    return (bra ? p.bra[0] : p.ket[0])^(_pdep_u64(_pext_u64(down,p.keep)&0xffff,0x1111111111111111ULL)*(bra ? 12 : 3));
}
#endif
ShortEncoder short_encoder(bool accelerated) {
#ifdef SCSF_GUGA_BMI2
    if (accelerated && guga_bmi2_available()) return &short_bmi2;
#else
    (void)accelerated;
#endif
    return &short_portable;
}
size_t hash_range(std::uint64_t h,size_t count) {
#ifdef __SIZEOF_INT128__
    return size_t((static_cast<unsigned __int128>(h)*count)>>64);
#else
    return h%count;
#endif
}
}

// Exact physical two-state recurrence, with only matching outer-path prefixes
// reused within one generator's temporary batch. No selected-space projector.


template<bool Track,bool Checkpoint,bool Compact,bool TinyOnly>
double GugaMemo::lean_lookup(const Path &bra,const Path &ket,const GugaTerm &term,
                             int begin,int parity,int kept,std::uint64_t steps,std::uint32_t meta,std::uint32_t tiny_meta,
                             int bra_begin,int ket_begin,int phase) {
    if constexpr(!Compact) {
        phase=bra.phase*ket.phase;
        bra_begin=bra.spins.get<Checkpoint>(begin);ket_begin=ket.spins.get<Checkpoint>(begin);
    }
    phase*=parity;
    auto evaluate=[&] { return term.two_body ? guga_two_body_impl<Checkpoint>(bra,ket,term.p,term.q,term.r,term.s) : guga_generator_impl<Checkpoint>(bra,ket,term.p,term.q); };
    meta|=(std::uint32_t(bra_begin)<<1)|(std::uint32_t(ket_begin)<<5);
    if (TinyOnly || format_==Tiny) {
        constexpr std::uint64_t valid=1ULL<<62,recent=1ULL<<63,keymask=valid-1;
        // Fixed low count nibble separates the variable-width exact formats.
        auto key=wide_tiny_key(steps,kept,tiny_meta,bra_begin,ket_begin);
        size_t count=tiny_count();
        if (!count) { if constexpr(Track) { ++misses_;++bypass_; } return evaluate(); }
        size_t base=count==1 ? 0 : 2*hash_range(plan_mix(key),count/2);
        auto &a=tiny_data()[base];auto &b=tiny_data()[base+(count>1)];TinySlot *slot=nullptr;
        if ((a.key&valid) && (a.key&keymask)==key) { slot=&a;if constexpr(Track) ++primary_hits_; }
        else if (count>1 && (b.key&valid) && (b.key&keymask)==key) { slot=&b;if constexpr(Track) ++secondary_hits_; }
        bool hit=slot;
        if constexpr(Track) { key_sites_+=kept;if (hit) ++hits_;else ++misses_; }
        if (!hit) {
            if (!(a.key&valid)) { slot=&a;if constexpr(Track) ++empty_misses_; }
            else if (count>1 && !(b.key&valid)) { slot=&b;if constexpr(Track) ++empty_misses_; }
            else { slot=(a.key&recent) ? &b : &a;if constexpr(Track) { ++full_misses_;++replacements_; } }
        }
        double value=hit && reuse_ ? phase*slot->value : evaluate();
        if (!hit) { slot->key=key|valid;slot->value=phase*value; }
        if (count>1) { if (slot==&a) a.key|=recent;else a.key&=~recent; }
        return value;
    }
    constexpr std::uint32_t valid=1u<<30,recent=1u<<31,keymask=valid-1;
    size_t count=lean_slots_.size();
    if (!count) { if constexpr(Track) { ++misses_;++bypass_; } return evaluate(); }
    auto hash=plan_mix(steps^(std::uint64_t(meta)*0x9e3779b97f4a7c15ULL));
    size_t base=count==1 ? 0 : 2*hash_range(hash,count/2);
    auto &a=lean_slots_[base];auto &b=lean_slots_[base+(count>1)];LeanSlot *slot=nullptr;
    if ((a.meta&valid) && a.steps==steps && (a.meta&keymask)==meta) { slot=&a;if constexpr(Track) ++primary_hits_; }
    else if (count>1 && (b.meta&valid) && b.steps==steps && (b.meta&keymask)==meta) { slot=&b;if constexpr(Track) ++secondary_hits_; }
    bool hit=slot;
    if constexpr(Track) { key_sites_+=kept;if (hit) ++hits_;else ++misses_; }
    if (!hit) {
        if (!(a.meta&valid)) { slot=&a;if constexpr(Track) ++empty_misses_; }
        else if (count>1 && !(b.meta&valid)) { slot=&b;if constexpr(Track) ++empty_misses_; }
        else { slot=(a.meta&recent) ? &b : &a;if constexpr(Track) { ++full_misses_;++replacements_; } }
    }
    double value=hit && reuse_ ? phase*slot->value : evaluate();
    if (!hit) { slot->steps=steps;slot->meta=meta|valid;slot->value=phase*value;slot->uses=0; }
    if constexpr(Track) if (slot->uses<std::numeric_limits<std::uint32_t>::max()) ++slot->uses;
    if (count>1) { if (slot==&a) a.meta|=recent;else a.meta&=~recent; }
    return value;
}

template<bool Track,bool Checkpoint>
double GugaMemo::lean_element(const Path &bra,const Path &ket,const Connection &c,
    const GugaPlan &plan,const GugaBitPlan &bits,std::uint64_t bra_down,std::uint64_t ket_down,bool accelerated) {
    if (!packed_ || bra.label.size()>64) {
        if constexpr(Track) { misses_+=c.generators.size();bypass_+=c.generators.size(); }
        return guga_element_impl<Checkpoint>(bra,ket,c);
    }
    auto encode=short_encoder(accelerated);double value=bra.label==ket.label ? c.scalar : 0.;
    for (size_t t=0;t<c.generators.size();++t) {
        const auto &term=c.generators[t];const auto &p=plan[t];const auto &b=bits[t];
        if (!p.fits || p.meta[3]>16 || bra.spins.get<Checkpoint>(p.begin)>=16 || ket.spins.get<Checkpoint>(p.begin)>=16) {
            if constexpr(Track) { ++misses_;++bypass_; }
            value+=term.coefficient*(term.two_body ? guga_two_body_impl<Checkpoint>(bra,ket,term.p,term.q,term.r,term.s) : guga_generator_impl<Checkpoint>(bra,ket,term.p,term.q));continue;
        }
        if (!b.compatible || ((bra_down^ket_down)&b.outside) || bra.spins.get<Checkpoint>(p.end+1)!=ket.spins.get<Checkpoint>(p.end+1)) {
            if constexpr(Track) { ++orthogonal_; }
            continue;
        }
        value+=term.coefficient*lean_lookup<Track,Checkpoint>(bra,ket,term,p.begin,p.parity,p.meta[3],encode(bra_down,b,true)|encode(ket_down,b,false),b.lean_meta);
    }
    return value;
}

template<bool Track,bool Checkpoint>
bool GugaMemo::lean_batch(const std::vector<Path> &paths,IndexView rows,
    IndexView cols,const std::vector<std::uint64_t> &down,const std::vector<double> &active,
    bool same,const Connection &c,const GugaPlan &plan,const GugaBitPlan &bits,bool accelerated,size_t maximum,GugaBatchScratch &scratch) {
    if (!packed_ || rows.empty() || cols.empty() || down.empty() || rows.size()>maximum/cols.size()) return false;
    for (const auto &p:plan) if (!p.fits || p.meta[3]>16) return false;
    auto &out=scratch.values;out.assign(rows.size()*cols.size(),0.);auto encode=short_encoder(accelerated);
    auto &a=scratch.bra;auto &b=scratch.ket;a.resize(rows.size());b.resize(cols.size());
    for (size_t t=0;t<c.generators.size();++t) {
        const auto &term=c.generators[t];const auto &p=plan[t];const auto &bp=bits[t];
        if (!bp.compatible) continue;
        for (size_t r=0;r<rows.size();++r) a[r][0]=encode(down[rows[r]],bp,true);
        for (size_t s=0;s<cols.size();++s) b[s][0]=encode(down[cols[s]],bp,false);
        for (size_t r=0;r<rows.size();++r) for (size_t s=0;s<cols.size();++s) {
            size_t i=rows[r],j=cols[s];if (i==j || (same && j>i) || (active[i]==0. && active[j]==0.)) continue;
            const auto &bra=paths[i];const auto &ket=paths[j];
            if (((down[i]^down[j])&bp.outside) || bra.spins.get<Checkpoint>(p.end+1)!=ket.spins.get<Checkpoint>(p.end+1)) {
                if constexpr(Track) { ++orthogonal_; }
            continue;
            }
            double value;
            if (bra.spins.get<Checkpoint>(p.begin)>=16 || ket.spins.get<Checkpoint>(p.begin)>=16) {
                if constexpr(Track) { ++misses_;++bypass_; }
                value=term.two_body ? guga_two_body_impl<Checkpoint>(bra,ket,term.p,term.q,term.r,term.s) : guga_generator_impl<Checkpoint>(bra,ket,term.p,term.q);
            } else value=lean_lookup<Track,Checkpoint>(bra,ket,term,p.begin,p.parity,p.meta[3],a[r][0]|b[s][0],bp.lean_meta);
            out[r*cols.size()+s]+=term.coefficient*value;
        }
    }
    return true;
}

template<bool Track,bool Checkpoint>
double GugaMemo::small_element_impl(const Path &bra,const Path &ket,const Connection &c,
    const GugaLeanPlan &plan,std::uint64_t bra_down,std::uint64_t ket_down,bool accelerated) {
    if (!packed_ || bra.label.size()>64) {
        if constexpr(Track) { misses_+=c.generators.size();bypass_+=c.generators.size(); }
        return guga_element_impl<Checkpoint>(bra,ket,c);
    }
    auto encode=pair_encode_ ? nullptr : lean_encoder(accelerated);
    auto pair=pair_encode_ ? lean_pair_encoder(accelerated) : nullptr;
    double value=bra.label==ket.label ? c.scalar : 0.;
    for (size_t t=0;t<c.generators.size();++t) {
        const auto &term=c.generators[t];const auto &p=plan[t];
        if (!p.compatible || ((bra_down^ket_down)&p.outside) || bra.spins.get<Checkpoint>(p.end+1)!=ket.spins.get<Checkpoint>(p.end+1)) {
            if constexpr(Track) ++orthogonal_;
            continue;
        }
        int boundary=format_==Tiny && p.kept==10 ? 4 : 16;
        if (!p.fits || (format_==Tiny && p.kept>10) || bra.spins.get<Checkpoint>(p.begin)>=boundary || ket.spins.get<Checkpoint>(p.begin)>=boundary) {
            if constexpr(Track) { ++misses_;++bypass_; }
            value+=term.coefficient*(term.two_body ? guga_two_body_impl<Checkpoint>(bra,ket,term.p,term.q,term.r,term.s) : guga_generator_impl<Checkpoint>(bra,ket,term.p,term.q));continue;
        }
        value+=term.coefficient*lean_lookup<Track,Checkpoint>(bra,ket,term,p.begin,p.parity,p.kept,
            pair ? pair(bra_down,ket_down,p) : encode(bra_down,p,true)|encode(ket_down,p,false),p.meta,p.tiny_meta);
    }
    return value;
}

template<bool Track,bool Checkpoint>
bool GugaMemo::small_batch_impl(const std::vector<Path> &paths,IndexView rows,
    IndexView cols,const std::vector<std::uint64_t> &down,const std::vector<double> &active,
    bool same,const Connection &c,const GugaLeanPlan &plan,bool accelerated,size_t maximum,GugaBatchScratch &scratch) {
    if (!packed_ || rows.empty() || cols.empty() || down.empty() || rows.size()>maximum/cols.size()) return false;
    auto &out=scratch.values;out.assign(rows.size()*cols.size(),0.);auto encode=lean_encoder(accelerated);
    auto &a=scratch.bra;auto &b=scratch.ket;a.resize(rows.size());b.resize(cols.size());
    for (size_t t=0;t<c.generators.size();++t) {
        const auto &term=c.generators[t];const auto &p=plan[t];if (!p.compatible) continue;
        bool supported=p.fits && (format_!=Tiny || p.kept<=10);
        if (supported) {
            for (size_t r=0;r<rows.size();++r) a[r][0]=encode(down[rows[r]],p,true);
            for (size_t s=0;s<cols.size();++s) b[s][0]=encode(down[cols[s]],p,false);
        }
        int boundary=format_==Tiny && p.kept==10 ? 4 : 16;
        for (size_t r=0;r<rows.size();++r) for (size_t s=0;s<cols.size();++s) {
            size_t i=rows[r],j=cols[s];if (i==j || (same && j>i) || (active[i]==0. && active[j]==0.)) continue;
            const auto &bra=paths[i];const auto &ket=paths[j];
            if (((down[i]^down[j])&p.outside) || bra.spins.get<Checkpoint>(p.end+1)!=ket.spins.get<Checkpoint>(p.end+1)) {
                if constexpr(Track) ++orthogonal_;
                continue;
            }
            double value;
            if (!supported || bra.spins.get<Checkpoint>(p.begin)>=boundary || ket.spins.get<Checkpoint>(p.begin)>=boundary) {
                if constexpr(Track) { ++misses_;++bypass_; }
                value=term.two_body ? guga_two_body_impl<Checkpoint>(bra,ket,term.p,term.q,term.r,term.s) : guga_generator_impl<Checkpoint>(bra,ket,term.p,term.q);
            } else value=lean_lookup<Track,Checkpoint>(bra,ket,term,p.begin,p.parity,p.kept,a[r][0]|b[s][0],p.meta,p.tiny_meta);
            out[r*cols.size()+s]+=term.coefficient*value;
        }
    }
    return true;
}

template<bool Track,bool Checkpoint>
double GugaMemo::compact_element_impl(const Path &bra,const Path &ket,
    const CompactSpinPath &a,const CompactSpinPath &b,const Connection &c,
    const GugaLeanPlan &plan,std::uint64_t a_down,std::uint64_t b_down,bool accelerated) {
    auto encode=pair_encode_ ? nullptr : lean_encoder(accelerated);
    auto pair=pair_encode_ ? lean_pair_encoder(accelerated) : nullptr;
    double value=0.;
    for (size_t t=0;t<c.generators.size();++t) {
        const auto &term=c.generators[t];const auto &p=plan[t];
        if (!p.compatible || ((a_down^b_down)&p.outside) || a.spin(p.end+1)!=b.spin(p.end+1)) {
            if constexpr(Track) ++orthogonal_;
            continue;
        }
        int boundary=format_==Tiny && p.kept==10 ? 4 : 16;
        int ab=a.spin(p.begin),bb=b.spin(p.begin);
        if (!p.fits || (format_==Tiny && p.kept>10) || ab>=boundary || bb>=boundary) {
            if constexpr(Track) { ++misses_;++bypass_; }
            value+=term.coefficient*(term.two_body ? guga_two_body_impl<Checkpoint>(bra,ket,term.p,term.q,term.r,term.s) :
                guga_generator_impl<Checkpoint>(bra,ket,term.p,term.q));
            continue;
        }
        auto steps=pair ? pair(a_down,b_down,p) : encode(a_down,p,true)|encode(b_down,p,false);
        value+=term.coefficient*lean_lookup<Track,Checkpoint,true>(bra,ket,term,p.begin,p.parity,p.kept,
            steps,p.meta,p.tiny_meta,ab,bb,a.phase*b.phase);
    }
    return value;
}

double GugaMemo::compact_element(const Path &bra,const Path &ket,
    const CompactSpinPath &a,const CompactSpinPath &b,const Connection &c,
    const GugaLeanPlan &plan,std::uint64_t a_down,std::uint64_t b_down,bool accelerated) {
    // The direct driver excludes the diagonal before this trusted path.
    if (!packed_ || !a.fits || !b.fits || &bra==&ket)
        return small_element(bra,ket,c,plan,a_down,b_down,accelerated);
    if (a.checkpoint || b.checkpoint)
        return collect_spin_ ? compact_element_impl<true,true>(bra,ket,a,b,c,plan,a_down,b_down,accelerated) :
            compact_element_impl<false,true>(bra,ket,a,b,c,plan,a_down,b_down,accelerated);
    return collect_spin_ ? compact_element_impl<true,false>(bra,ket,a,b,c,plan,a_down,b_down,accelerated) :
        compact_element_impl<false,false>(bra,ket,a,b,c,plan,a_down,b_down,accelerated);
}

// Short rank-two plans dominate many molecular selected blocks. Resolve the
// paired encoder and tiny-cache format once, and expose both generator slots
// to the optimizer. Keys, replacement policy, phases and recurrence fallbacks
// are exactly the general compact kernel's; no coefficient cache is added.
template<bool Track,bool Checkpoint,bool Paired>
double GugaMemo::compact_short_impl(const Path &bra,const Path &ket,
    const CompactSpinPath &a,const CompactSpinPath &b,const Connection &c,
    const GugaLeanPlan &plan,std::uint64_t a_down,std::uint64_t b_down,bool accelerated) {
    auto encode=Paired ? nullptr : lean_encoder(accelerated);
    auto pair=Paired ? lean_pair_encoder(accelerated) : nullptr;
    auto term_value=[&](size_t t) {
        const auto &term=c.generators[t];const auto &p=plan[t];
        if (!p.compatible || ((a_down^b_down)&p.outside) || a.spin(p.end+1)!=b.spin(p.end+1)) {
            if constexpr(Track) ++orthogonal_;
            return 0.;
        }
        int ab=a.spin(p.begin),bb=b.spin(p.begin),boundary=p.kept==10 ? 4 : 16;
        if (!p.fits || p.kept>10 || ab>=boundary || bb>=boundary) {
            if constexpr(Track) { ++misses_;++bypass_; }
            return term.coefficient*(term.two_body ? guga_two_body_impl<Checkpoint>(bra,ket,term.p,term.q,term.r,term.s) :
                guga_generator_impl<Checkpoint>(bra,ket,term.p,term.q));
        }
        std::uint64_t steps;
        if constexpr(Paired) steps=pair(a_down,b_down,p);
        else steps=encode(a_down,p,true)|encode(b_down,p,false);
        return term.coefficient*lean_lookup<Track,Checkpoint,true,true>(bra,ket,term,p.begin,p.parity,p.kept,
            steps,p.meta,p.tiny_meta,ab,bb,a.phase*b.phase);
    };
    double value=term_value(0);
    if (c.generators.size()==2) value+=term_value(1);
    return value;
}

double GugaMemo::compact_short_element(const Path &bra,const Path &ket,
    const CompactSpinPath &a,const CompactSpinPath &b,const Connection &c,
    const GugaLeanPlan &plan,std::uint64_t a_down,std::uint64_t b_down,bool accelerated) {
    if (!packed_ || format_!=Tiny || !a.fits || !b.fits || &bra==&ket ||
        c.generators.empty() || c.generators.size()>2)
        return compact_element(bra,ket,a,b,c,plan,a_down,b_down,accelerated);
    if (a.checkpoint || b.checkpoint) {
        if (pair_encode_) return collect_spin_ ? compact_short_impl<true,true,true>(bra,ket,a,b,c,plan,a_down,b_down,accelerated) :
            compact_short_impl<false,true,true>(bra,ket,a,b,c,plan,a_down,b_down,accelerated);
        return collect_spin_ ? compact_short_impl<true,true,false>(bra,ket,a,b,c,plan,a_down,b_down,accelerated) :
            compact_short_impl<false,true,false>(bra,ket,a,b,c,plan,a_down,b_down,accelerated);
    }
    if (pair_encode_) return collect_spin_ ? compact_short_impl<true,false,true>(bra,ket,a,b,c,plan,a_down,b_down,accelerated) :
        compact_short_impl<false,false,true>(bra,ket,a,b,c,plan,a_down,b_down,accelerated);
    return collect_spin_ ? compact_short_impl<true,false,false>(bra,ket,a,b,c,plan,a_down,b_down,accelerated) :
        compact_short_impl<false,false,false>(bra,ket,a,b,c,plan,a_down,b_down,accelerated);
}

template<bool Track,bool Checkpoint>
bool GugaMemo::compact_batch_impl(const std::vector<Path> &paths,const std::vector<CompactSpinPath> &summary,
    IndexView rows,IndexView cols,const std::vector<std::uint64_t> &down,const std::vector<double> &active,
    bool same,const Connection &c,const GugaLeanPlan &plan,bool accelerated,size_t maximum,GugaBatchScratch &scratch) {
    if (!packed_ || rows.empty() || cols.empty() || rows.size()>maximum/cols.size()) return false;
    for (auto indices:{rows,cols}) for (size_t i:indices) if (!summary[i].fits) return false;
    auto &out=scratch.values;out.assign(rows.size()*cols.size(),0.);
    auto &ae=scratch.bra;auto &be=scratch.ket;ae.resize(rows.size());be.resize(cols.size());
    auto encode=lean_encoder(accelerated);
    for (size_t t=0;t<c.generators.size();++t) {
        const auto &term=c.generators[t];const auto &p=plan[t];if (!p.compatible) continue;
        bool supported=p.fits && (format_!=Tiny || p.kept<=10);
        if (supported) {
            for (size_t r=0;r<rows.size();++r) ae[r][0]=encode(down[rows[r]],p,true);
            for (size_t s=0;s<cols.size();++s) be[s][0]=encode(down[cols[s]],p,false);
        }
        int boundary=format_==Tiny && p.kept==10 ? 4 : 16;
        for (size_t r=0;r<rows.size();++r) for (size_t s=0;s<cols.size();++s) {
            size_t i=rows[r],j=cols[s];
            if (i==j || (same && j>i) || (active[i]==0. && active[j]==0.)) continue;
            const auto &a=summary[i],&b=summary[j];
            bool boundary_equal;
            boundary_equal=a.spin(p.end+1)==b.spin(p.end+1);
            if (((down[i]^down[j])&p.outside) || !boundary_equal) {
                if constexpr(Track) ++orthogonal_;
                continue;
            }
            int ab,bb,phase;
            ab=a.spin(p.begin);bb=b.spin(p.begin);phase=a.phase*b.phase;
            double value;
            if (!supported || ab>=boundary || bb>=boundary) {
                if constexpr(Track) { ++misses_;++bypass_; }
                value=term.two_body ? guga_two_body_impl<Checkpoint>(paths[i],paths[j],term.p,term.q,term.r,term.s) :
                    guga_generator_impl<Checkpoint>(paths[i],paths[j],term.p,term.q);
            } else value=lean_lookup<Track,Checkpoint,true>(paths[i],paths[j],term,p.begin,p.parity,p.kept,
                ae[r][0]|be[s][0],p.meta,p.tiny_meta,ab,bb,phase);
            out[r*cols.size()+s]+=term.coefficient*value;
        }
    }
    return true;
}

bool GugaMemo::compact_batch(const std::vector<Path> &paths,const std::vector<CompactSpinPath> &summary,
    IndexView rows,IndexView cols,const std::vector<std::uint64_t> &down,const std::vector<double> &active,
    bool same,const Connection &c,const GugaLeanPlan &plan,bool accelerated,size_t maximum,GugaBatchScratch &scratch) {
    if (rows.empty() || cols.empty()) return false;

    if (summary[rows[0]].checkpoint || summary[cols[0]].checkpoint)
        return collect_spin_ ? compact_batch_impl<true,true>(paths,summary,rows,cols,down,active,same,c,plan,accelerated,maximum,scratch) :
            compact_batch_impl<false,true>(paths,summary,rows,cols,down,active,same,c,plan,accelerated,maximum,scratch);
    return collect_spin_ ? compact_batch_impl<true,false>(paths,summary,rows,cols,down,active,same,c,plan,accelerated,maximum,scratch) :
        compact_batch_impl<false,false>(paths,summary,rows,cols,down,active,same,c,plan,accelerated,maximum,scratch);
}

double GugaMemo::small_element(const Path &bra,const Path &ket,const Connection &c,
    const GugaLeanPlan &plan,std::uint64_t bra_down,std::uint64_t ket_down,bool accelerated) {
    if (bra.spins.is_compressed() || ket.spins.is_compressed())
        return collect_spin_ ? small_element_impl<true,true>(bra,ket,c,plan,bra_down,ket_down,accelerated) :
                               small_element_impl<false,true>(bra,ket,c,plan,bra_down,ket_down,accelerated);
    return collect_spin_ ? small_element_impl<true,false>(bra,ket,c,plan,bra_down,ket_down,accelerated) :
                           small_element_impl<false,false>(bra,ket,c,plan,bra_down,ket_down,accelerated);
}
bool GugaMemo::small_batch(const std::vector<Path> &paths,IndexView rows,
    IndexView cols,const std::vector<std::uint64_t> &down,const std::vector<double> &active,
    bool same,const Connection &c,const GugaLeanPlan &plan,bool accelerated,size_t maximum,GugaBatchScratch &scratch) {
    if (rows.empty() || cols.empty()) return false;
    if (paths[rows[0]].spins.is_compressed() || paths[cols[0]].spins.is_compressed())
        return collect_spin_ ? small_batch_impl<true,true>(paths,rows,cols,down,active,same,c,plan,accelerated,maximum,scratch) :
                               small_batch_impl<false,true>(paths,rows,cols,down,active,same,c,plan,accelerated,maximum,scratch);
    return collect_spin_ ? small_batch_impl<true,false>(paths,rows,cols,down,active,same,c,plan,accelerated,maximum,scratch) :
                           small_batch_impl<false,false>(paths,rows,cols,down,active,same,c,plan,accelerated,maximum,scratch);
}

template<bool Checkpoint>
double GugaMemo::bit_coupling(const Path &bra,const Path &ket,const GugaTerm &term,
                             const GugaPreparedTerm &p,const BitWords &steps,std::uint64_t static_meta) {
    PackedKey key; key.meta=p.meta; key.steps=steps;
    key.meta[1]=bra.spins.get<Checkpoint>(p.begin); key.meta[2]=ket.spins.get<Checkpoint>(p.begin);
    auto hash=plan_mix(p.hash^key.steps[0]);
    hash=plan_mix(hash^key.steps[1]);
    hash=plan_mix(hash^(std::uint64_t(std::uint32_t(key.meta[1]))<<32)^std::uint32_t(key.meta[2]));
    Slot *slot=slots_.empty() ? nullptr : &slots_[hash&(slots_.size()-1)];
    key_sites_+=p.meta[3];
    int phase=bra.phase*ket.phase*p.parity;
    bool hit=slot && slot->uses && slot->key==key;
    if (hit) { ++hits_; ++slot->uses; } else ++misses_;
    if (hit && reuse_) return phase*slot->value;
    double coupling=term.two_body ? guga_two_body_impl<Checkpoint>(bra,ket,term.p,term.q,term.r,term.s) :
                                   guga_generator_impl<Checkpoint>(bra,ket,term.p,term.q);
    if (slot && !hit) {
        replacements_+=slot->uses!=0; slot->key=key; slot->value=phase*coupling; slot->uses=1;
    } else if (!slot) ++bypass_;
    return coupling;
}

template<bool Checkpoint>
double GugaMemo::bit_element_impl(const Path &bra,const Path &ket,const Connection &c,
                            const GugaPlan &plan,const GugaBitPlan &bits,
                            std::uint64_t bra_down,std::uint64_t ket_down,bool accelerated) {
    if (format_!=Generic) return collect_spin_ ? lean_element<true,Checkpoint>(bra,ket,c,plan,bits,bra_down,ket_down,accelerated) :
                                          lean_element<false,Checkpoint>(bra,ket,c,plan,bits,bra_down,ket_down,accelerated);
    if (!packed_ || bra.label.size()>64) return planned_element_impl<Checkpoint>(bra,ket,c,plan);
    auto encode=bit_encoder(accelerated);
    double value=bra.label==ket.label ? c.scalar : 0.;
    for (size_t t=0;t<c.generators.size();++t) {
        const auto &term=c.generators[t]; const auto &p=plan[t]; const auto &b=bits[t];
        if (!p.fits) {
            ++misses_; ++bypass_;
            value+=term.coefficient*(term.two_body ? guga_two_body_impl<Checkpoint>(bra,ket,term.p,term.q,term.r,term.s) :
                                                   guga_generator_impl<Checkpoint>(bra,ket,term.p,term.q));
            continue;
        }
        if (!b.compatible || ((bra_down^ket_down)&b.outside) || bra.spins.get<Checkpoint>(p.end+1)!=ket.spins.get<Checkpoint>(p.end+1)) {
            ++orthogonal_; value+=term.coefficient*(bra.phase*ket.phase*0.); continue;
        }
        auto a=encode(bra_down,b,true),d=encode(ket_down,b,false);
        value+=term.coefficient*bit_coupling<Checkpoint>(bra,ket,term,p,{a[0]|d[0],a[1]|d[1]},b.compact_meta);
    }
    return value;
}

template<bool Checkpoint>
bool GugaMemo::batch_elements_impl(const std::vector<Path> &paths,IndexView rows,
                             IndexView cols,const std::vector<std::uint64_t> &down,
                             const std::vector<double> &active,bool same,const Connection &c,
                             const GugaPlan &plan,const GugaBitPlan &bits,bool accelerated,
                             size_t maximum,GugaBatchScratch &scratch) {
    if (format_!=Generic) return collect_spin_ ? lean_batch<true,Checkpoint>(paths,rows,cols,down,active,same,c,plan,bits,accelerated,maximum,scratch) :
                                          lean_batch<false,Checkpoint>(paths,rows,cols,down,active,same,c,plan,bits,accelerated,maximum,scratch);
    if (!packed_ || rows.empty() || cols.empty() || down.empty() || rows.size()>maximum/cols.size()) return false;
    for (const auto &p:plan) if (!p.fits) return false;
    auto &out=scratch.values; out.assign(rows.size()*cols.size(),0.);
    auto encode=bit_encoder(accelerated);
    auto &a=scratch.bra; auto &b=scratch.ket; a.resize(rows.size()); b.resize(cols.size());
    for (size_t t=0;t<c.generators.size();++t) {
        const auto &term=c.generators[t]; const auto &p=plan[t]; const auto &bp=bits[t];
        if (!bp.compatible) continue;
        for (size_t r=0;r<rows.size();++r) a[r]=encode(down[rows[r]],bp,true);
        for (size_t s=0;s<cols.size();++s) b[s]=encode(down[cols[s]],bp,false);
        for (size_t r=0;r<rows.size();++r) for (size_t s=0;s<cols.size();++s) {
            size_t i=rows[r],j=cols[s];
            if (i==j || (same && j>i) || (active[i]==0. && active[j]==0.)) continue;
            const auto &bra=paths[i]; const auto &ket=paths[j];
            if (((down[i]^down[j])&bp.outside) || bra.spins.get<Checkpoint>(p.end+1)!=ket.spins.get<Checkpoint>(p.end+1)) {
                ++orthogonal_; continue;
            }
            out[r*cols.size()+s]+=term.coefficient*bit_coupling<Checkpoint>(bra,ket,term,p,
                {a[r][0]|b[s][0],a[r][1]|b[s][1]},bp.compact_meta);
        }
    }
    return true;
}

template<bool Checkpoint>
double GugaMemo::packed_element(const Path &bra, const Path &ket, const Connection &connection) {
    double value = bra.label == ket.label ? connection.scalar : 0.;
    size_t first = 0, last = bra.label.size();
    while (first < last && bra.label[first] == ket.label[first]) ++first;
    while (last > first && bra.label[last-1] == ket.label[last-1]) --last;
    for (const auto &term : connection.generators) {
        int begin = std::min(term.p,term.q), end = std::max(term.p,term.q);
        if (term.two_body) { begin = std::min({begin,term.r,term.s}); end = std::max({end,term.r,term.s}); }
        if ((first != last && (static_cast<size_t>(begin) > first || static_cast<size_t>(end+1) < last)) ||
            bra.spins.get<Checkpoint>(end+1) != ket.spins.get<Checkpoint>(end+1)) {
            ++orthogonal_;
            value += term.coefficient*(bra.phase*ket.phase*0.);
            continue;
        }
        PackedKey key;
        key.meta[0] = term.two_body;
        key.meta[1] = bra.spins.get<Checkpoint>(begin); key.meta[2] = ket.spins.get<Checkpoint>(begin);
        int kept = 0, parity = 1;
        bool fits = true;
        for (int i = begin; i <= end; ++i) {
            ++key_sites_;
            bool endpoint = i == term.p || i == term.q || (term.two_body && (i == term.r || i == term.s));
            if (!endpoint && bra.label[i] == ket.label[i] && (ket.label[i] == '0' || ket.label[i] == '2')) {
                int active = (std::min(term.p,term.q) < i && i < std::max(term.p,term.q));
                if (term.two_body) active += std::min(term.r,term.s) < i && i < std::max(term.r,term.s);
                if (ket.label[i] == '2' && active%2) parity = -parity;
                continue;
            }
            if (kept == 32) { fits = false; break; }
            if (i == term.p) key.meta[4] = kept;
            if (i == term.q) key.meta[5] = kept;
            if (term.two_body && i == term.r) key.meta[6] = kept;
            if (term.two_body && i == term.s) key.meta[7] = kept;
            key.steps[kept/16] |= std::uint64_t(4*step(bra.label[i])+step(ket.label[i])) << (4*(kept%16));
            ++kept;
        }
        key.meta[3] = kept;
        Slot *slot = nullptr;
        if (fits && !slots_.empty()) {
            // Exact descriptors decide identity. The hash only chooses a slot;
            // collisions replace entries and can never return another coupling.
            std::uint64_t hash = 0x9e3779b97f4a7c15ULL;
            auto mix = [&](std::uint64_t x) {
                x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
                x ^= x >> 27; x *= 0x94d049bb133111ebULL;
                hash ^= (x^(x >> 31))+0x9e3779b97f4a7c15ULL+(hash << 6)+(hash >> 2);
            };
            for (auto x : key.steps) mix(x);
            for (int x : key.meta) mix(static_cast<unsigned int>(x));
            slot = &slots_[hash&(slots_.size()-1)];
        }
        int phase = bra.phase*ket.phase*parity;
        bool hit = slot && slot->uses && slot->key == key;
        if (hit) { ++hits_; ++slot->uses; } else ++misses_;
        double coupling;
        if (hit && reuse_) coupling = phase*slot->value;
        else {
            coupling = term.two_body ? guga_two_body_impl<Checkpoint>(bra,ket,term.p,term.q,term.r,term.s) :
                                      guga_generator_impl<Checkpoint>(bra,ket,term.p,term.q);
            if (slot && !hit) {
                replacements_ += slot->uses != 0;
                slot->key = key; slot->value = phase*coupling; slot->uses = 1;
            } else if (!slot) ++bypass_;
        }
        value += term.coefficient*coupling;
    }
    return value;
}

template<bool Checkpoint>
double GugaMemo::element_impl(const Path &bra, const Path &ket, const Connection &connection) {
    if (packed_) return packed_element<Checkpoint>(bra,ket,connection);
    double value = bra.label == ket.label ? connection.scalar : 0.0;
    for (const auto &term : connection.generators) {
        int begin = std::min(term.p,term.q), end = std::max(term.p,term.q);
        if (term.two_body) { begin = std::min({begin,term.r,term.s}); end = std::max({end,term.r,term.s}); }
        if (bra.label.compare(0,begin,ket.label,0,begin) ||
            bra.label.compare(end+1,std::string::npos,ket.label,end+1,std::string::npos) ||
            bra.spins.get<Checkpoint>(end+1) != ket.spins.get<Checkpoint>(end+1)) {
            ++orthogonal_;
            value += term.coefficient*(bra.phase*ket.phase*0.0);
            continue;
        }
        std::string key;
        auto integer = [&](int x) { key.append(reinterpret_cast<const char *>(&x),sizeof(x)); };
        integer(term.two_body); integer(bra.spins.get<Checkpoint>(begin)); integer(ket.spins.get<Checkpoint>(begin));
        std::array<int,4> mapped{};
        int kept = 0, parity = 1;
        for (int i = begin; i <= end; ++i) {
            bool endpoint = i == term.p || i == term.q || (term.two_body && (i == term.r || i == term.s));
            if (!endpoint && bra.label[i] == ket.label[i] &&
                (ket.label[i] == '0' || ket.label[i] == '2')) {
                // Common closed spectators contribute -1 for each active
                // generator. Empty spectators are identities. The same parity
                // applies to E[p,s] when q==r, so normal ordering is preserved.
                int active = (std::min(term.p,term.q) < i && i < std::max(term.p,term.q));
                if (term.two_body) active += std::min(term.r,term.s) < i && i < std::max(term.r,term.s);
                if (ket.label[i] == '2' && active%2) parity = -parity;
                continue;
            }
            if (i == term.p) mapped[0] = kept;
            if (i == term.q) mapped[1] = kept;
            if (term.two_body && i == term.r) mapped[2] = kept;
            if (term.two_body && i == term.s) mapped[3] = kept;
            key.push_back(static_cast<char>(4*step(bra.label[i])+step(ket.label[i])));
            ++kept;
        }
        integer(kept); integer(mapped[0]); integer(mapped[1]);
        if (term.two_body) { integer(mapped[2]); integer(mapped[3]); }
        int phase = bra.phase*ket.phase*parity;
        auto found = entries_.find(key);
        double coupling;
        if (found != entries_.end()) {
            ++hits_;
            ++found->second.uses;
        } else ++misses_;
        if (reuse_ && found != entries_.end()) coupling = phase*found->second.value;
        else {
            coupling = term.two_body ? guga_two_body_impl<Checkpoint>(bra,ket,term.p,term.q,term.r,term.s) :
                                      guga_generator_impl<Checkpoint>(bra,ket,term.p,term.q);
            // Conservative node/bucket/key accounting; allocator RSS is separate.
            size_t bytes = sizeof(std::pair<const std::string,Entry>)+key.capacity()+1+128;
            if (found == entries_.end()) {
                if (bytes <= maximum_-bytes_) {
                    entries_.emplace(std::move(key),Entry{phase*coupling,1});
                    bytes_ += bytes;
                } else ++bypass_;
            }
        }
        value += term.coefficient*coupling;
    }
    return value;
}

double GugaMemo::element(const Path &bra,const Path &ket,const Connection &c) {
    return bra.spins.is_compressed() || ket.spins.is_compressed() ?
        element_impl<true>(bra,ket,c) : element_impl<false>(bra,ket,c);
}
double GugaMemo::planned_element(const Path &bra,const Path &ket,const Connection &c,const GugaPlan &plan) {
    return bra.spins.is_compressed() || ket.spins.is_compressed() ?
        planned_element_impl<true>(bra,ket,c,plan) : planned_element_impl<false>(bra,ket,c,plan);
}
double GugaMemo::bit_element(const Path &bra,const Path &ket,const Connection &c,
    const GugaPlan &plan,const GugaBitPlan &bits,std::uint64_t a,std::uint64_t b,bool accelerated) {
    return bra.spins.is_compressed() || ket.spins.is_compressed() ?
        bit_element_impl<true>(bra,ket,c,plan,bits,a,b,accelerated) :
        bit_element_impl<false>(bra,ket,c,plan,bits,a,b,accelerated);
}
bool GugaMemo::batch_elements(const std::vector<Path> &paths,IndexView rows,IndexView cols,
    const std::vector<std::uint64_t> &down,const std::vector<double> &active,bool same,
    const Connection &c,const GugaPlan &plan,const GugaBitPlan &bits,bool accelerated,
    size_t maximum,GugaBatchScratch &scratch) {
    if (rows.empty() || cols.empty()) return false;
    return paths[rows[0]].spins.is_compressed() || paths[cols[0]].spins.is_compressed() ?
        batch_elements_impl<true>(paths,rows,cols,down,active,same,c,plan,bits,accelerated,maximum,scratch) :
        batch_elements_impl<false>(paths,rows,cols,down,active,same,c,plan,bits,accelerated,maximum,scratch);
}

std::map<std::string,size_t> GugaMemo::info() const {
    size_t entries = entries_.size();
    if (packed_) for (const auto &slot : slots_) entries += slot.uses != 0;
    for (const auto &slot:lean_slots_) entries+=(slot.meta&(1u<<30))!=0;
    for (size_t i=0;i<tiny_count();++i) entries+=(tiny_data()[i].key&(1ULL<<62))!=0;
    return {{"reuse_hits",hits_},{"reuse_misses",misses_},{"reuse_bypass",bypass_},
            {"reuse_orthogonal",orthogonal_},{"reuse_bytes",bytes_},{"reuse_entries",entries},
            {"reuse_replacements",replacements_},{"reuse_key_sites",key_sites_},
            {"reuse_slots",slots_.size()+lean_slots_.size()+tiny_count()},
            {"aligned_tiny_cache",bool(aligned_tiny_)},{"paired_spin_encoding",pair_encode_},
            {"compact_spin_cache",format_!=Generic},
            {"lean_mode",size_t(format_)},{"spin_statistics",format_==Generic || collect_spin_}};
}

std::map<size_t,size_t> GugaMemo::frequencies() const {
    std::map<size_t,size_t> result;
    for (const auto &item : entries_) ++result[item.second.uses];
    for (const auto &slot : slots_) if (slot.uses) ++result[slot.uses];
    if (collect_spin_) for (const auto &slot:lean_slots_) if (slot.meta&(1u<<30)) ++result[slot.uses];
    return result;
}

void bind_guga(pybind11::module_ &m) {
    m.attr("guga_factor_table_bytes") = sizeof(FactorTable);
    m.def("_guga_segment_factor", [](int da,int db,int a,int b,int p,int q,int site,int mode) {
        if (da < 0 || da > 3 || db < 0 || db > 3 || a < 0 || b < 0 || mode < 0 || mode > 2)
            throw std::invalid_argument("invalid segment arguments");
        return segment(da,db,a,b,p,q,site,mode);
    });
    m.def("guga_local_norm_bound", [](const Config &occupations, const std::vector<int> &indices) {
        if (indices.size() != 2 && indices.size() != 4)
            throw std::invalid_argument("GUGA operator needs two or four indices");
        for (int n : occupations) if (n < 0 || n > 2) throw std::invalid_argument("occupations must be 0,1,2");
        for (int p : indices) if (p < 0 || static_cast<size_t>(p) >= occupations.size())
            throw std::invalid_argument("orbital index out of range");
        pybind11::gil_scoped_release release;
        return guga_local_bound(occupations,indices[0],indices[1],
                                indices.size() == 4 ? indices[2] : 0,
                                indices.size() == 4 ? indices[3] : 0,indices.size() == 4);
    },pybind11::arg("occupations"),pybind11::arg("indices"));
    // Independent generator/two-body matrices for signed acceptance tests.
    m.def("guga_operator_block", [](const std::vector<std::string> &rows,
                                   const std::vector<std::string> &cols, int k, int two_s,
                                   const std::vector<int> &indices, size_t maximum, size_t budget,
                                   size_t max_bonds, int guga_mode) {
        if (guga_mode < 0 || guga_mode > 2) throw std::invalid_argument("invalid guga_mode");
        if (k < 0 || two_s < 0 || (indices.size() != 2 && indices.size() != 4))
            throw std::invalid_argument("invalid GUGA operator dimensions");
        for (int p : indices) if (p < 0 || p >= k) throw std::invalid_argument("orbital index out of range");
        if (!cols.empty() && rows.size() > maximum/cols.size())
            throw BudgetError("GUGA block exceeds max_block_elements");
        if (!rows.empty() && !cols.empty() && k && indices.size() == 4 && max_bonds < 2)
            throw BudgetError("GUGA segment scratch exceeds max_bond_elements");
        std::vector<Path> bra, ket;
        for (const auto &label : rows) bra.emplace_back(label, k, two_s, budget, false, guga_mode);
        for (const auto &label : cols) ket.emplace_back(label, k, two_s, budget, false, guga_mode);
        for (const auto &a : bra) for (const auto &b : ket)
            if (a.electrons != b.electrons) throw std::invalid_argument("CSFs must share electron number");
        Array result({static_cast<pybind11::ssize_t>(rows.size()), static_cast<pybind11::ssize_t>(cols.size())});
        double *out = result.mutable_data();
        pybind11::gil_scoped_release release;
        for (size_t i = 0; i < bra.size(); ++i) for (size_t j = 0; j < ket.size(); ++j)
            out[i*ket.size()+j] = indices.size() == 2 ?
                guga_generator(bra[i], ket[j], indices[0], indices[1]) :
                guga_two_body(bra[i], ket[j], indices[0], indices[1], indices[2], indices[3]);
        return result;
    }, pybind11::arg("rows"), pybind11::arg("cols"), pybind11::arg("nbasis"), pybind11::arg("two_s"),
       pybind11::arg("indices"), pybind11::arg("max_block_elements") = 1000000,
       pybind11::arg("max_tensor_entries") = 1000000, pybind11::arg("max_bond_elements") = 1000000,
       pybind11::arg("guga_mode") = 0);
}
#include "guga_selected_impl.h"
#include "guga_weighted_impl.h"
} // namespace scsf

/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#pragma once

#include <pybind11/numpy.h>
#include "index_view.h"
#include "spin_vertices.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace scsf {
using Array = pybind11::array_t<double, pybind11::array::c_style>;
using Config = std::vector<int>;

struct Site {
    int before, after, occupation, nstate;
    std::array<int, 2> state;
    std::array<std::vector<int>, 2> destination;
    std::array<std::vector<double>, 2> coefficient;
};

struct Path {
    std::string label;
    std::vector<Site> sites;
    SpinVertices spins; // GUGA doubled-spin vertices, full or exact checkpoints
    int phase = 1; // diagonal conversion from segment-table to the CSF CG phase convention
    int electrons = 0;
    int guga_mode = 0; // 0 analytic reference, 1 factor table, 2 spectator runs
    std::vector<int> next_single, closed_prefix;

    Path(const std::string &steps, int k, int two_s, size_t &budget, bool tensors = true,
         int factor_mode = 0, bool spin_checkpoints = false) : label(steps), guga_mode(factor_mode) {
        if (steps.size() != static_cast<size_t>(k))
            throw std::invalid_argument("CSF label length differs from orbital count");
        if (!tensors) {
            size_t entries = (steps.size()+1)*(guga_mode == 2 ? 3 : 1);
            if (entries > budget) throw std::length_error("GUGA walks exceed max_tensor_entries");
            budget -= entries;
            spins.reserve(steps.size()+1);
            spins.push_back(0);
            for (char step : steps) {
                if (step != '0' && step != 'u' && step != 'd' && step != '2')
                    throw std::invalid_argument("CSF symbols must be 0,u,d,2");
                int spin = spins.back()+(step == 'u')-(step == 'd');
                if (spin < 0) throw std::invalid_argument("negative intermediate CSF spin");
                spins.push_back(spin);
                electrons += step == '2' ? 2 : step != '0';
                if ((step == 'd' || step == '2') && (spin & 1)) phase = -phase;
            }
            if (spins.back() != two_s) throw std::invalid_argument("CSF final spin differs from two_s");
            if (guga_mode == 2) {
                next_single.resize(steps.size()+1,k);
                closed_prefix.resize(steps.size()+1,0);
                for (int p = k-1; p >= 0; --p)
                    next_single[p] = steps[p] == 'u' || steps[p] == 'd' ? p : next_single[p+1];
                for (int p = 0; p < k; ++p) closed_prefix[p+1] = closed_prefix[p]+(steps[p] == '2');
            }
            if (spin_checkpoints) spins.compress();
            return;
        }
        int j = 0;
        for (char step : steps) {
            Site site;
            site.before = j;
            if (step == 'u' || step == 'd') {
                j += step == 'u' ? 1 : -1;
                site.occupation = 1;
                site.nstate = 2;
                site.state = {1, 2}; // local |alpha>, |beta>
            } else if (step == '0' || step == '2') {
                site.occupation = step == '0' ? 0 : 2;
                site.nstate = 1;
                site.state = {step == '0' ? 0 : 3, 0};
            } else {
                throw std::invalid_argument("CSF symbols must be 0,u,d,2");
            }
            if (j < 0)
                throw std::invalid_argument("negative intermediate CSF spin");
            site.after = j;
            electrons += site.occupation;
            size_t entries = static_cast<size_t>(site.before+1)*site.nstate;
            if (entries > budget)
                throw std::length_error("spin tensors exceed max_tensor_entries");
            budget -= entries;
            for (int t = 0; t < site.nstate; ++t) {
                for (int i = 0; i <= site.before; ++i) {
                    int m = -site.before + 2*i;
                    int next = i;
                    double coefficient = 1.0;
                    if (site.occupation == 1) {
                        int sigma = t == 0 ? 1 : -1;
                        int next_m = m + sigma;
                        next = (next_m+j)/2;
                        if (std::abs(next_m) > j) {
                            coefficient = 0.0;
                            next = 0;
                        } else if (step == 'u') {
                            coefficient = std::sqrt((site.before+sigma*m+2.0) /
                                                    (2.0*(site.before+1)));
                        } else {
                            coefficient = (sigma == 1 ? -1.0 : 1.0) *
                                std::sqrt((site.before-sigma*m) / (2.0*(site.before+1)));
                        }
                    }
                    site.destination[t].push_back(next);
                    site.coefficient[t].push_back(coefficient);
                }
            }
            sites.push_back(std::move(site));
        }
        if (j != two_s)
            throw std::invalid_argument("CSF final spin differs from two_s");
    }
};

struct Operator {
    int orbital, spin;
    bool create;
};


struct Term {
    double coefficient;
    std::array<Operator, 4> operators;
    int count;
    size_t contraction = 0;
    int sign = 1;
};

struct GugaTerm {
    double coefficient;
    int p, q, r, s;
    bool two_body;
};

struct Connection {
    bool guga = false;
    double scalar = 0.0;
    std::vector<Term> terms;
    std::vector<GugaTerm> generators;
    size_t guga_input_terms = 0;
    double bound = 0.0;
    double term_bound = 0.0; // external paths of one occupation block: no identity actions
    // Representative local actions, preserving the original term sum order.
    std::vector<size_t> representatives;
    // Row-major local maps [action, open-site, ket-state]. Codes 0,+/-1,+/-2
    // mean annihilated or a signed destination in the bra's local state list.
    std::vector<int> open_sites;
    std::vector<signed char> action_signs, local_maps;

    size_t term_count() const { return guga ? guga_input_terms : terms.size(); }
    size_t action_count() const { return guga ? generators.size() : representatives.size(); }
};

struct BoundedTransfer {
    double bound;
    int p, r;
};

// Integral-independent work shared by every spin-path pair in an occupation
// block. No intermediate selected-space projector is introduced.
struct GugaPreparedTerm {
    int begin=0, end=0, parity=1;
    bool fits=true;
    std::array<int,32> sites{};
    std::array<int,8> meta{};
    std::uint64_t hash=0;
};
using GugaPlan = std::vector<GugaPreparedTerm>;
void prepare_guga_plan(const Config &,const Config &,const Connection &,GugaPlan &);

struct GugaBitTerm {
    std::uint64_t keep=0, outside=0, compact_meta=0;
    std::array<std::uint64_t,2> bra{}, ket{};
    std::uint32_t lean_meta=0;
    bool compatible=true;
};
using GugaBitPlan=std::vector<GugaBitTerm>;
// Lean-only preparation: no legacy site array, hash, or second step word.
struct GugaLeanTerm {
    std::uint64_t keep=0,outside=0,bra=0,ket=0;
    std::uint32_t meta=0,tiny_meta=0;
    std::int16_t begin=0,end=0;
    std::int8_t parity=1;
    std::uint8_t kept=0;
    bool fits=true,compatible=true;
};
using GugaLeanPlan=std::vector<GugaLeanTerm>;
// Four-bit exact boundary spins for <=64 orbitals. Wider/high-spin paths
// retain the full Path fallback. This metadata contains no H coefficients.
struct CompactSpinPath {
    std::array<std::uint64_t,5> words{};
    std::int8_t phase=1;
    bool fits=true,checkpoint=false;
    int spin(int p) const { return (words[p/16]>>(4*(p%16)))&15; }
};
void prepare_guga_lean_connection(const Config &,const Config &,const Connection &,GugaLeanPlan &,bool tiny);

void prepare_guga_mask_connection(const Connection &,GugaLeanPlan &,bool tiny,
    std::uint64_t bra_single,std::uint64_t bra_double,std::uint64_t ket_single,std::uint64_t ket_double);

struct GugaBatchScratch {
    std::vector<double> values;
    std::vector<std::array<std::uint64_t,2>> bra,ket;
    size_t bytes() const { return values.capacity()*sizeof(double)+(bra.capacity()+ket.capacity())*sizeof(bra[0]); }
};
void prepare_guga_bits(const Config &,const Config &,const GugaPlan &,GugaBitPlan &);
void prepare_guga_bit_connection(const Config &,const Config &,const Connection &,GugaPlan &,GugaBitPlan &,bool lean=false);
bool guga_bmi2_available();

// Per-CSR-worker spin-only reuse experiment. Keys retain open/endpoint steps,
// boundary spins and relative operator indices; common closed spectators are
// removed with an explicit parity. No physical integral enters the key/value.
class GugaMemo {
public:
    enum Format { Generic, Wide, Tiny };
private:
    struct Entry { double value; size_t uses; };
    std::unordered_map<std::string,Entry> entries_;
    size_t maximum_, bytes_ = 0, hits_ = 0, misses_ = 0, bypass_ = 0, orthogonal_ = 0;
    bool reuse_;
    struct PackedKey {
        std::array<std::uint64_t,2> steps{};
        std::array<int,8> meta{};
        bool operator==(const PackedKey &other) const { return steps == other.steps && meta == other.meta; }
    };
    struct Slot { PackedKey key; double value = 0.; size_t uses = 0; };
    std::vector<Slot> slots_;
    size_t replacements_ = 0, key_sites_ = 0;
    bool packed_;
    size_t primary_hits_=0,secondary_hits_=0,empty_misses_=0,full_misses_=0;
    struct LeanSlot { std::uint64_t steps=0; double value=0.; std::uint32_t meta=0,uses=0; };
    struct TinySlot { std::uint64_t key=0; double value=0.; };
    struct TinyDelete {
        void operator()(TinySlot *p) const { ::operator delete(p,std::align_val_t(64)); }
    };
    std::vector<LeanSlot> lean_slots_;
    std::vector<TinySlot> tiny_slots_;
    std::unique_ptr<TinySlot,TinyDelete> aligned_tiny_;
    size_t tiny_count_=0;
    TinySlot *tiny_data_=nullptr;
    bool align_tiny_=false,pair_encode_=false;
    TinySlot *tiny_data() { return tiny_data_; }
    const TinySlot *tiny_data() const { return tiny_data_; }
    size_t tiny_count() const { return tiny_count_; }
    Format format_;
    bool collect_spin_;
    template<bool Track,bool Checkpoint,bool Compact=false,bool TinyOnly=false> double lean_lookup(const Path &,const Path &,const GugaTerm &,
        int,int,int,std::uint64_t,std::uint32_t,std::uint32_t tiny_meta=0,
        int bra_begin=0,int ket_begin=0,int phase=1);
    template<bool Track,bool Checkpoint> double compact_element_impl(const Path &,const Path &,
        const CompactSpinPath &,const CompactSpinPath &,const Connection &,const GugaLeanPlan &,
        std::uint64_t,std::uint64_t,bool);
    template<bool Track,bool Checkpoint,bool Paired> double compact_short_impl(const Path &,const Path &,
        const CompactSpinPath &,const CompactSpinPath &,const Connection &,const GugaLeanPlan &,
        std::uint64_t,std::uint64_t,bool);
    template<bool Track,bool Checkpoint> bool compact_batch_impl(const std::vector<Path> &,
        const std::vector<CompactSpinPath> &,IndexView,IndexView,const std::vector<std::uint64_t> &,
        const std::vector<double> &,bool,const Connection &,const GugaLeanPlan &,bool,size_t,GugaBatchScratch &);
    template<bool Track,bool Checkpoint> double lean_element(const Path &,const Path &,const Connection &,
        const GugaPlan &,const GugaBitPlan &,std::uint64_t,std::uint64_t,bool);
    template<bool Track,bool Checkpoint> bool lean_batch(const std::vector<Path> &,IndexView,
        IndexView,const std::vector<std::uint64_t> &,const std::vector<double> &,
        bool,const Connection &,const GugaPlan &,const GugaBitPlan &,bool,size_t,GugaBatchScratch &);
    template<bool Track,bool Checkpoint> double small_element_impl(const Path &,const Path &,const Connection &,
        const GugaLeanPlan &,std::uint64_t,std::uint64_t,bool);
    template<bool Track,bool Checkpoint> bool small_batch_impl(const std::vector<Path> &,IndexView,
        IndexView,const std::vector<std::uint64_t> &,const std::vector<double> &,
        bool,const Connection &,const GugaLeanPlan &,bool,size_t,GugaBatchScratch &);
    template<bool Checkpoint> double packed_element(const Path &, const Path &, const Connection &);
    template<bool Checkpoint> double bit_coupling(const Path &,const Path &,const GugaTerm &,const GugaPreparedTerm &,
                        const std::array<std::uint64_t,2> &,std::uint64_t);
    template<bool Checkpoint> double planned_element_impl(const Path &,const Path &,const Connection &,const GugaPlan &);
    template<bool Checkpoint> double bit_element_impl(const Path &,const Path &,const Connection &,const GugaPlan &,
                       const GugaBitPlan &,std::uint64_t,std::uint64_t,bool);
    template<bool Checkpoint> bool batch_elements_impl(const std::vector<Path> &,IndexView,
                        IndexView,const std::vector<std::uint64_t> &,
                        const std::vector<double> &,bool,const Connection &,
                        const GugaPlan &,const GugaBitPlan &,bool,size_t,GugaBatchScratch &);
    template<bool Checkpoint> double element_impl(const Path &bra, const Path &ket, const Connection &connection);
public:
    double compact_short_element(const Path &,const Path &,const CompactSpinPath &,const CompactSpinPath &,
        const Connection &,const GugaLeanPlan &,std::uint64_t,std::uint64_t,bool);
    GugaMemo(size_t maximum, bool reuse, bool packed=false,Format format=Generic,
             bool collect_spin=true,bool align_tiny=false,bool pair_encode=false);
    bool tiny_enabled() const { return reuse_ && packed_ && format_==Tiny && tiny_count()!=0; }
    bool tiny_aligned() const { return bool(aligned_tiny_); }
    bool tiny_line_aligned() const { return !tiny_count() || reinterpret_cast<std::uintptr_t>(tiny_data())%64==0; }
    bool tiny_cacheable(const GugaLeanTerm &p,const std::vector<Path> &paths,
                        IndexView rows,IndexView cols,int electrons) const {
        if (!tiny_enabled() || !p.fits || p.kept>10) return false;
        if (!p.compatible) return true; // exact structural zero
        int boundary=p.kept==10 ? 4 : 16;
        if (std::min(int(p.begin),electrons)<boundary) return true;
        for (auto indices:{rows,cols}) for (size_t i:indices)
            if (paths[i].spins[p.begin]>=boundary) return false;
        return true;
    }
    double element(const Path &bra, const Path &ket, const Connection &connection);
    double planned_element(const Path &,const Path &,const Connection &,const GugaPlan &);
    double bit_element(const Path &,const Path &,const Connection &,const GugaPlan &,
                       const GugaBitPlan &,std::uint64_t,std::uint64_t,bool);
    bool batch_elements(const std::vector<Path> &,IndexView,
                        IndexView,const std::vector<std::uint64_t> &,
                        const std::vector<double> &,bool,const Connection &,
                        const GugaPlan &,const GugaBitPlan &,bool,size_t,GugaBatchScratch &);
    double small_element(const Path &,const Path &,const Connection &,const GugaLeanPlan &,
                         std::uint64_t,std::uint64_t,bool);
    double compact_element(const Path &,const Path &,const CompactSpinPath &,const CompactSpinPath &,
        const Connection &,const GugaLeanPlan &,std::uint64_t,std::uint64_t,bool);
    bool compact_batch(const std::vector<Path> &,const std::vector<CompactSpinPath> &,IndexView,IndexView,
        const std::vector<std::uint64_t> &,const std::vector<double> &,bool,const Connection &,
        const GugaLeanPlan &,bool,size_t,GugaBatchScratch &);
    bool small_batch(const std::vector<Path> &,IndexView,
        IndexView,const std::vector<std::uint64_t> &,const std::vector<double> &,
        bool,const Connection &,const GugaLeanPlan &,bool,size_t,GugaBatchScratch &);
    // Read hot-loop counters without scanning every retained memo slot.
    std::array<size_t,9> product_counters() const {
        return {hits_,misses_,orthogonal_,replacements_,primary_hits_,secondary_hits_,empty_misses_,full_misses_,
                bypass_};
    }
    std::map<std::string,size_t> info() const;
    std::map<size_t,size_t> frequencies() const;
};

// Backend-owned, immutable-integral cache. Shared handles keep an evicted entry
// alive for its current worker only; the retained payload has a strict byte cap.
class ConnectionCache {
    using Key = std::pair<Config, Config>;
    struct Entry {
        std::shared_ptr<const Connection> value;
        size_t bytes;
        std::list<Key>::iterator position;
    };
    Array one_, two_;
    size_t maximum_, bytes_ = 0, hits_ = 0, evictions_ = 0;
    std::atomic<size_t> misses_{0};
    bool guga_;
    size_t max_transfer_bounds_;
    std::vector<double> transfer_bounds_;
    std::atomic<size_t> bound_entries_{0};
    std::once_flag bound_once_;
    std::once_flag ordered_once_;
    std::vector<BoundedTransfer> ordered_transfers_;
    std::vector<size_t> ordered_offsets_;
    std::atomic<bool> ordered_ready_{false};
    std::atomic<size_t> ordered_cap_{0};
    std::list<Key> order_;
    std::map<Key, Entry> entries_;
    mutable std::mutex mutex_;
public:
    ConnectionCache(const Array &one, const Array &two, size_t maximum, bool guga = false,
                    size_t max_transfer_bounds = 0)
        : one_(one), two_(two), maximum_(maximum), guga_(guga), max_transfer_bounds_(max_transfer_bounds) {}
    void validate(const Array &one, const Array &two) const;
    std::shared_ptr<const Connection> get(const Config &bra, const Config &ket,
        const double *h, const double *v, int k, size_t max_terms, bool guga = false);
    void clear();
    std::map<std::string, size_t> info() const;
    void prepare_transfer_bounds(int k);
    double selection_bound(const Config &source, const std::array<int,2> &holes,
                           const std::array<int,2> &particles, int order, int k, bool &reused) const;
    std::map<std::string,size_t> transfer_info() const;
    bool prepare_ordered_transfers(int k, size_t max_bytes);
    std::pair<const BoundedTransfer *, const BoundedTransfer *> ordered_transfers(int q, int s, int k) const;
    std::map<std::string,size_t> search_info() const;
};

struct BudgetError : std::runtime_error { using std::runtime_error::runtime_error; };

Config configuration(const std::string &label, int two_s);
void check_bonds(const Path &bra, const Path &ket, size_t max_bond_elements);
double transfer_bound(const Config &ket, const std::array<int, 2> &holes,
                      const std::array<int, 2> &particles, int order,
                      const double *h, const double *v, int k);
double prescreen_bound(const Config &bra, const Config &ket, const double *h,
                       const double *v, int k);
Connection connection(const Config &bra, const Config &ket, const double *h, const double *v,
                      int k, size_t max_terms, bool guga = false, bool compute_bounds = true);
double element(const Path &bra, const Path &ket, int two_s, const Connection &connection,
               bool compiled = true);
double guga_generator(const Path &bra, const Path &ket, int p, int q);
double guga_two_body(const Path &bra, const Path &ket, int p, int q, int r, int s);
double guga_element(const Path &bra, const Path &ket, const Connection &connection);
double guga_segment(int da, int db, int a, int b, int p, int q, int site, int mode);
bool guga_connected_paths(const Config &target, const std::vector<Path> &parents,
                          const Connection &connection, size_t maximum_bytes,
                          const std::function<void(const std::string &)> &callback,
                          std::map<std::string,size_t> &stats);
void bind_walks(pybind11::module_ &module);
double guga_local_bound(const Config &ket, int p, int q, int r, int s, bool two_body);
void bind_guga(pybind11::module_ &module);
Array block(const Array &one, const Array &two, const std::vector<std::string> &rows,
            const std::vector<std::string> &cols, int two_s, size_t max_bond_elements,
            size_t max_tensor_entries, size_t max_connection_terms,
            std::shared_ptr<ConnectionCache> cache, bool compiled, bool guga, int guga_mode);
void bind_sci(pybind11::module_ &module);
void bind_products(pybind11::module_ &module);
} // namespace scsf

/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#pragma once
#include "structured_eri.h"

#include <pybind11/numpy.h>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
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
struct Occupation;
namespace structured { class Workers; }

struct Path {
    std::string label;
    std::vector<int> spins;
    int phase = 1; // diagonal conversion from segment-table to the CSF CG phase convention
    int electrons = 0;
    int guga_mode = 0; // 0 analytic reference, 1 factor table, 2 spectator runs
    std::vector<int> next_single, closed_prefix;

    Path(const std::string &steps, int k, int two_s, int factor_mode) : label(steps), guga_mode(factor_mode) {
        if (steps.size() != static_cast<size_t>(k))
            throw std::invalid_argument("CSF label length differs from orbital count");
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
    }
};

struct Operator {
    int orbital, spin;
    bool create;
};


struct GugaTerm {
    double coefficient;
    int p, q, r, s;
    bool two_body;
};

struct Connection {
    double scalar = 0.0;
    std::vector<GugaTerm> generators;
    double term_bound = 0.0;
};

struct BoundedTransfer {
    double bound;
    int p, r;
};

using Stats = std::map<std::string, size_t>;

struct BudgetError : std::runtime_error { using std::runtime_error::runtime_error; };

inline size_t memory_left(size_t limit,size_t used,const char *phase) {
    if(used>limit)
        throw BudgetError(std::string(phase)+" needs "+std::to_string(used)+
                          " bytes with "+std::to_string(limit)+" bytes available (--max-mem)");
    return limit-used;
}

class HCITransferBounds {
    Array one_, two_;
    structured::SpatialERI eri_;
    OrbitalSymmetry symmetry_;
    std::vector<double> transfer_bounds_,single_bounds_;
    mutable std::recursive_mutex table_lock_;
    std::once_flag bound_once_, ordered_once_, ordered_singles_once_;
    std::unique_ptr<BoundedTransfer[]> ordered_transfers_;
    size_t ordered_transfer_count_=0;
    std::vector<BoundedTransfer> ordered_singles_;
    std::vector<size_t> ordered_offsets_,ordered_single_offsets_,single_offsets_;
    int ordered_single_electrons_=0;
    void prepare_single_bounds(int k);
    size_t single_rows(int k) const;
    size_t single_row(int p,int q,int k) const;
public:
    HCITransferBounds(const Array &one, const Array &two);
    HCITransferBounds(const Array &one,const Array &two,structured::SpatialERI eri,
                      const OrbitalSymmetry &symmetry);
    std::unique_lock<std::recursive_mutex> lock_tables() const {
        return std::unique_lock<std::recursive_mutex>(table_lock_);
    }
    bool optional_tables() const {return eri_.format!=structured::ERIFormat::dense || symmetry_.enabled();}
    size_t bytes() const;
    size_t ordered_double_count() const {return ordered_transfer_count_;}
    size_t ordered_single_count() const {return ordered_singles_.size();}
    void prepare_transfer_bounds(int k,size_t limit);
    double selection_bound(const Config &source, const std::array<int,2> &holes,
                           const std::array<int,2> &particles, int order, int k) const;
    double single_bound(const Config &source,int p,int q,int k,const Occupation &key) const;
    bool prepare_ordered_transfers(int k,size_t limit,structured::Workers &workers);
    std::pair<const BoundedTransfer *, const BoundedTransfer *> ordered_transfers(int q, int s, int k) const;
    bool prepare_ordered_singles(int k,int nelec,size_t limit);
    std::pair<const BoundedTransfer *, const BoundedTransfer *> ordered_singles(int q,int k) const;
};

Config configuration(const std::string &label, int two_s);
inline void validate_config(const Config &config) {
    for (int n : config)
        if (n < 0 || n > 2) throw std::invalid_argument("occupations must be 0,1,2");
}

double transfer_bound(const Config &ket, const std::array<int, 2> &holes,
                      const std::array<int, 2> &particles, int order,
                      const double *h, const structured::SpatialERI &v, int k);
double guga_generator(const Path &bra, const Path &ket, int p, int q);
double guga_two_body(const Path &bra, const Path &ket, int p, int q, int r, int s);
double guga_element(const Path &bra, const Path &ket, const Connection &connection);
double guga_segment(int da, int db, int a, int b, int p, int q, int site, int mode);
bool guga_connected_paths(const Config &target, const std::vector<Path> &parents,
                          const Connection &connection, size_t maximum_bytes,
                          const std::function<void(const std::string &)> &callback,
                          std::map<std::string,size_t> &stats);
double guga_local_bound(const Config &ket, int p, int q, int r, int s, bool two_body);
void bind_one_electron(pybind11::module_ &module);
} // namespace scsf

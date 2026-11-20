/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#include "csf.h"
#include <algorithm>
#include <limits>

namespace scsf {
namespace {
// A spin-free term touches at most four spatial orbitals. In fixed local
// occupations there are at most 16 input and output spin states. Build exact
// integer matrix columns and bound ||A||_2 by sqrt(||A||_1 ||A||_inf).
double local_norm(int p, int q, int r, int s, int occupations, bool two_body) {
    int k = std::max({p,q,r,s})+1;
    int dimension = 1 << (2*k);
    std::array<int,256> rows{};
    int column_max = 0;
    for (int source = 0; source < dimension; ++source) {
        bool valid = true;
        for (int i = 0; i < k; ++i)
            if (__builtin_popcount(static_cast<unsigned>((source >> (2*i)) & 3)) !=
                ((occupations >> (2*i)) & 3)) { valid = false; break; }
        if (!valid) continue;
        std::array<int,256> column{};
        for (int sigma = 0; sigma < 2; ++sigma) for (int tau = 0; tau < (two_body ? 2 : 1); ++tau) {
            int state = source, sign = 1;
            std::array<int,4> sites{2*q+sigma,2*s+tau,2*r+tau,2*p+sigma};
            if (!two_body) sites[1] = 2*p+sigma;
            for (int o = 0; o < (two_body ? 4 : 2); ++o) {
                int bit = 1 << sites[o];
                bool create = o >= (two_body ? 2 : 1);
                if (bool(state & bit) == create) { sign = 0; break; }
                if (__builtin_popcount(static_cast<unsigned>(state & (bit-1))) & 1) sign = -sign;
                state ^= bit;
            }
            if (sign) column[state] += sign;
        }
        int sum = 0;
        for (int i = 0; i < dimension; ++i) {
            int magnitude = std::abs(column[i]);
            sum += magnitude;
            rows[i] += magnitude;
        }
        column_max = std::max(column_max,sum);
    }
    int row_max = *std::max_element(rows.begin(),rows.begin()+dimension);
    if (!column_max || !row_max) return 0.0;
    return std::nextafter(std::sqrt(static_cast<double>(column_max*row_max)),
                           std::numeric_limits<double>::infinity());
}

struct Norms {
    // Fixed 1 MiB shared table, independent of orbitals/integrals/spin. Cells
    // are filled lazily to keep small cold calculations cheap. Racing misses
    // may compute the same exact integer bound twice; no lock is necessary.
    std::array<std::atomic<double>,131072> cells;
    Norms() { for (auto &cell : cells) cell.store(-1.0,std::memory_order_relaxed); }
};
} // namespace

double guga_local_bound(const Config &ket, int p, int q, int r, int s, bool two_body) {
    if (!two_body) r = s = p;
    std::array<int,4> active{p,q,r,s};
    std::sort(active.begin(),active.end());
    int count = static_cast<int>(std::unique(active.begin(),active.end())-active.begin());
    int occ = 0;
    for (int i = 0; i < count; ++i) occ |= ket[active[i]] << (2*i);
    auto index = [&](int orbital) {
        return static_cast<int>(std::find(active.begin(),active.begin()+count,orbital)-active.begin());
    };
    int pattern = index(p)+(index(q) << 2);
    if (two_body) pattern += (index(r) << 4)+(index(s) << 6);
    static Norms table;
    auto &cell = table.cells[(two_body ? 65536 : 0)+(occ << 8)+pattern];
    double value = cell.load(std::memory_order_relaxed);
    if (value < 0.0) {
        value = local_norm(pattern & 3,(pattern >> 2) & 3,
                           two_body ? (pattern >> 4) & 3 : 0,
                           two_body ? (pattern >> 6) & 3 : 0,occ,two_body);
        cell.store(value,std::memory_order_relaxed);
    }
    return value;
}
} // namespace scsf

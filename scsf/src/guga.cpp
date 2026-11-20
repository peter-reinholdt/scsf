/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#include "csf.h"
#include <algorithm>

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
double generator(const Path &bra, const Path &ket, int p, int q) {
    if (p > q) return generator(ket, bra, q, p);
    if (p == q) return bra.label == ket.label ? occupation(ket.label[p]) : 0.0;
    if (bra.label.compare(0, p, ket.label, 0, p) ||
        bra.label.compare(q+1, std::string::npos, ket.label, q+1, std::string::npos) ||
        bra.spins[q+1] != ket.spins[q+1]) return 0.0;
    double value = 1.0;
    for (int i = p; i <= q; ++i) {
        if (bra.guga_mode == 2 && i > p && i < q &&
            bra.next_single[i] > i && ket.next_single[i] > i) {
            int stop = std::min({q,bra.next_single[i],ket.next_single[i]});
            if (bra.label.compare(i,stop-i,ket.label,i,stop-i) ||
                std::abs(bra.spins[i+1]-ket.spins[i+1]) != 1) return 0.0;
            if ((ket.closed_prefix[stop]-ket.closed_prefix[i])%2) value = -value;
            i = stop-1;
            continue;
        }
        value *= segment(step(bra.label[i]), step(ket.label[i]), bra.spins[i+1], ket.spins[i+1], p, q, i,bra.guga_mode);
        if (value == 0.0) break;
    }
    return value;
}

double product(const Path &bra, const Path &ket, int p, int q, int r, int s) {
    if (r == s) return occupation(ket.label[s])*generator(bra, ket, p, q);
    if (p == q) return occupation(bra.label[p])*generator(bra, ket, r, s);
    if (!occupation(ket.label[s]) || occupation(ket.label[r]) == 2) return 0.0;
    int begin = std::min({p,q,r,s}), end = std::max({p,q,r,s});
    // Outside both generators the two outer walks and the intermediate must
    // coincide. Reject once, then omit their exact identity transfers.
    if (bra.label.compare(0, begin, ket.label, 0, begin) ||
        bra.label.compare(end+1, std::string::npos, ket.label, end+1, std::string::npos) ||
        bra.spins[end+1] != ket.spins[end+1]) return 0.0;
    std::array<int, 2> spins{ket.spins[begin], -1};
    std::array<double, 2> weights{1.0, 0.0};
    int count = 1;
    int lo = std::min(r, s), hi = std::max(r, s);
    for (int i = begin; i <= end; ++i) {
        int b = ket.spins[i+1];
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
                double left = segment(step(bra.label[i]), d, bra.spins[i+1], after[j], p, q, i,bra.guga_mode);
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

double guga_generator(const Path &bra, const Path &ket, int p, int q) {
    return bra.phase*ket.phase*generator(bra, ket, p, q);
}

double guga_segment(int da, int db, int a, int b, int p, int q, int site, int mode) {
    return segment(da,db,a,b,p,q,site,mode);
}

double guga_two_body(const Path &bra, const Path &ket, int p, int q, int r, int s) {
    return bra.phase*ket.phase*(product(bra, ket, p, q, r, s)-
                               (q == r ? generator(bra, ket, p, s) : 0.0));
}

double guga_element(const Path &bra, const Path &ket, const Connection &connection) {
    double value = bra.label == ket.label ? connection.scalar : 0.0;
    for (const auto &term : connection.generators) {
        double coupling = term.two_body ? guga_two_body(bra, ket, term.p, term.q, term.r, term.s) :
                                         guga_generator(bra, ket, term.p, term.q);
        value += term.coefficient*coupling;
    }
    return value;
}

} // namespace scsf

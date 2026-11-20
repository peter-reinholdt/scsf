/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#include "csf.h"
#include <pybind11/stl.h>
#include <algorithm>
#include <unordered_set>

namespace scsf {
namespace {
int code(char c) { return c == 'u' ? 1 : c == 'd' ? 2 : c == '2' ? 3 : 0; }
int occ(char c) { return c == '2' ? 2 : c != '0'; }
int change(int d) { return (d == 1)-(d == 2); }
constexpr int states = 15, root = 7; // outer delta [-2,2], intermediate delta [-1,1]
using Mask = std::uint16_t;
struct Layer { std::array<std::array<Mask,2>,states> next{}; };
struct Counts {
    size_t graphs = 0, states = 0, edges = 0, branches = 0, dead = 0, leaves = 0,
           duplicates = 0, peak = 0;
};

// Boolean support of E[p,q] or E[p,q]E[r,s], without restricting the intermediate
// walk to the selected space. At any boundary the intermediate differs from
// the ket spin by at most one, and the outer bra by at most two. Backward masks
// retain only states which can reach the common closing boundary.
bool channel(const Config &target, const Path &ket, int p, int q, int r, int s, bool product,
             size_t maximum, size_t retained, const std::function<bool(const std::string &)> &emit, Counts &count) {
    int begin = std::min(p,q), end = std::max(p,q);
    if (product) { begin = std::min({begin,r,s}); end = std::max({end,r,s}); }
    for (int i = 0; i < static_cast<int>(target.size()); ++i)
        if ((i < begin || i > end) && target[i] != occ(ket.label[i])) return true;
    size_t length = end-begin+1;
    size_t bytes = length*sizeof(Layer)+(length+1)*sizeof(Mask)+ket.label.capacity()+1;
    if (bytes > maximum) return false;
    count.peak = std::max(count.peak,bytes+retained);
    ++count.graphs;
    std::vector<Layer> layers(length);
    std::vector<Mask> reachable(length+1,0);
    reachable[length] = Mask(1) << root;
    for (int i = end; i >= begin; --i) {
        auto &layer = layers[i-begin];
        int b0 = ket.spins[i], b1 = ket.spins[i+1];
        int nmiddle = occ(ket.label[i])+(i == r)-(i == s);
        for (int state = 0; state < states; ++state) {
            int a0 = b0+state/3-2, m0 = b0+state%3-1;
            if (a0 < 0 || m0 < 0 || (!product && state%3 != 1)) continue;
            ++count.states;
            for (int choice = 0; choice < (target[i] == 1 ? 2 : 1); ++choice) {
                int da = target[i] == 1 ? choice+1 : target[i] == 2 ? 3 : 0;
                int a1 = a0+change(da), delta = a1-b1;
                if (a1 < 0 || delta < -2 || delta > 2) continue;
                for (int md = -1; md <= 1; ++md) {
                    if (!product && md) continue;
                    int next = (delta+2)*3+md+1;
                    if (!(reachable[i-begin+1] & (Mask(1) << next))) continue;
                    int m1 = b1+md;
                    if (m1 < 0) continue;
                    int dm = code(ket.label[i]);
                    if (product) {
                        int diff = m1-m0;
                        if (nmiddle == 1 && std::abs(diff) == 1) dm = diff == 1 ? 1 : 2;
                        else if ((nmiddle == 0 || nmiddle == 2) && diff == 0) dm = nmiddle == 2 ? 3 : 0;
                        else continue;
                        if (guga_segment(dm,code(ket.label[i]),m1,b1,r,s,i,ket.guga_mode) == 0.) continue;
                    }
                    if (guga_segment(da,dm,a1,m1,p,q,i,ket.guga_mode) == 0.) continue;
                    layer.next[state][choice] |= Mask(1) << next;
                    reachable[i-begin] |= Mask(1) << state;
                    ++count.edges;
                }
            }
        }
    }
    if (!(reachable[0] & (Mask(1) << root))) { ++count.dead; return true; }
    std::string label = ket.label;
    auto visit = [&](auto &&self, int i, Mask mask) -> bool {
        if (i > end) { ++count.leaves; return emit(label); }
        const auto &layer = layers[i-begin];
        for (int choice = 0; choice < (target[i] == 1 ? 2 : 1); ++choice) {
            ++count.branches;
            Mask next = 0;
            for (int state = 0; state < states; ++state)
                if (mask & (Mask(1) << state)) next |= layer.next[state][choice];
            if (!next) { ++count.dead; continue; }
            label[i] = target[i] == 1 ? (choice ? 'd' : 'u') : target[i] == 2 ? '2' : '0';
            if (!self(self,i+1,next)) return false;
        }
        return true;
    };
    return visit(visit,begin,Mask(1) << root);
}
} // namespace

bool guga_connected_paths(const Config &target, const std::vector<Path> &parents,
                          const Connection &connection, size_t maximum_bytes,
                          const std::function<void(const std::string &)> &callback,
                          std::map<std::string,size_t> &stats) {
    Counts counts;
    std::unordered_set<std::string> emitted;
    size_t retained = 0;
    bool complete = true;
    auto emit = [&](const std::string &label) {
        if (emitted.count(label)) { ++counts.duplicates; return true; }
        size_t bytes = sizeof(std::string)+label.capacity()+1+128;
        // Include space for the next channel graph while retaining the union.
        size_t graph = target.size()*sizeof(Layer)+(target.size()+1)*sizeof(Mask)+label.capacity()+1;
        if (bytes > maximum_bytes-retained || graph > maximum_bytes-retained-bytes) return false;
        emitted.insert(label);
        retained += bytes;
        counts.peak = std::max(counts.peak,retained+graph);
        callback(label);
        return true;
    };
    for (const auto &ket : parents) {
        if (connection.scalar != 0.) {
            bool same = true;
            for (size_t i = 0; i < target.size(); ++i) same &= target[i] == occ(ket.label[i]);
            if (same && !emit(ket.label)) { complete = false; break; }
        }
        for (const auto &term : connection.generators) {
            if (term.coefficient == 0.) continue;
            if (!channel(target,ket,term.p,term.q,term.r,term.s,term.two_body,
                         maximum_bytes-retained,retained,emit,counts)) { complete = false; break; }
            // Normal ordering subtracts this channel; it is part of the union
            // even when the product's intermediate path has no support.
            if (term.two_body && term.q == term.r &&
                !channel(target,ket,term.p,term.s,0,0,false,
                         maximum_bytes-retained,retained,emit,counts)) { complete = false; break; }
        }
        if (!complete) break;
    }
    stats["walk_graphs"] += counts.graphs;
    stats["walk_states"] += counts.states;
    stats["walk_edges"] += counts.edges;
    stats["walk_branches"] += counts.branches;
    stats["walk_dead"] += counts.dead;
    stats["walk_leaves"] += counts.leaves;
    stats["walk_duplicates"] += counts.duplicates;
    stats["walk_peak_bytes"] = std::max(stats["walk_peak_bytes"],counts.peak);
    stats["walk_fallbacks"] += !complete;
    return complete;
}

} // namespace scsf

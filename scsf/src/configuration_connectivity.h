/* This file is part of SCSF, licensed under GNU GPL version 3 or later. */
#pragma once
#include "csf.h"

namespace scsf {
// Enumerate every minimal zero/one/two occupation transfer exactly once.
// Disjoint particle/hole multisets eliminate unchanged/spectator duplicates.
template<class Callback>
void transfers(const Config &source, Callback callback) {
    validate_config(source);
    callback(std::array<int, 2>{}, std::array<int, 2>{}, 0);
    int k = static_cast<int>(source.size());
    for (int q = 0; q < k; ++q) {
        if (!source[q]) continue;
        for (int p = 0; p < k; ++p)
            if (p != q && source[p] < 2) callback(std::array<int, 2>{q, 0}, std::array<int, 2>{p, 0}, 1);
        for (int s = q; s < k; ++s) {
            if (source[s] < (q == s ? 2 : 1)) continue;
            for (int p = 0; p < k; ++p) {
                if (p == q || p == s || source[p] == 2) continue;
                for (int r = p; r < k; ++r) {
                    if (r == q || r == s || source[r] >= (p == r ? 1 : 2)) continue;
                    callback(std::array<int, 2>{q, s}, std::array<int, 2>{p, r}, 2);
                }
            }
        }
    }
}

inline Config endpoint(const Config &source, const std::array<int, 2> &holes,
                const std::array<int, 2> &particles, int order) {
    Config target = source;
    for (int i = 0; i < order; ++i) { --target[holes[i]]; ++target[particles[i]]; }
    return target;
}

} // namespace scsf

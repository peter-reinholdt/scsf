/* This file is part of SCSF, licensed under GNU GPL version 3 or later. */
#pragma once
#include "selected_basis.h"
#include <set>

namespace scsf {
inline std::uint64_t occupation_key(const Config &config) {
    std::uint64_t key = 0;
    for (size_t p = 0; p < config.size(); ++p)
        key |= static_cast<std::uint64_t>(config[p]) << (2*p);
    return key;
}

struct Transfer {
    std::array<int, 2> holes, particles;
    int order;
};

// Enumerate every minimal zero/one/two occupation transfer exactly once.
// Disjoint particle/hole multisets eliminate unchanged/spectator duplicates.
// The count budget applies *before* any optional screening, as before.
template<class Callback>
void transfers(const Config &source, size_t maximum, Callback callback) {
    validate_config(source);
    size_t count = 0;
    auto emit = [&](std::array<int, 2> holes, std::array<int, 2> particles, int order) {
        if (count >= maximum) throw BudgetError("configuration search exceeds max_configurations");
        ++count;
        callback(holes, particles, order);
    };
    emit({}, {}, 0);
    int k = static_cast<int>(source.size());
    for (int q = 0; q < k; ++q) {
        if (!source[q]) continue;
        for (int p = 0; p < k; ++p)
            if (p != q && source[p] < 2) emit({q, 0}, {p, 0}, 1);
        for (int s = q; s < k; ++s) {
            if (source[s] < (q == s ? 2 : 1)) continue;
            for (int p = 0; p < k; ++p) {
                if (p == q || p == s || source[p] == 2) continue;
                for (int r = p; r < k; ++r) {
                    if (r == q || r == s || source[r] >= (p == r ? 1 : 2)) continue;
                    emit({q, s}, {p, r}, 2);
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

// Count canonical transfers without enumerating particle pairs. The same
// pre-screening guard applies even when a sorted-list tail is never visited.
inline size_t transfer_count(const Config &source, size_t maximum) {
    size_t count = 0, available = 0, empty = 0;
    for (int n : source) { available += n < 2; empty += n == 0; }
    auto add = [&](size_t n) {
        if (n > maximum-count) throw BudgetError("configuration search exceeds max_configurations");
        count += n;
    };
    add(1);
    for (size_t q = 0; q < source.size(); ++q) {
        if (!source[q]) continue;
        add(available-(source[q] < 2));
        for (size_t s = q; s < source.size(); ++s) {
            if (source[s] < (q == s ? 2 : 1)) continue;
            size_t slots = available-(source[q] < 2)-(s != q && source[s] < 2);
            add(empty); // Two particles can share only an initially empty site.
            size_t a = slots, b = slots ? slots-1 : 0;
            if (a%2) b /= 2; else a /= 2;
            if (b && a > (maximum-count)/b)
                throw BudgetError("configuration search exceeds max_configurations");
            add(a*b);
        }
    }
    return count;
}

inline std::set<Config> connected(const Config &source, size_t maximum) {
    std::set<Config> result;
    transfers(source, maximum, [&](const auto &holes, const auto &particles, int order) {
        result.insert(endpoint(source, holes, particles, order));
    });
    return result;
}

} // namespace scsf

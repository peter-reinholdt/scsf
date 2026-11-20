/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#include "csf.h"
#include <algorithm>

namespace scsf {
HCITransferBounds::HCITransferBounds(const Array &one, const Array &two, size_t ordered_cap)
    : one_(one), two_(two), ordered_cap_(ordered_cap) {
    if (one.ndim()!=2 || one.shape(0)!=one.shape(1) || two.ndim()!=4)
        throw std::invalid_argument("invalid HCI bound integral dimensions");
    for (int axis=0;axis<4;++axis)
        if (two.shape(axis)!=one.shape(0))
            throw std::invalid_argument("invalid HCI bound integral dimensions");
}

void HCITransferBounds::prepare_transfer_bounds(int k) {
    std::call_once(bound_once_,[&] {
        constexpr size_t max_entries = 1000000;
        if (k < 2) return;
        size_t count = 1;
        for (int i = 0; i < 4; ++i) {
            if (count > max_entries/static_cast<size_t>(k)) return;
            count *= static_cast<size_t>(k);
        }
        std::vector<double> values(count,0.0);
        // A minimal double transfer has disjoint hole/particle multisets.
        // Its coarse integral bound is independent of source occupations.
        const Config unused;
        for (int q = 0; q < k; ++q) for (int s = q; s < k; ++s)
            for (int p = 0; p < k; ++p) {
                if (p == q || p == s) continue;
                for (int r = p; r < k; ++r) {
                    if (r == q || r == s) continue;
                    size_t index = ((static_cast<size_t>(q)*k+s)*k+p)*k+r;
                    values[index] = transfer_bound(unused,{q,s},{p,r},2,one_.data(),two_.data(),k);
                }
            }
        transfer_bounds_.swap(values);
    });
}

double HCITransferBounds::selection_bound(const Config &source, const std::array<int,2> &holes,
                                       const std::array<int,2> &particles, int order, int k) const {
    if (order == 2 && !transfer_bounds_.empty()) {
        size_t index = ((static_cast<size_t>(holes[0])*k+holes[1])*k+particles[0])*k+particles[1];
        return transfer_bounds_[index];
    }
    return transfer_bound(source,holes,particles,order,one_.data(),two_.data(),k);
}

bool HCITransferBounds::prepare_ordered_transfers(int k) {
    std::call_once(ordered_once_, [&] {
        const size_t max_bytes = ordered_cap_;
        if (k < 2 || max_bytes/sizeof(size_t) < 2) return;
        const size_t extent = static_cast<size_t>(k);
        if (extent > (max_bytes/sizeof(size_t)-1)/extent) return;
        size_t offset_count = extent*extent+1;
        size_t available = max_bytes-offset_count*sizeof(size_t), count = 0;
        for (int q = 0; q < k; ++q) for (int s = q; s < k; ++s) {
            size_t slots = k-(q == s ? 1 : 2);
            size_t add = slots*(slots+1)/2;
            if (add > available/sizeof(BoundedTransfer)-count) return;
            count += add;
        }
        std::vector<BoundedTransfer> values;
        values.reserve(count);
        std::vector<size_t> offsets(offset_count);
        const Config unused;
        for (int q = 0; q < k; ++q) for (int s = 0; s < k; ++s) {
            size_t at = static_cast<size_t>(q)*k+s;
            offsets[at] = values.size();
            if (s < q) continue;
            for (int p = 0; p < k; ++p) {
                if (p == q || p == s) continue;
                for (int r = p; r < k; ++r) {
                    if (r == q || r == s) continue;
                    double bound = selection_bound(unused,{q,s},{p,r},2,k);
                    values.push_back({bound,p,r});
                }
            }
            std::sort(values.begin()+offsets[at], values.end(), [](const auto &a, const auto &b) {
                if (a.bound != b.bound) return a.bound > b.bound;
                return a.p != b.p ? a.p < b.p : a.r < b.r;
            });
        }
        offsets.back() = values.size();
        // Account allocated capacities, including the per-hole-pair offsets.
        if (values.capacity()*sizeof(BoundedTransfer)+offsets.capacity()*sizeof(size_t) > max_bytes) return;
        ordered_transfers_.swap(values);
        ordered_offsets_.swap(offsets);
    });
    return !ordered_offsets_.empty();
}

std::pair<const BoundedTransfer *, const BoundedTransfer *>
HCITransferBounds::ordered_transfers(int q, int s, int k) const {
    size_t at = static_cast<size_t>(q)*k+s;
    return {ordered_transfers_.data()+ordered_offsets_[at],
            ordered_transfers_.data()+ordered_offsets_[at+1]};
}

} // namespace scsf

/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#include "csf.h"
#include <algorithm>

namespace scsf {
HCITransferBounds::HCITransferBounds(const Array &one, const Array &two)
    : one_(one), two_(two) {
    if (one.ndim()!=2 || one.shape(0)!=one.shape(1) || two.ndim()!=4)
        throw std::invalid_argument("invalid HCI bound integral dimensions");
    for (int axis=0;axis<4;++axis)
        if (two.shape(axis)!=one.shape(0))
            throw std::invalid_argument("invalid HCI bound integral dimensions");
}

size_t HCITransferBounds::bytes() const {
    return (transfer_bounds_.capacity()+single_bounds_.capacity())*sizeof(double)
        +(ordered_transfers_.capacity()+ordered_singles_.capacity())*sizeof(BoundedTransfer)
        +ordered_offsets_.capacity()*sizeof(size_t);
}

void HCITransferBounds::prepare_transfer_bounds(int k,size_t limit) {
    std::call_once(bound_once_,[&] {
        constexpr size_t max_entries = 1000000;
        if (k < 2) return;
        if(k<=64 && single_bounds_.empty()) {
            memory_left(limit,size_t(k)*k*(k+1)*sizeof(double),"HCI single transfer bounds");
            prepare_single_bounds(k);
        }
        size_t count = 1;
        for (int i = 0; i < 4; ++i) {
            if (count > max_entries/static_cast<size_t>(k)) return;
            count *= static_cast<size_t>(k);
        }
        memory_left(limit,bytes()+count*sizeof(double),"HCI transfer bounds");
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

bool HCITransferBounds::prepare_ordered_transfers(int k,size_t limit) {
    std::call_once(ordered_once_, [&] {
        if (k < 2) return;
        const size_t max_bytes = memory_left(limit,bytes(),"HCI transfer bounds");
        const size_t extent = static_cast<size_t>(k);
        if (max_bytes/sizeof(size_t)<2 || extent>(max_bytes/sizeof(size_t)-1)/extent)
            throw BudgetError("ordered HCI offset table exceeds --max-mem");
        size_t offset_count = extent*extent+1;
        size_t available = max_bytes-offset_count*sizeof(size_t), count = 0;
        for (int q = 0; q < k; ++q) for (int s = q; s < k; ++s) {
            size_t slots = k-(q == s ? 1 : 2);
            size_t add = slots*(slots+1)/2;
            if (add > available/sizeof(BoundedTransfer)-count)
                throw BudgetError("ordered HCI double transfers exceed --max-mem");
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
        memory_left(max_bytes,values.capacity()*sizeof(BoundedTransfer)+offsets.capacity()*sizeof(size_t),
                    "ordered HCI double transfers");
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

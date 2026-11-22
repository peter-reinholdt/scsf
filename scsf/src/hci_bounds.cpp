/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#include "structured_internal.h"
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
        +(ordered_transfer_count_+ordered_singles_.capacity())*sizeof(BoundedTransfer)
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

bool HCITransferBounds::prepare_ordered_transfers(int k,size_t limit,structured::Workers &workers) {
    std::call_once(ordered_once_, [&] {
        if (k < 2) return;
        const size_t max_bytes = memory_left(limit,bytes(),"HCI transfer bounds");
        const size_t extent = static_cast<size_t>(k);
        if (max_bytes/sizeof(size_t)<2 || extent>(max_bytes/sizeof(size_t)-1)/extent)
            throw BudgetError("ordered HCI offset table exceeds --max-mem");
        size_t offset_count = extent*extent+1;
        size_t available = max_bytes-offset_count*sizeof(size_t), count = 0;
        std::vector<size_t> offsets(offset_count);
        for (int q = 0; q < k; ++q) for (int s = 0; s < k; ++s) {
            offsets[size_t(q)*k+s]=count;
            if (s < q) continue;
            const size_t slots=k-(q == s ? 1 : 2);
            const size_t add=slots*(slots+1)/2;
            if (add > available/sizeof(BoundedTransfer)-count)
                throw BudgetError("ordered HCI double transfers exceed --max-mem");
            count+=add;
        }
        offsets.back()=count;
        // Each hole pair owns an independent range. Fill it on its worker
        // before sorting, without initializing the entire table on the caller.
        std::unique_ptr<BoundedTransfer[]> values(new BoundedTransfer[count]);
        const Config unused;
        workers.run(offset_count-1,[&](size_t at,size_t) {
            if (offsets[at]==offsets[at+1]) return;
            const int q=at/k,s=at%k;
            auto *first=values.get()+offsets[at],*out=first;
            for (int p = 0; p < k; ++p) {
                if (p == q || p == s) continue;
                for (int r = p; r < k; ++r) {
                    if (r == q || r == s) continue;
                    double bound = selection_bound(unused,{q,s},{p,r},2,k);
                    *out++={bound,p,r};
                }
            }
            std::sort(first,out,[](const auto &a, const auto &b) {
                if (a.bound != b.bound) return a.bound > b.bound;
                return a.p != b.p ? a.p < b.p : a.r < b.r;
            });
        },4);
        memory_left(max_bytes,count*sizeof(BoundedTransfer)+offsets.capacity()*sizeof(size_t),
                    "ordered HCI double transfers");
        ordered_transfers_.swap(values);
        ordered_transfer_count_=count;
        ordered_offsets_.swap(offsets);
    });
    return !ordered_offsets_.empty();
}

std::pair<const BoundedTransfer *, const BoundedTransfer *>
HCITransferBounds::ordered_transfers(int q, int s, int k) const {
    size_t at = static_cast<size_t>(q)*k+s;
    return {ordered_transfers_.get()+ordered_offsets_[at],
            ordered_transfers_.get()+ordered_offsets_[at+1]};
}

} // namespace scsf

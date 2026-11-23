/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#include "structured_internal.h"
#include "configuration_connectivity.h"
#include <algorithm>

namespace scsf {
HCITransferBounds::HCITransferBounds(const Array &one, const Array &two)
    : one_(one), two_(two),eri_{two.data(),size_t(one.ndim()==2 ? one.shape(0) : 0),
          structured::ERIFormat::dense,{}} {
    if (one.ndim()!=2 || one.shape(0)!=one.shape(1) || two.ndim()!=4)
        throw std::invalid_argument("invalid HCI bound integral dimensions");
    for (int axis=0;axis<4;++axis)
        if (two.shape(axis)!=one.shape(0))
            throw std::invalid_argument("invalid HCI bound integral dimensions");
}

HCITransferBounds::HCITransferBounds(const Array &one,const Array &two,structured::SpatialERI eri,
                                    const OrbitalSymmetry &symmetry)
    :one_(one),two_(two),eri_(std::move(eri)),symmetry_(symmetry) {}

size_t HCITransferBounds::bytes() const {
    const auto lock=lock_tables();
    return (transfer_bounds_.capacity()+single_bounds_.capacity())*sizeof(double)
        +(ordered_transfer_count_+ordered_singles_.capacity())*sizeof(BoundedTransfer)
        +(ordered_offsets_.capacity()+ordered_single_offsets_.capacity()+single_offsets_.capacity())*sizeof(size_t);
}

size_t HCITransferBounds::single_rows(int k) const {
    if(!symmetry_.nontrivial())return size_t(k)*k;
    size_t count=0;
    for(const auto &group:symmetry_.orbitals)count+=group.size()*group.size();
    return count;
}

size_t HCITransferBounds::single_row(int p,int q,int k) const {
    return single_offsets_.empty() ? size_t(p)*k+q : single_offsets_[q]+symmetry_.positions[p];
}

void HCITransferBounds::prepare_transfer_bounds(int k,size_t limit) {
    std::call_once(bound_once_,[&] {
        constexpr size_t max_entries = 1000000;
        if (k < 2) return;
        if(k<=64 && single_bounds_.empty()) {
            memory_left(limit,single_rows(k)*(k+1)*sizeof(double)
                +(symmetry_.nontrivial() ? size_t(k)*sizeof(size_t) : 0),"HCI single transfer bounds");
            prepare_single_bounds(k);
        }
        if(optional_tables())return;
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
                    values[index] = transfer_bound(unused,{q,s},{p,r},2,one_.data(),eri_,k);
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
    return transfer_bound(source,holes,particles,order,one_.data(),eri_,k);
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
        std::array<size_t,OrbitalSymmetry::extent> pair_counts{};
        if(symmetry_.nontrivial())
            for(unsigned g=0;g<pair_counts.size();++g)
                for(unsigned h=g;h<pair_counts.size();++h) {
                    const size_t a=symmetry_.orbitals[g].size(),b=symmetry_.orbitals[h].size();
                    pair_counts[g^h]+=g==h ? structured::triangle_size(a) : a*b;
                }
        for (int q = 0; q < k; ++q) for (int s = 0; s < k; ++s) {
            offsets[size_t(q)*k+s]=count;
            if (s < q) continue;
            const size_t slots=k-(q == s ? 1 : 2);
            size_t add=structured::triangle_size(slots);
            if(symmetry_.nontrivial()) {
                const auto code=symmetry_.transfer(q,s);
                add=pair_counts[code]-symmetry_.orbitals[code^symmetry_[q]].size();
                if(q!=s)add-=symmetry_.orbitals[code^symmetry_[s]].size()-1;
            }
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
            particle_pairs(symmetry_,k,q,s,[&](int p,int r) {
                double bound=selection_bound(unused,{q,s},{p,r},2,k);
                *out++={bound,p,r};
            });
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
    if(!ordered_transfer_count_)return {nullptr,nullptr};
    return {ordered_transfers_.get()+ordered_offsets_[at],
            ordered_transfers_.get()+ordered_offsets_[at+1]};
}

} // namespace scsf

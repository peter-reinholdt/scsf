/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#include "csf.h"
#include <algorithm>

namespace scsf {
void ConnectionCache::validate(const Array &one, const Array &two) const {
    if (one.data() != one_.data() || two.data() != two_.data() ||
        one.size() != one_.size() || two.size() != two_.size())
        throw std::invalid_argument("connection cache belongs to a different integral snapshot");
}

std::shared_ptr<const Connection> ConnectionCache::get(
    const Config &bra, const Config &ket, const double *h, const double *v,
    int k, size_t max_terms, bool guga) {
    if (guga != guga_) throw std::invalid_argument("connection cache belongs to a different coupling engine");
    if (!maximum_) {
        // With retention disabled, neither a key nor a shared LRU is needed.
        // Keep logical miss statistics without serializing independent workers.
        misses_.fetch_add(1, std::memory_order_relaxed);
        return std::make_shared<Connection>(connection(bra, ket, h, v, k, max_terms, guga));
    }
    Key key{bra, ket};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = entries_.find(key);
        if (found != entries_.end()) {
            if (found->second.value->term_count() > max_terms)
                throw BudgetError("operator terms exceed max_connection_terms");
            ++hits_;
            order_.splice(order_.begin(), order_, found->second.position);
            return found->second.value;
        }
        ++misses_;
    }
    auto prepared = std::make_shared<Connection>(connection(bra, ket, h, v, k, max_terms, guga));
    // Account vector capacities, keys, objects, plus conservative tree/list
    // bookkeeping. Allocator overhead/RSS and the one active handle per worker
    // are separate from the retained-cache byte limit.
    size_t bytes = sizeof(Connection)+prepared->terms.capacity()*sizeof(Term)+
                   prepared->generators.capacity()*sizeof(GugaTerm)+
                   prepared->representatives.capacity()*sizeof(size_t)+
                   prepared->open_sites.capacity()*sizeof(int)+
                   prepared->action_signs.capacity()+prepared->local_maps.capacity()+
                   4*bra.size()*sizeof(int)+sizeof(Entry)+sizeof(Key)*2+256;
    if (bytes > maximum_) return prepared;
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = entries_.find(key);
    if (found != entries_.end()) return found->second.value;
    while (bytes_ > maximum_-bytes) {
        auto last = entries_.find(order_.back());
        bytes_ -= last->second.bytes;
        entries_.erase(last);
        order_.pop_back();
        ++evictions_;
    }
    order_.push_front(key);
    try {
        entries_.emplace(std::move(key), Entry{prepared, bytes, order_.begin()});
    } catch (...) {
        order_.pop_front();
        throw;
    }
    bytes_ += bytes;
    return prepared;
}

void ConnectionCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    order_.clear();
    bytes_ = hits_ = evictions_ = 0;
    misses_.store(0, std::memory_order_relaxed);
}

std::map<std::string, size_t> ConnectionCache::info() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {{"bytes", bytes_}, {"max_bytes", maximum_}, {"entries", entries_.size()},
            {"hits", hits_}, {"misses", misses_.load(std::memory_order_relaxed)}, {"evictions", evictions_}};
}

void ConnectionCache::prepare_transfer_bounds(int k) {
    std::call_once(bound_once_,[&] {
        if (k < 2 || !max_transfer_bounds_) return;
        size_t count = 1;
        for (int i = 0; i < 4; ++i) {
            if (count > max_transfer_bounds_/static_cast<size_t>(k)) return;
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
        bound_entries_.store(count,std::memory_order_release);
    });
}

double ConnectionCache::selection_bound(const Config &source, const std::array<int,2> &holes,
                                       const std::array<int,2> &particles, int order, int k, bool &reused) const {
    if (order == 2 && bound_entries_.load(std::memory_order_acquire)) {
        reused = true;
        size_t index = ((static_cast<size_t>(holes[0])*k+holes[1])*k+particles[0])*k+particles[1];
        return transfer_bounds_[index];
    }
    reused = false;
    return transfer_bound(source,holes,particles,order,one_.data(),two_.data(),k);
}

std::map<std::string,size_t> ConnectionCache::transfer_info() const {
    size_t entries = bound_entries_.load(std::memory_order_acquire);
    return {{"entries",entries},{"bytes",entries*sizeof(double)},{"max_entries",max_transfer_bounds_}};
}

bool ConnectionCache::prepare_ordered_transfers(int k, size_t max_bytes) {
    std::call_once(ordered_once_, [&] {
        ordered_cap_ = max_bytes;
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
                    bool reused;
                    double bound = selection_bound(unused,{q,s},{p,r},2,k,reused);
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
        ordered_ready_.store(true,std::memory_order_release);
    });
    if (max_bytes != ordered_cap_.load(std::memory_order_acquire))
        throw std::invalid_argument("sorted-transfer cap is fixed after first use; rebuild the backend");
    return ordered_ready_.load(std::memory_order_acquire);
}

std::pair<const BoundedTransfer *, const BoundedTransfer *>
ConnectionCache::ordered_transfers(int q, int s, int k) const {
    size_t at = static_cast<size_t>(q)*k+s;
    return {ordered_transfers_.data()+ordered_offsets_[at],
            ordered_transfers_.data()+ordered_offsets_[at+1]};
}

std::map<std::string,size_t> ConnectionCache::search_info() const {
    bool ready = ordered_ready_.load(std::memory_order_acquire);
    return {{"ordered_ready",ready},{"ordered_entries",ready ? ordered_transfers_.size() : 0},
            {"ordered_bytes",ready ? ordered_transfers_.capacity()*sizeof(BoundedTransfer)+
                                    ordered_offsets_.capacity()*sizeof(size_t) : 0},
            {"max_ordered_bytes",ordered_cap_.load(std::memory_order_acquire)}};
}
} // namespace scsf

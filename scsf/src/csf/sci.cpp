/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#include "csf.h"
#include "selected_basis.h"
#include "selected_index.h"
#include "configuration_connectivity.h"
#include "pair_descriptor.h"
#include "block_application.h"
#include "selected_spin_trie.h"
#include "parallel.h"
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <complex>
#include <thread>
#include <exception>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace py = pybind11;
namespace scsf {
namespace {

// Nanoseconds are accumulated per stage (worker time, not wall time).
struct Timer {
    size_t &value;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    explicit Timer(size_t &counter) : value(counter) {}
    ~Timer() {
        value += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-start).count();
    }
};

// Every task owns its tensors and writes disjoint rows/configurations. Only
// cache access, the candidate/selected sets, and the CSR entry count are shared.
// Workers join on every exception before any Python object is touched again.
template<class Callback>
void parallel_for(size_t count, size_t workers, Callback callback) {
    if (!count) return;
    workers = std::min(workers, count);
    if (workers == 1) {
        for (size_t i = 0; i < count; ++i) callback(i, 0);
        return;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> stop{false};
    std::mutex mutex;
    std::exception_ptr failure;
    auto run = [&](size_t worker) {
        try {
            while (!stop) {
                size_t i = next.fetch_add(1);
                if (i >= count) break;
                callback(i, worker);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!failure) failure = std::current_exception();
            stop = true;
        }
    };
    std::vector<std::thread> threads;
    try {
        for (size_t w = 1; w < workers; ++w) threads.emplace_back(run, w);
        run(0);
    } catch (...) {
        stop = true;
        for (auto &thread : threads) thread.join();
        throw;
    }
    for (auto &thread : threads) thread.join();
    if (failure) std::rethrow_exception(failure);
}

// Reuse threads across the thousands of source configurations in one HCI call.
// Each batch still has a barrier: even on failure every worker must finish
// before its callback and referenced source/tensor storage may go out of scope.
class SelectionWorkers {
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable ready, finished;
    bool shutdown = false;
    size_t generation = 0, completed = 0, count = 0, active = 0;
    std::atomic<size_t> next{0};
    std::atomic<bool> stop{false};
    std::exception_ptr failure;
    std::function<void(size_t, size_t)> callback;

    void execute(size_t worker) {
        try {
            while (!stop) {
                size_t task = next.fetch_add(1);
                if (task >= count) break;
                callback(task, worker);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!failure) failure = std::current_exception();
            stop = true;
        }
    }

    void run(size_t worker) {
        size_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            ready.wait(lock, [&] { return shutdown || generation != seen; });
            if (shutdown) return;
            seen = generation;
            bool participate = worker < active;
            lock.unlock();
            if (participate) execute(worker);
            lock.lock();
            ++completed;
            finished.notify_one();
        }
    }

public:
    explicit SelectionWorkers(size_t workers) {
        try {
            for (size_t w = 1; w < workers; ++w) threads.emplace_back([this, w] { run(w); });
        } catch (...) {
            { std::lock_guard<std::mutex> lock(mutex); shutdown = true; }
            ready.notify_all();
            for (auto &thread : threads) thread.join();
            throw;
        }
    }

    ~SelectionWorkers() {
        { std::lock_guard<std::mutex> lock(mutex); shutdown = true; }
        ready.notify_all();
        for (auto &thread : threads) thread.join();
    }

    void apply(size_t tasks, size_t workers, std::function<void(size_t, size_t)> job) {
        if (workers == 1 || !tasks) {
            for (size_t i = 0; i < tasks; ++i) job(i, 0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            callback = std::move(job);
            count = tasks;
            active = workers;
            completed = 0;
            next = 0;
            stop = false;
            failure = nullptr;
            ++generation;
        }
        ready.notify_all();
        execute(0);
        std::unique_lock<std::mutex> lock(mutex);
        finished.wait(lock, [&] { return completed == threads.size(); });
        callback = {};
        if (failure) std::rethrow_exception(failure);
    }
};

// A target owns its publication lock; unrelated occupations never serialize
// on a global selected-label set. Registry keys and State addresses stay valid
// until every source worker has joined. Only integer guards are global atomics.
class SharedSelection {
    struct State {
        std::mutex mutex;
        std::unordered_set<std::string> labels;
        const size_t missing;
        bool accounted = false, promoted = false;
        std::atomic<bool> complete{false};
        explicit State(size_t n) : missing(n) {}
    };
    struct ConfigHash {
        size_t operator()(const Config &c) const {
            size_t value = 1469598103934665603ULL;
            for (int n : c) value = (value ^ static_cast<size_t>(n+1))*1099511628211ULL;
            return value;
        }
    };
    struct alignas(64) Shard {
        std::mutex mutex;
        std::unordered_map<Config,std::unique_ptr<State>,ConfigHash> states;
    };
    std::vector<std::unique_ptr<Shard>> shards;
    const Groups &internal;
    const std::vector<size_t> &dimensions;
    const size_t candidate_limit, selected_limit;
    alignas(64) std::atomic<size_t> candidate_count{0};
    alignas(64) std::atomic<size_t> selected_count{0};

    State *state(const Config &target, bool create) {
        auto &shard = *shards[ConfigHash{}(target) & (shards.size()-1)];
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto found = shard.states.find(target);
        if (found != shard.states.end()) return found->second.get();
        if (!create) return nullptr;
        auto covered = internal.find(target);
        size_t missing = dimensions[std::count(target.begin(),target.end(),1)]-
                         (covered == internal.end() ? 0 : covered->second.size());
        auto value = std::make_unique<State>(missing);
        auto pointer = value.get();
        shard.states.emplace(target,std::move(value));
        return pointer;
    }
    static void reserve(std::atomic<size_t> &counter, size_t add, size_t maximum, const char *message) {
        size_t previous = counter.load(std::memory_order_relaxed);
        do {
            if (add > maximum-previous) throw BudgetError(message);
        } while (!counter.compare_exchange_weak(previous,previous+add,std::memory_order_relaxed));
    }
    void account_locked(State &value) {
        if (value.accounted) return;
        reserve(candidate_count,value.missing,candidate_limit,"HCI search exceeds max_candidates");
        value.accounted = true;
    }
public:
    SharedSelection(const Groups &groups, const std::vector<size_t> &dims,
                    size_t workers, size_t candidates, size_t selected)
        : internal(groups), dimensions(dims), candidate_limit(candidates), selected_limit(selected) {
        size_t count = 64;
        while (count < 4*workers) count *= 2;
        for (size_t i = 0; i < count; ++i) shards.push_back(std::make_unique<Shard>());
    }
    void account(const Config &target) {
        auto &value = *state(target,true);
        std::lock_guard<std::mutex> lock(value.mutex);
        account_locked(value);
    }
    bool complete(const Config &target) {
        auto value = state(target,false);
        return value && value->complete.load(std::memory_order_acquire);
    }
    auto target(const Config &key) { return state(key,true); }
    bool contains(State *handle, const std::string &label) {
        auto &value = *handle;
        if (value.complete.load(std::memory_order_acquire)) return true;
        std::lock_guard<std::mutex> lock(value.mutex);
        return value.labels.count(label);
    }
    void insert(State *handle, const std::string &label) {
        auto &value = *handle;
        std::lock_guard<std::mutex> lock(value.mutex);
        if (value.labels.count(label)) return;
        reserve(selected_count,1,selected_limit,"selected basis exceeds max_csfs");
        value.labels.insert(label);
        if (value.labels.size() == value.missing) value.complete.store(true,std::memory_order_release);
    }
    void promote(const Config &target) { promote(state(target,true)); }
    void promote(State *handle) {
        auto &value = *handle;
        std::lock_guard<std::mutex> lock(value.mutex);
        if (value.promoted) return;
        account_locked(value);
        reserve(selected_count,value.missing,selected_limit,"selected basis exceeds max_csfs");
        value.promoted = true;
        value.complete.store(true,std::memory_order_release);
    }
    void publish(std::set<std::string> &labels, std::set<Config> &promoted) const {
        for (const auto &shard : shards) for (const auto &item : shard->states) {
            labels.insert(item.second->labels.begin(),item.second->labels.end());
            if (item.second->promoted) promoted.insert(item.first);
        }
    }
    size_t candidates() const { return candidate_count.load(); }
    size_t selected() const { return selected_count.load(); }
    size_t shard_count() const { return shards.size(); }
};

std::shared_ptr<const Connection> prepare(
    const Config &a, const Config &b, const double *h, const double *v, int k,
    size_t maximum, const std::shared_ptr<ConnectionCache> &cache, bool guga) {
    if (cache) return cache->get(a, b, h, v, k, maximum, guga);
    return std::make_shared<Connection>(connection(a, b, h, v, k, maximum, guga));
}

void validate_workers(size_t workers) {
    if (!workers || workers > 256)
        throw std::invalid_argument("nthreads must be between 1 and 256");
}

int validate_integrals(const Array &one, const Array &two) {
    if (one.ndim() != 2 || one.shape(0) != one.shape(1) ||
        one.shape(0) > std::numeric_limits<int>::max())
        throw std::invalid_argument("one-electron array must be square");
    int k = static_cast<int>(one.shape(0));
    if (two.ndim() != 4)
        throw std::invalid_argument("two-electron array must have four spatial-orbital axes");
    for (int p = 0; p < 4; ++p)
        if (two.shape(p) != k) throw std::invalid_argument("integral orbital counts differ");
    return k;
}

struct ReuseCount { size_t pairs = 0, elements = 0, actions = 0; };
using ReuseHistogram = std::map<std::string,ReuseCount>;

std::string reduced_pattern(const Config &bra, const Config &ket) {
    std::string key;
    for (size_t p = 0; p < bra.size(); ++p) {
        if (bra[p] == ket[p] && bra[p] != 1) continue;
        // The ordered pair, rather than physical orbital position. Common
        // open shells are retained; CG phases are deliberately not inferred.
        key.push_back('0'+3*bra[p]+ket[p]);
    }
    return key;
}

template<class Callback>
void paths(const Config &config, int two_s, Callback callback) {
    int k = static_cast<int>(config.size());
    std::vector<int> remaining(k+1, 0);
    for (int p = k-1; p >= 0; --p)
        remaining[p] = remaining[p+1]+(config[p] == 1);
    std::string label(k, '0');
    std::function<void(int, int)> visit = [&](int p, int spin) {
        if (std::abs(spin-two_s) > remaining[p] || (spin+remaining[p]-two_s)%2) return;
        if (p == k) { callback(label); return; }
        if (config[p] != 1) {
            label[p] = config[p] == 0 ? '0' : '2';
            visit(p+1, spin);
        } else {
            label[p] = 'u';
            visit(p+1, spin+1);
            if (spin) { label[p] = 'd'; visit(p+1, spin-1); }
        }
    };
    visit(0, 0);
}

template<class Callback>
void paths_until(const Config &config, int two_s, Callback callback) {
    std::vector<int> remaining(config.size()+1,0);
    for (int p = static_cast<int>(config.size())-1; p >= 0; --p)
        remaining[p] = remaining[p+1]+(config[p] == 1);
    std::string label(config.size(),'0');
    auto visit = [&](auto &&self, size_t p, int spin) -> bool {
        if (std::abs(spin-two_s) > remaining[p] || (spin+remaining[p]-two_s)%2) return true;
        if (p == config.size()) return callback(label);
        if (config[p] != 1) {
            label[p] = config[p] ? '2' : '0';
            return self(self,p+1,spin);
        }
        label[p] = 'u';
        if (!self(self,p+1,spin+1)) return false;
        if (spin) { label[p] = 'd'; return self(self,p+1,spin-1); }
        return true;
    };
    visit(visit,0,0);
}

// Ballot-path dimensions for each open-shell count, saturated above the
// largest permitted selected basis. No floating-point equality is used to
// decide that an occupation block is complete.
std::vector<size_t> spin_dimensions(int maximum, int two_s, size_t cap) {
    std::vector<size_t> dimensions(maximum+1, 0), current(maximum+2, 0), next(maximum+2, 0);
    current[0] = 1;
    for (int m = 0; m <= maximum; ++m) {
        if (two_s <= maximum) dimensions[m] = current[two_s];
        std::fill(next.begin(), next.end(), 0);
        for (int j = 0; j <= m; ++j) {
            auto add = [&](int spin) {
                next[spin] = current[j] > cap-next[spin] ? cap : next[spin]+current[j];
            };
            if (j < maximum) add(j+1);
            if (j) add(j-1);
        }
        current.swap(next);
    }
    return dimensions;
}

bool screen_bound(double bound, double coefficient, double eps, int k) {
    // Both the coarse transfer and prepared projected-action bounds include
    // outward rounding and a contraction-error envelope.
    // Evaluate the floating-point boundary band rather than pruning it.
    // External target paths cannot equal an internal parent. The occupation
    // identity actions therefore contribute zero within their configuration.
    if (bound == 0.0) return true;
    long double product = static_cast<long double>(bound)*coefficient;
    long double margin = 512.0L*std::numeric_limits<double>::epsilon()*(k+1)*
                         (std::abs(product)+eps);
    return product+margin < static_cast<long double>(eps);
}

py::tuple select(const Array &one, const Array &two, const std::vector<std::string> &labels,
                 int two_s, const Array &coefficients, double eps, bool screening,
                 size_t max_candidates, size_t max_csfs, size_t max_configurations,
                 size_t max_bond_elements, size_t max_tensor_entries,
                 size_t max_connection_terms, std::shared_ptr<ConnectionCache> cache,
                 size_t nthreads, bool prescreen, bool compiled, bool guga,
                 bool ordered_search, size_t max_ordered_bytes, int guga_mode,
                 int walk_engine, size_t max_walk_bytes, int selection_rule, double parent_fraction, bool parallel_shared, bool return_groups) {
    int k = validate_integrals(one, two);
    validate_workers(nthreads);
    if (guga_mode < 0 || guga_mode > 2) throw std::invalid_argument("invalid guga_mode");
    if (walk_engine < 0 || walk_engine > 3 || (walk_engine && !guga))
        throw std::invalid_argument("connected walks require GUGA and walk_engine 0..3");
    if (selection_rule < 0 || selection_rule > 3)
        throw std::invalid_argument("selection_rule must be 0..3");
    if (!std::isfinite(parent_fraction) || parent_fraction <= 0.0 || parent_fraction > 1.0)
        throw std::invalid_argument("parent_fraction must be finite and in (0,1]");
    if (cache) cache->validate(one, two);
    if (coefficients.ndim() != 1 ||
        coefficients.shape(0) != static_cast<py::ssize_t>(labels.size()) ||
        !std::isfinite(eps) || eps < 0.0 || two_s < 0 || labels.size() > max_csfs)
        throw std::invalid_argument("invalid native CSF selection inputs");
    const double *coeffs = coefficients.data(), *h = one.data(), *v = two.data();
    for (size_t i = 0; i < labels.size(); ++i)
        if (!std::isfinite(coeffs[i])) throw std::invalid_argument("coefficients must be finite");
    Stats stats{{"configuration_pairs", 0}, {"screened_pairs", 0}, {"candidates", 0},
                {"matrix_elements", 0}, {"operator_terms", 0}, {"added", 0},
                {"prescreened_pairs", 0}, {"prepared_pairs", 0},
                {"complete_pairs", 0}, {"unreachable_pairs", 0}};
    std::set<std::string> selected, candidates;
    std::set<Config> accounted_targets;
    size_t logical_candidates = 0;
    std::set<Config> promoted;
    size_t promoted_additions = 0;
    double retained_weight = 0.0, total_weight = 0.0;
    stats["selection_rule"] = selection_rule;
    {
        py::gil_scoped_release release;
        try {
            if (((screening && prescreen) || selection_rule == 3) && cache) {
                Timer timer(stats["bound_setup_ns"]);
                cache->prepare_transfer_bounds(k);
            }
            bool ordered_ready = false;
            if (ordered_search && ((screening && prescreen) || selection_rule == 3) && cache) {
                Timer timer(stats["ordered_setup_ns"]);
                ordered_ready = cache->prepare_ordered_transfers(k,max_ordered_bytes);
            }
            stats["ordered_search"] = ordered_ready;
            Groups sources;
            { Timer timer(stats["index_ns"]); sources = groups(labels, k, two_s, 0, labels.size()); }
            std::set<std::string> internal(labels.begin(), labels.end());
            std::map<Config, size_t> covered;
            for (const auto &source : sources) covered.emplace(source.first, source.second.size());
            int electrons = sources.empty() ? 0 :
                std::accumulate(sources.begin()->first.begin(), sources.begin()->first.end(), 0);
            int max_open = std::min(electrons, 2*k-electrons);
            size_t dimension_cap = walk_engine || selection_rule || parallel_shared ? std::max(max_csfs,max_candidates) : max_csfs;
            if (dimension_cap != std::numeric_limits<size_t>::max()) ++dimension_cap;
            auto dimensions = spin_dimensions(max_open, two_s, dimension_cap);
            bool source_parallel = false;
            std::unique_ptr<SharedSelection> shared_selection;
            auto missing = [&](const Config &target) {
                auto found = sources.find(target);
                return dimensions[std::count(target.begin(),target.end(),1)]-
                       (found == sources.end() ? 0 : found->second.size());
            };
            // Called serially or under selected_mutex for witness tasks.
            auto account = [&](const Config &target) {
                if (source_parallel) { shared_selection->account(target); return; }
                if (accounted_targets.count(target)) return;
                size_t add = missing(target);
                if (add > max_candidates-logical_candidates)
                    throw BudgetError("HCI search exceeds max_candidates");
                logical_candidates += add;
                accounted_targets.insert(target);
            };
            auto promote = [&](const Config &target) {
                if (source_parallel) { shared_selection->promote(target); return; }
                if (promoted.count(target)) return;
                account(target);
                size_t add = missing(target);
                if (add > max_csfs-labels.size()-promoted_additions)
                    throw BudgetError("selected basis exceeds max_csfs");
                promoted_additions += add;
                promoted.insert(target);
                covered[target] = dimensions[std::count(target.begin(),target.end(),1)];
            };
            // Large parents first, with a stable configuration-key tie break.
            // This reduces repeated testing of targets selected by earlier parents.
            std::vector<const Groups::value_type *> ordered;
            for (const auto &source : sources) ordered.push_back(&source);
            if (parent_fraction < 1.0) {
                // Reuse the immutable occupation index. Scale before squaring,
                // so coefficient normalization/underflow is not an assumption.
                double scale = 0.0;
                for (size_t i = 0; i < labels.size(); ++i)
                    scale = std::max(scale,std::abs(coeffs[i]));
                std::vector<std::pair<double,const Groups::value_type *>> weighted;
                if (scale) for (const auto *source : ordered) {
                    double norm = 0.0;
                    for (size_t i : source->second) {
                        double c = coeffs[i]/scale;
                        norm += c*c;
                    }
                    if (norm) weighted.emplace_back(norm,source);
                }
                std::sort(weighted.begin(),weighted.end(),[](const auto &a,const auto &b) {
                    return a.first != b.first ? a.first > b.first : a.second->first < b.second->first;
                });
                for (const auto &item : weighted) total_weight += item.first;
                stats["parent_configurations"] = weighted.size();
                ordered.clear();
                for (const auto &item : weighted) {
                    if (retained_weight >= parent_fraction*total_weight) break;
                    ordered.push_back(item.second);
                    retained_weight += item.first;
                    for (size_t i : item.second->second)
                        stats["retained_parent_csfs"] += coeffs[i] != 0.0;
                }
                stats["retained_parent_configurations"] = ordered.size();
                // Restore the reference key tie break before sorting parents
                // by their maximum coefficient below.
                std::sort(ordered.begin(),ordered.end(),[](const auto *a,const auto *b) {
                    return a->first < b->first;
                });
            }
            auto weight = [&](const auto *source) {
                double maximum = 0.0;
                for (size_t i : source->second) maximum = std::max(maximum, std::abs(coeffs[i]));
                return maximum;
            };
            std::stable_sort(ordered.begin(), ordered.end(), [&](const auto *a, const auto *b) {
                return weight(a) > weight(b);
            });
            // Small parent sets retain target-level parallelism. Otherwise each
            // source is a complete pipeline job: no per-source team barrier.
            source_parallel = parallel_shared && nthreads > 1 && ordered.size() >= 2*nthreads;
            size_t source_workers = source_parallel ? std::min(nthreads,ordered.size()) : 1;
            if (source_parallel)
                shared_selection = std::make_unique<SharedSelection>(sources,dimensions,source_workers,
                                                                    max_candidates,max_csfs-labels.size());
            std::unique_ptr<SelectionWorkers> pool;
            if (!source_parallel) pool = std::make_unique<SelectionWorkers>(nthreads);
            std::vector<Stats> source_stats(source_workers);
            auto process_source = [&](size_t source_index, size_t source_worker) {
                auto &stats = source_stats[source_worker];
                ++stats["source_tasks"];
                Timer source_timer(stats["source_busy_ns"]);
                const auto &source = *ordered[source_index];
                std::vector<size_t> parents;
                double maximum = 0.0;
                for (size_t i : source.second) {
                    if (coeffs[i] != 0.0) parents.push_back(i);
                    maximum = std::max(maximum, std::abs(coeffs[i]));
                }
                if (parents.empty()) return;
                std::stable_sort(parents.begin(), parents.end(), [&](size_t a, size_t b) {
                    return std::abs(coeffs[a]) > std::abs(coeffs[b]);
                });
                std::vector<Path> parent_paths;
                size_t parent_budget = max_tensor_entries/source_workers;
                std::once_flag parent_once;
                std::vector<Transfer> patterns;
                size_t logical_pairs = 0;
                {
                    Timer timer(stats["connectivity_ns"]);
                    if (ordered_ready) {
                        logical_pairs = transfer_count(source.first,max_configurations);
                        patterns.push_back({{}, {}, 0});
                        ++stats["transfer_visits"];
                        for (int q = 0; q < k; ++q) {
                            if (!source.first[q]) continue;
                            for (int p = 0; p < k; ++p) if (p != q && source.first[p] < 2) {
                                patterns.push_back({{q,0},{p,0},1});
                                ++stats["transfer_visits"];
                            }
                            for (int s = q; s < k; ++s) {
                                if (source.first[s] < (q == s ? 2 : 1)) continue;
                                auto range = cache->ordered_transfers(q,s,k);
                                for (auto item = range.first; item != range.second; ++item) {
                                    ++stats["transfer_visits"];
                                    if (screen_bound(item->bound,maximum,eps,k)) break;
                                    if (source.first[item->p] == 2 ||
                                        source.first[item->r] >= (item->p == item->r ? 1 : 2)) continue;
                                    patterns.push_back({{q,s},{item->p,item->r},2});
                                }
                            }
                        }
                    } else {
                        transfers(source.first, max_configurations,
                              [&](const auto &holes, const auto &particles, int order) {
                        patterns.push_back({holes, particles, order});
                        });
                        logical_pairs = patterns.size();
                        stats["transfer_visits"] += logical_pairs;
                    }
                }
                stats["configuration_pairs"] += logical_pairs;
                std::vector<Config> endpoints;
                size_t rejected = logical_pairs-patterns.size(), complete = 0, unreachable = 0, reused_bounds = 0;
                stats["ordered_skipped_pairs"] += rejected;
                {
                    // Time a whole batch, not every scalar bound. Per-transfer
                    // clock/map bookkeeping was material in the cc-pVDZ profile.
                    Timer timer(stats["prescreen_ns"]);
                    for (const auto &pattern : patterns) {
                        if ((screening && prescreen) || selection_rule == 3) {
                            bool reused = false;
                            double upper = cache ? cache->selection_bound(source.first,pattern.holes,
                                                                          pattern.particles,pattern.order,k,reused) :
                                transfer_bound(source.first,pattern.holes,pattern.particles,pattern.order,h,v,k);
                            reused_bounds += reused;
                            if (screen_bound(upper, maximum, eps, k)) { ++rejected; continue; }
                        }
                        Config target = endpoint(source.first, pattern.holes, pattern.particles, pattern.order);
                        size_t dimension = dimensions[std::count(target.begin(), target.end(), 1)];
                        if (!dimension) { ++unreachable; continue; }
                        auto found = covered.find(target);
                        if ((found != covered.end() && found->second == dimension) ||
                            (source_parallel && shared_selection->complete(target))) { ++complete; continue; }
                        if (selection_rule == 3) { promote(target); continue; }
                        endpoints.push_back(std::move(target));
                    }
                }
                stats["prescreened_pairs"] += rejected;
                stats["bound_table_pairs"] += reused_bounds;
                stats["screened_pairs"] += rejected;
                stats["complete_pairs"] += complete;
                stats["unreachable_pairs"] += unreachable;
                { Timer timer(stats["connectivity_ns"]);
                  std::sort(endpoints.begin(), endpoints.end()); }
                // Keep only surviving keys/bounds. Rejected transfers never
                // allocate a target key or enter the persistent term cache.
                std::vector<std::pair<const Config *, double>> tasks;
                for (const Config &target : endpoints) {
                    ++stats["prepared_pairs"];
                    std::shared_ptr<const Connection> terms;
                    { Timer timer(stats["preparation_ns"]);
                      terms = prepare(target, source.first, h, v, k, max_connection_terms, cache, guga); }
                    stats["operator_terms"] += terms->term_count();
                    stats["operator_actions"] += terms->action_count();
                    if ((screening || selection_rule == 2) && screen_bound(target == source.first ? terms->term_bound : terms->bound,
                                                  maximum, eps, k)) {
                        ++stats["screened_pairs"];
                        continue;
                    }
                    if (selection_rule == 2) { promote(target); continue; }
                    tasks.emplace_back(&target, target == source.first ? terms->term_bound : terms->bound);
                }
                std::stable_sort(tasks.begin(), tasks.end(), [](const auto &a, const auto &b) {
                    return a.second > b.second;
                });
                size_t workers = source_parallel ? 1 : std::min(nthreads, std::max(size_t(1), tasks.size()));
                std::vector<Stats> worker_stats(workers);
                std::mutex selected_mutex;
                auto process_target = [&](size_t task, size_t worker) {
                    auto &local = source_parallel ? stats : worker_stats[worker];
                    const auto &target = *tasks[task].first;
                    std::shared_ptr<const Connection> terms;
                    { Timer timer(local["preparation_ns"]);
                      terms = prepare(target, source.first, h, v, k, max_connection_terms, cache, guga); }
                    if (walk_engine || selection_rule || source_parallel) {
                        // The reference visits every missing label of a surviving
                        // configuration on its first visit. Preserve that logical
                        // candidate guard even when structural zeros are skipped.
                        std::lock_guard<std::mutex> lock(selected_mutex);
                        account(target);
                    }
                    auto ensure_parents = [&] {
                        std::call_once(parent_once, [&]() {
                            Timer timer(local["tensor_ns"]);
                            for (size_t i : parents)
                                parent_paths.emplace_back(labels[i], k, two_s, parent_budget, !guga, guga_mode);
                        });
                    };
                    bool accepted = false;
                    auto target_state = source_parallel ? shared_selection->target(target) : nullptr;
                    auto evaluate = [&](const std::string &label) {
                        if (source_parallel) {
                            if (internal.count(label) || shared_selection->contains(target_state,label)) return;
                            ++local["generated_candidates"];
                        } else {
                            std::lock_guard<std::mutex> lock(selected_mutex);
                            if (internal.count(label) || selected.count(label)) return;
                            if (!walk_engine && !selection_rule) {
                                candidates.insert(label);
                                if (candidates.size() > max_candidates)
                                    throw BudgetError("HCI search exceeds max_candidates");
                            }
                            ++local["generated_candidates"];
                        }
                        ensure_parents();
                        // Parent tensors are shared read-only. Divide the remaining
                        // tensor budget between active workers rather than multiply it.
                        size_t budget = parent_budget/workers;
                        Path bra = [&]() {
                            Timer timer(local["tensor_ns"]);
                            return Path(label, k, two_s, budget, !guga, guga_mode);
                        }();
                        for (size_t j = 0; j < parents.size(); ++j) {
                            check_bonds(bra, parent_paths[j], max_bond_elements);
                            double hij;
                            { Timer timer(local["contraction_ns"]);
                              hij = element(bra, parent_paths[j], two_s, *terms, compiled); }
                            ++local["matrix_elements"];
                            local["contractions"] += terms->action_count();
                            if (std::abs(hij*coeffs[parents[j]]) > eps) {
                                if (source_parallel) {
                                    if (selection_rule == 1) {
                                        shared_selection->promote(target_state);
                                        accepted = true;
                                    } else shared_selection->insert(target_state,label);
                                    break;
                                }
                                std::lock_guard<std::mutex> lock(selected_mutex);
                                if (selection_rule == 1) {
                                    promote(target);
                                    accepted = true;
                                    break;
                                }
                                bool inserted = selected.insert(label).second;
                                if (selected.size() > max_csfs-labels.size())
                                    throw BudgetError("selected basis exceeds max_csfs");
                                if (inserted) ++covered[target];
                                break;
                            }
                        }
                    };
                    if (selection_rule == 1) {
                        paths_until(target,two_s,[&](const std::string &label) {
                            evaluate(label);
                            return !accepted && !(source_parallel && shared_selection->complete(target));
                        });
                        return;
                    }
                    size_t dimension = dimensions[std::count(target.begin(),target.end(),1)];
                    bool use_walks = walk_engine == 1 || (walk_engine == 2 && dimension >= 16 &&
                        dimension/2/std::max(size_t(1),terms->action_count()) >= parents.size());
                    bool complete = false;
                    if (use_walks) {
                        ensure_parents();
                        if (k && max_bond_elements < 2)
                            throw std::length_error("GUGA segment scratch exceeds max_bond_elements");
                        size_t tensor_before = local["tensor_ns"], contraction_before = local["contraction_ns"];
                        { Timer timer(local["walk_ns"]);
                          complete = guga_connected_paths(target,parent_paths,*terms,max_walk_bytes/workers,
                                                           evaluate,local); }
                        local["walk_ns"] -= local["tensor_ns"]-tensor_before+
                                            local["contraction_ns"]-contraction_before;
                    } else if (walk_engine == 2) ++local["walk_adaptive_skipped"];
                    else if (walk_engine == 3) ++local["walk_counted_pairs"];
                    if (!complete) {
                        if (source_parallel) paths_until(target,two_s,[&](const std::string &label) {
                            evaluate(label);
                            return !shared_selection->complete(target);
                        });
                        else paths(target, two_s, evaluate);
                    }
                };
                if (source_parallel) for (size_t task = 0; task < tasks.size(); ++task) process_target(task,0);
                else pool->apply(tasks.size(),workers,process_target);
                if (!source_parallel) for (const auto &local : worker_stats)
                    for (const auto &item : local) stats[item.first] += item.second;
                stats["workers"] = std::max(stats["workers"], workers);
            };
            if (source_parallel) parallel_for(ordered.size(),source_workers,process_source);
            else for (size_t i = 0; i < ordered.size(); ++i) process_source(i,0);
            for (const auto &local : source_stats) for (const auto &item : local) {
                if (item.first == "workers") stats[item.first] = std::max(stats[item.first],item.second);
                else stats[item.first] += item.second;
            }
            stats["source_parallel"] = source_parallel;
            if (source_parallel) {
                shared_selection->publish(selected,promoted);
                logical_candidates = shared_selection->candidates();
                promoted_additions = shared_selection->selected();
                stats["selection_shards"] = shared_selection->shard_count();
                stats["workers"] = source_workers;
            }
        } catch (const std::length_error &error) { throw BudgetError(error.what()); }
    }
    if (selection_rule) {
        // Materialize each admitted occupation once, after immutable-parent
        // searching. The final label order is independent of worker scheduling.
        py::gil_scoped_release release;
        std::set<std::string> internal(labels.begin(),labels.end());
        for (const auto &target : promoted)
            paths(target,two_s,[&](const std::string &label) {
                if (!internal.count(label)) selected.insert(label);
            });
        stats["promoted_configurations"] = promoted.size();
        stats["promoted_csfs"] = promoted_additions;
    }
    stats["candidates"] = walk_engine || selection_rule || stats["source_parallel"] ? logical_candidates : candidates.size();
    stats["added"] = selected.size();
    py::dict result = py::cast(stats);
    if (parent_fraction < 1.0) {
        result["parent_fraction"] = parent_fraction;
        result["retained_parent_csfs"] = stats["retained_parent_csfs"];
        result["retained_parent_weight"] = total_weight ? retained_weight/total_weight : 1.0;
    }
    std::vector<std::string> additions(selected.begin(),selected.end());
    if (return_groups) {
        Groups appended;
        {
            py::gil_scoped_release release;
            for (size_t i = 0; i < additions.size(); ++i)
                appended[configuration(additions[i],two_s)].push_back(labels.size()+i);
        }
        // Preserve first-appearance order of occupation keys as add_csf does.
        std::vector<const Groups::value_type *> order;
        for (const auto &group : appended) order.push_back(&group);
        std::sort(order.begin(),order.end(),[](const auto *a,const auto *b) {
            return a->second.front() < b->second.front();
        });
        py::list metadata;
        for (const auto *group : order)
            metadata.append(py::make_tuple(py::tuple(py::cast(group->first)),group->second));
        return py::make_tuple(std::move(additions),result,metadata);
    }
    return py::make_tuple(std::move(additions),result);
}

using ComplexArray = py::array_t<std::complex<double>, py::array::c_style>;

int validate_one_body(const ComplexArray &one) {
    if (one.ndim()!=2 || one.shape(0)!=one.shape(1))
        throw std::invalid_argument("one-electron operator must be square");
    for (py::ssize_t i=0;i<one.size();++i)
        if (!std::isfinite(one.data()[i].real()) || !std::isfinite(one.data()[i].imag()))
            throw std::invalid_argument("one-electron operator must be finite");
    return static_cast<int>(one.shape(0));
}

// Spin-scalar O=sum[pq] o[p,q] E[p,q]. No Hermiticity assumption,
// determinant expansion, four-index tensor, or stored operator matrix.
ComplexArray one_electron_product(const ComplexArray &one,
        const std::vector<std::string> &labels, int two_s, const ComplexArray &x,
        size_t nthreads, size_t budget) {
    int k=validate_one_body(one);
    validate_workers(nthreads);
    if (x.ndim()!=2 || static_cast<size_t>(x.shape(0))>labels.size())
        throw std::invalid_argument("one-electron coefficients need a selected-basis prefix block");
    for (py::ssize_t i=0;i<x.size();++i)
        if (!std::isfinite(x.data()[i].real()) || !std::isfinite(x.data()[i].imag()))
            throw std::invalid_argument("one-electron coefficients must be finite");
    size_t columns=x.shape(1), source_size=x.shape(0);
    ComplexArray output({static_cast<py::ssize_t>(labels.size()),x.shape(1)});
    auto *y=output.mutable_data();
    std::fill(y,y+output.size(),std::complex<double>{});
    {
        py::gil_scoped_release release;
        auto selected=groups(labels,k,two_s,0,labels.size());
        std::vector<Path> walks;
        for (const auto &label:labels) walks.emplace_back(label,k,two_s,budget,false,2);
        struct Task { const Groups::value_type *group; size_t begin,end; };
        std::vector<Task> tasks;
        for (const auto &a:selected) for (size_t i=0;i<a.second.size();i+=32)
            tasks.push_back({&a,i,std::min(i+32,a.second.size())});
        parallel_for(tasks.size(),nthreads,[&](size_t task,size_t) {
            const auto &tile=tasks[task];const auto &a=*tile.group;
            std::complex<double> diagonal{};
            for (int p=0;p<k;++p) diagonal+=double(a.first[p])*one.data()[p*k+p];
            for (size_t r=tile.begin;r<tile.end;++r) {
                size_t i=a.second[r];
                if (i<source_size) for (size_t c=0;c<columns;++c)
                    y[i*columns+c]=diagonal*x.data()[i*columns+c];
            }
            Config ket=a.first;
            for (int p=0;p<k;++p) {
                if (!a.first[p]) continue;
                --ket[p];
                for (int q=0;q<k;++q) {
                    auto integral=one.data()[p*k+q];
                    if (p==q || ket[q]==2 || integral==std::complex<double>{}) continue;
                    ++ket[q];auto found=selected.find(ket);--ket[q];
                    if (found==selected.end()) continue;
                    for (size_t r=tile.begin;r<tile.end;++r) {
                        size_t i=a.second[r];
                        for (size_t j:found->second) {
                            if (j>=source_size) continue;
                            auto value=integral*guga_generator(walks[i],walks[j],p,q);
                            for (size_t c=0;c<columns;++c) y[i*columns+c]+=value*x.data()[j*columns+c];
                        }
                    }
                }
                ++ket[p];
            }
        });
    }
    return output;
}

py::tuple select_one_electron(const ComplexArray &one,
        const std::vector<std::string> &labels,int two_s,const Array &weights,
        double eps,size_t nthreads,size_t budget,size_t max_new) {
    int k=validate_one_body(one);validate_workers(nthreads);
    if (weights.ndim()!=1 || static_cast<size_t>(weights.size())!=labels.size() ||
            !std::isfinite(eps) || eps<0)
        throw std::invalid_argument("invalid one-electron selection coefficients/epsilon");
    for (py::ssize_t i=0;i<weights.size();++i)
        if (!std::isfinite(weights.data()[i]) || weights.data()[i]<0)
            throw std::invalid_argument("one-electron selection weights must be finite and nonnegative");
    std::set<std::string> accepted;
    Stats stats{{"matrix_elements",0},{"configuration_pairs",0}};
    {
        py::gil_scoped_release release;
        auto selected=groups(labels,k,two_s,0,labels.size());
        std::unordered_set<std::string> internal(labels.begin(),labels.end());
        std::vector<Path> walks;
        for (const auto &label:labels) walks.emplace_back(label,k,two_s,budget,false,2);
        std::vector<const Groups::value_type *> jobs;
        for (const auto &a:selected) jobs.push_back(&a);
        size_t workers=std::min(nthreads,std::max(size_t(1),jobs.size()));
        std::vector<std::set<std::string>> additions(workers);
        std::vector<Stats> counters(workers);
        parallel_for(jobs.size(),workers,[&](size_t job,size_t worker) {
            const auto &source=*jobs[job];
            bool active=false;
            for (size_t j:source.second) active|=weights.data()[j]!=0.;
            if (!active) return;
            std::vector<Path> graph_parents;
            size_t graph_budget=budget/workers;
            Config target=source.first;
            for (int q=0;q<k;++q) {
                if (!source.first[q]) continue;
                --target[q];
                for (int p=0;p<k;++p) {
                    auto integral=one.data()[p*k+q];
                    if (p==q || target[p]==2 || integral==std::complex<double>{}) continue;
                    ++target[p];++counters[worker]["configuration_pairs"];
                    auto evaluate=[&](const std::string &label) {
                        if (internal.count(label) || additions[worker].count(label)) return;
                        size_t scratch=graph_budget;
                        Path bra(label,k,two_s,scratch,false,2);
                        for (size_t j:source.second) {
                            if (!weights.data()[j]) continue;
                            ++counters[worker]["matrix_elements"];
                            if (std::abs(integral*guga_generator(bra,walks[j],p,q))*weights.data()[j]>eps) {
                                additions[worker].insert(label);break;
                            }
                        }
                    };
                    bool connected=false;
                    if (std::count(target.begin(),target.end(),1)>=12) {
                        if (graph_parents.empty()) {
                            for (size_t j:source.second) if (weights.data()[j])
                                graph_parents.emplace_back(labels[j],k,two_s,graph_budget,false,2);
                        }
                        Connection channel;channel.guga=true;
                        channel.generators.push_back({1.,p,q,0,0,false});
                        connected=guga_connected_paths(target,graph_parents,channel,
                            (4*1024*1024)/workers,evaluate,counters[worker]);
                    }
                    if (!connected) paths(target,two_s,evaluate);
                    --target[p];
                }
                ++target[q];
            }
        });
        for (const auto &local:additions) for (const auto &label:local) {
            accepted.insert(label);
            if (accepted.size()>max_new) throw BudgetError("selected basis exceeds max_csfs");
        }
        for (const auto &local:counters) for (const auto &item:local) stats[item.first]+=item.second;
        stats["workers"]=workers;stats["added"]=accepted.size();
    }
    std::vector<std::string> additions(accepted.begin(),accepted.end());
    Groups appended;
    for (size_t i=0;i<additions.size();++i)
        appended[configuration(additions[i],two_s)].push_back(labels.size()+i);
    std::vector<const Groups::value_type *> order;
    for (const auto &group:appended) order.push_back(&group);
    std::sort(order.begin(),order.end(),[](auto a,auto b){return a->second.front()<b->second.front();});
    py::list metadata;
    for (const auto *group:order)
        metadata.append(py::make_tuple(py::tuple(py::cast(group->first)),group->second));
    return py::make_tuple(additions,py::cast(stats),metadata);
}

py::tuple build_csr(const Array &one, const Array &two,
                    const std::vector<std::string> &labels, int two_s,
                    size_t nrow, size_t ncol, bool symmetric, size_t start,
                    size_t max_entries, size_t max_bond_elements,
                    size_t max_tensor_entries, size_t max_connection_terms,
                    std::shared_ptr<ConnectionCache> cache, size_t nthreads, bool compiled, bool guga,
                    int matrix_index, size_t max_index_bytes, bool collect_reuse, size_t max_reuse_patterns, int guga_mode,
                    bool coupling_reuse, size_t max_reuse_bytes, bool packed_reuse, bool parallel_shared,
                    bool parallel_private, double matrix_cutoff, bool fast_coupling, size_t reuse_per_worker) {
    int k = validate_integrals(one, two);
    validate_workers(nthreads);
    if (!std::isfinite(matrix_cutoff) || matrix_cutoff < 0.)
        throw std::invalid_argument("matrix_cutoff must be finite and nonnegative");
    if (parallel_private) parallel_shared = true;
    if (guga_mode < 0 || guga_mode > 2) throw std::invalid_argument("invalid guga_mode");
    if (matrix_index < 0 || matrix_index > 3) throw std::invalid_argument("invalid matrix_index");
    if (fast_coupling && !guga) throw std::invalid_argument("ranked CSR coupling requires GUGA");
    bool fast = fast_coupling && k <= 64;
    if (cache) cache->validate(one, two);
    if (nrow > labels.size() || ncol > labels.size() || start > nrow ||
        (symmetric && nrow != ncol) || (!symmetric && start) || two_s < 0)
        throw std::invalid_argument("invalid native CSR dimensions/storage mode");
    const double *h = one.data(), *v = two.data();
    using Rows = std::vector<std::vector<std::pair<std::int64_t, double>>>;
    Rows rows(parallel_private ? 0 : nrow-start);
    std::vector<double> data;
    std::vector<std::int64_t> indices, indptr{0};
    py::array_t<double> shared_data;
    py::array_t<std::int64_t> shared_indices, shared_indptr;
    Stats stats{{"configuration_pairs", 0}, {"configuration_checks", 0},
                {"configuration_lookups", 0}, {"matrix_elements", 0}, {"operator_terms", 0},
                {"rows", nrow-start}, {"entries", 0}, {"fast_coupling",fast},
                {"fast_coupling_fallback",fast_coupling && !fast}};
    std::vector<ReuseHistogram> reuse;
    std::vector<std::unique_ptr<GugaMemo>> memo;
    {
        py::gil_scoped_release release;
        try {
            Groups left, right;
            { Timer timer(stats["index_ns"]);
              left = groups(labels, k, two_s, start, nrow);
              right = groups(labels, k, two_s, 0, ncol); }
            // For <=32 orbitals this is a complete, collision-free occupation
            // key, not a fingerprint. Larger bases retain complete-vector lookup.
            std::unordered_map<std::uint64_t, const Groups::value_type *> packed_right;
            if (k <= 32) {
                Timer timer(stats["index_ns"]);
                packed_right.reserve(right.size());
                for (const auto &b : right) packed_right.emplace(occupation_key(b.first), &b);
            }
            std::unique_ptr<SelectedIndex> selected_index;
            {
                Timer timer(stats["index_ns"]);
                selected_index = std::make_unique<SelectedIndex>(right,matrix_index,max_index_bytes);
                stats["matrix_index"] = selected_index->mode();
                stats["matrix_index_bytes"] = selected_index->bytes();
                stats["matrix_index_entries"] = selected_index->entries();
                stats["matrix_index_fallback"] = matrix_index && !selected_index->mode();
            }
            std::atomic<size_t> entries{0};
            struct MatrixTask { const Groups::value_type *group; size_t begin, end; };
            std::vector<MatrixTask> tasks;
            if (!parallel_private) for (const auto &a : left) {
                size_t tile = a.second.size();
                if (parallel_shared && nthreads > 1 && tile >= 64)
                    tile = std::max(size_t(1),(tile+4*nthreads-1)/(4*nthreads));
                for (size_t begin = 0; begin < a.second.size(); begin += tile)
                    tasks.push_back({&a,begin,std::min(a.second.size(),begin+tile)});
            }
            stats["parallel_shared"] = parallel_shared;
            stats["parallel_private"] = parallel_private;
            size_t workers = std::min(nthreads, std::max(size_t(1),
                parallel_private ? nrow-start : tasks.size()));
            // Contiguous row ownership with private CSR blocks.
            // An occupation may cross owners; each owner evaluates only its rows.
            struct alignas(64) PrivateBlock {
                size_t begin=0, end=0, entries=0;
                Rows rows;
                std::vector<MatrixTask> tasks;
                std::vector<double> values;
                std::vector<std::int64_t> columns, offsets{0};
            };
            std::vector<PrivateBlock> blocks(parallel_private ? workers : 0);
            if (parallel_private) for (size_t w=0; w<workers; ++w) {
                auto &block=blocks[w];
                size_t count=nrow-start, chunk=count/workers, extra=count%workers;
                block.begin=start+w*chunk+std::min(w,extra);
                block.end=block.begin+chunk+(w<extra);
                block.rows.resize(block.end-block.begin);
                for (const auto &a:left) {
                    auto begin=std::lower_bound(a.second.begin(),a.second.end(),block.begin);
                    auto end=std::lower_bound(begin,a.second.end(),block.end);
                    if (begin!=end) block.tasks.push_back({&a,size_t(begin-a.second.begin()),
                                                           size_t(end-a.second.begin())});
                }
                stats["row_tiles"]+=block.tasks.size();
            }
            else stats["row_tiles"]=tasks.size();
            std::vector<Stats> worker_stats(workers);
            if (collect_reuse) reuse.resize(workers);
            if (guga && (collect_reuse || coupling_reuse || fast))
                for (size_t w = 0; w < workers; ++w)
                    memo.push_back(std::make_unique<GugaMemo>(std::min(max_reuse_bytes/workers,reuse_per_worker),
                        coupling_reuse,packed_reuse,fast ? GugaMemo::Tiny : GugaMemo::Generic));
            std::vector<Connection> connections(workers);
            std::vector<GugaLeanPlan> plans(workers);
            auto down_mask = [&](const Path &p) {
                std::uint64_t result=0;
                if (fast) for (int i=0;i<k;++i) if (p.label[i]=='d') result|=std::uint64_t(1)<<i;
                return result;
            };
            auto process_tile = [&](const MatrixTask &tile, size_t worker) {
                auto &stats = worker_stats[worker];
                const auto &a = *tile.group;
                size_t last_row = a.second[tile.end-1];
                size_t pending_entries = 0;
                auto flush_entries = [&] {
                    if (parallel_private && max_entries==std::numeric_limits<size_t>::max()) {
                        blocks[worker].entries+=pending_entries;
                        pending_entries=0;
                        return;
                    }
                    size_t previous = entries.load(std::memory_order_relaxed);
                    do {
                        if (pending_entries > max_entries-previous)
                            throw BudgetError("CSR matrix exceeds max_matrix_entries");
                    } while (!entries.compare_exchange_weak(previous,previous+pending_entries,
                                                            std::memory_order_relaxed));
                    pending_entries = 0;
                };
                size_t row_budget = max_tensor_entries/workers;
                std::vector<Path> bra;
                { Timer timer(stats["tensor_ns"]);
                  for (size_t r = tile.begin; r < tile.end; ++r)
                      bra.emplace_back(labels[a.second[r]], k, two_s, row_budget, !guga, guga_mode); }
                std::vector<std::uint64_t> bra_down;
                if (fast) for (const auto &p:bra) bra_down.push_back(down_mask(p));
                // Prefer endpoint lookup when its transfer-pattern upper bound
                // is cheaper than scanning every selected configuration. Cap
                // endpoint traversal; the scan remains a safe fallback.
                size_t occupied = 0, doubles = 0;
                for (int n : a.first) { occupied += n != 0; doubles += n == 2; }
                long double removal_pairs = doubles+static_cast<long double>(occupied)*
                                            (occupied ? occupied-1 : 0)/2;
                long double upper = 1.0L+static_cast<long double>(occupied)*k+
                                    removal_pairs*k*(k+1.0L)/2;
                std::vector<const Groups::value_type *> partners;
                {
                Timer timer(stats["connectivity_ns"]);
                if (selected_index->mode()) {
                    partners = selected_index->query(a.first,symmetric ? last_row :
                                                    std::numeric_limits<size_t>::max(),stats);
                } else if (upper < right.size() && upper <= 100000) {
                    Config target = a.first;
                    std::uint64_t source_key = k <= 32 ? occupation_key(a.first) : 0;
                    transfers(a.first, static_cast<size_t>(upper), [&](const auto &holes,
                                                                      const auto &particles, int order) {
                        ++stats["configuration_lookups"];
                        if (k <= 32) {
                            auto key = source_key;
                            for (int i = 0; i < order; ++i) {
                                key -= std::uint64_t(1) << (2*holes[i]);
                                key += std::uint64_t(1) << (2*particles[i]);
                            }
                            auto found = packed_right.find(key);
                            if (found != packed_right.end()) partners.push_back(found->second);
                        } else {
                            for (int i = 0; i < order; ++i) { --target[holes[i]]; ++target[particles[i]]; }
                            auto found = right.find(target);
                            if (found != right.end()) partners.push_back(&*found);
                            for (int i = 0; i < order; ++i) { ++target[holes[i]]; --target[particles[i]]; }
                        }
                    });
                } else {
                    for (const auto &b : right) partners.push_back(&b);
                }
                }
                for (const auto *partner : partners) {
                    const auto &b = *partner;
                    ++stats["configuration_checks"];
                    if (symmetric && b.second.front() > last_row) continue;
                    int distance = 0;
                    for (int p = 0; p < k; ++p) distance += std::abs(a.first[p]-b.first[p]);
                    if (distance > 4) continue;
                    ++stats["configuration_pairs"];
                    std::shared_ptr<const Connection> owner;
                    const Connection *terms;
                    { Timer timer(stats["preparation_ns"]);
                      if (fast) {
                          auto &c=connections[worker];
                          ranked_connection(PairDescriptor(a.first,b.first),h,v,max_connection_terms,false,c);
                          prepare_guga_lean_connection(a.first,b.first,c,plans[worker],true);
                          terms=&c;
                      } else {
                          owner=prepare(a.first,b.first,h,v,k,max_connection_terms,cache,guga);
                          terms=owner.get();
                      } }
                    stats["operator_terms"] += terms->term_count();
                    stats["operator_actions"] += terms->action_count();
                    size_t before_elements = stats["matrix_elements"];
                    size_t budget = row_budget;
                    std::vector<size_t> columns;
                    std::vector<Path> ket;
                    { Timer timer(stats["tensor_ns"]);
                    for (size_t j : b.second) {
                        if (symmetric && j > last_row) continue;
                        columns.push_back(j);
                        ket.emplace_back(labels[j], k, two_s, budget, !guga, guga_mode);
                    }
                    }
                    std::vector<std::uint64_t> ket_down;
                    if (fast) for (const auto &p:ket) ket_down.push_back(down_mask(p));
                    for (size_t r = 0; r < bra.size(); ++r) {
                        size_t i = a.second[tile.begin+r];
                        for (size_t c = 0; c < columns.size(); ++c) {
                            size_t j = columns[c];
                            if (symmetric && j > i) continue;
                            check_bonds(bra[r], ket[c], max_bond_elements);
                            double value;
                            { Timer timer(stats["contraction_ns"]);
                              value = fast ? memo[worker]->small_element(bra[r],ket[c],*terms,plans[worker],bra_down[r],ket_down[c],true) :
                                  memo.empty() ? element(bra[r],ket[c],two_s,*terms,compiled) :
                                                 memo[worker]->element(bra[r],ket[c],*terms); }
                            ++stats["matrix_elements"];
                            stats["contractions"] += terms->action_count();
                            if (std::abs(value) > matrix_cutoff || i == j) {
                                if (parallel_shared) {
                                    ++pending_entries;
                                    if (pending_entries >= 64 || pending_entries > max_entries) flush_entries();
                                } else if (entries.fetch_add(1) >= max_entries)
                                    throw BudgetError("CSR matrix exceeds max_matrix_entries");
                                auto &row=parallel_private ? blocks[worker].rows[i-blocks[worker].begin] :
                                                             rows[i-start];
                                row.emplace_back(static_cast<std::int64_t>(j), value);
                            }
                        }
                    }
                    if (collect_reuse) {
                        Timer timer(stats["reuse_profile_ns"]);
                        auto key = reduced_pattern(a.first,b.first);
                        auto &hist = reuse[worker];
                        auto found = hist.find(key);
                        if (found == hist.end() && hist.size() < max_reuse_patterns/workers)
                            found = hist.emplace(std::move(key),ReuseCount{}).first;
                        size_t elements = stats["matrix_elements"]-before_elements;
                        if (found == hist.end()) {
                            ++stats["reuse_overflow_pairs"];
                            stats["reuse_overflow_elements"] += elements;
                        } else {
                            ++found->second.pairs;
                            found->second.elements += elements;
                            found->second.actions += elements*terms->action_count();
                        }
                    }
                }
                if (parallel_shared) flush_entries();
            };
            if (parallel_private) {
                parallel_for(workers,workers,[&](size_t w,size_t) {
                    auto &block=blocks[w];
                    for (const auto &tile:block.tasks) process_tile(tile,w);
                    // Sort and pack privately before publishing. Old/new CSR
                    // storage remains untouched if a worker or guard fails.
                    size_t count=0;
                    for (const auto &row:block.rows) count+=row.size();
                    block.values.reserve(count); block.columns.reserve(count);
                    block.offsets.reserve(block.rows.size()+1);
                    for (auto &row:block.rows) {
                        std::sort(row.begin(),row.end());
                        for (const auto &entry:row) {
                            block.columns.push_back(entry.first);
                            block.values.push_back(entry.second);
                        }
                        block.offsets.push_back(block.values.size());
                        std::vector<std::pair<std::int64_t,double>>().swap(row);
                    }
                    Rows().swap(block.rows);
                });
                if (max_entries==std::numeric_limits<size_t>::max()) {
                    size_t total=0;
                    for (const auto &block:blocks) {
                        if (block.entries>max_entries-total) throw BudgetError("CSR index capacity exceeded");
                        total+=block.entries;
                    }
                    entries.store(total);
                }
            } else parallel_for(tasks.size(),workers,[&](size_t task,size_t worker) {
                process_tile(tasks[task],worker);
            });
            for (const auto &local : worker_stats)
                for (const auto &item : local) stats[item.first] += item.second;
            for (const auto &cache : memo)
                for (const auto &item : cache->info()) stats[item.first] += item.second;
            stats["workers"] = workers;
            Timer timer(stats["assembly_ns"]);
            if (parallel_private) {
                indptr.reserve(nrow-start+1);
                for (const auto &block:blocks) {
                    size_t offset=indptr.back();
                    for (size_t i=1;i<block.offsets.size();++i)
                        indptr.push_back(offset+block.offsets[i]);
                }
                {
                    py::gil_scoped_acquire acquire;
                    shared_data=py::array_t<double>(entries.load());
                    shared_indices=py::array_t<std::int64_t>(entries.load());
                    shared_indptr=py::array_t<std::int64_t>(indptr.size());
                }
                std::copy(indptr.begin(),indptr.end(),shared_indptr.mutable_data());
                auto *values=shared_data.mutable_data();
                auto *columns=shared_indices.mutable_data();
                parallel_for(workers,workers,[&](size_t w,size_t) {
                    const auto &block=blocks[w];
                    size_t offset=indptr[block.begin-start];
                    std::copy(block.values.begin(),block.values.end(),values+offset);
                    std::copy(block.columns.begin(),block.columns.end(),columns+offset);
                });
            } else if (parallel_shared) {
                // Allocate uninitialized final NumPy CSR arrays once. Workers
                // sort and first-touch disjoint output row ranges; no serial
                // nnz-sized concatenation, extra CSR copy, or shared push_back.
                parallel_for(rows.size(),workers,[&](size_t r,size_t) {
                    std::sort(rows[r].begin(),rows[r].end());
                });
                indptr.reserve(rows.size()+1);
                for (const auto &row : rows)
                    indptr.push_back(indptr.back()+static_cast<std::int64_t>(row.size()));
                {
                    py::gil_scoped_acquire acquire;
                    shared_data = py::array_t<double>(entries.load());
                    shared_indices = py::array_t<std::int64_t>(entries.load());
                    shared_indptr = py::array_t<std::int64_t>(indptr.size());
                }
                double *values = shared_data.mutable_data();
                auto *columns = shared_indices.mutable_data();
                std::copy(indptr.begin(),indptr.end(),shared_indptr.mutable_data());
                parallel_for(rows.size(),workers,[&](size_t r,size_t) {
                    size_t at = indptr[r];
                    for (const auto &entry : rows[r]) {
                        columns[at] = entry.first;
                        values[at++] = entry.second;
                    }
                });
            } else {
                data.reserve(entries);
                indices.reserve(entries);
                for (auto &row : rows) {
                    std::sort(row.begin(), row.end());
                    for (const auto &entry : row) {
                        indices.push_back(entry.first);
                        data.push_back(entry.second);
                    }
                    indptr.push_back(static_cast<std::int64_t>(data.size()));
                }
            }
        } catch (const std::length_error &error) { throw BudgetError(error.what()); }
    }
    stats["entries"] = parallel_shared ? shared_data.size() : data.size();
    auto array = [](const auto &values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        py::array_t<T> result(values.size());
        std::copy(values.begin(), values.end(), result.mutable_data());
        return result;
    };
    py::dict output = py::cast(stats);
    if (collect_reuse) {
        ReuseHistogram combined;
        for (const auto &hist : reuse) for (const auto &item : hist) {
            auto &count = combined[item.first];
            count.pairs += item.second.pairs;
            count.elements += item.second.elements;
            count.actions += item.second.actions;
        }
        py::dict histogram;
        for (const auto &item : combined)
            histogram[py::str(item.first)] = py::cast(std::map<std::string,size_t>{
                {"pairs",item.second.pairs},{"elements",item.second.elements},{"actions",item.second.actions}});
        output["reuse_histogram"] = histogram;
    }
    if (!memo.empty() && collect_reuse) {
        std::map<size_t,size_t> frequencies;
        for (const auto &cache : memo) for (const auto &item : cache->frequencies()) frequencies[item.first] += item.second;
        output["generator_reuse_frequencies"] = py::cast(frequencies);
    }
    if (parallel_shared) return py::make_tuple(shared_data,shared_indices,shared_indptr,output);
    return py::make_tuple(array(data), array(indices), array(indptr), output);
}
#include "symbolic_tiles.h"
#include "direct_impl.h"
} // namespace

void bind_sci(py::module_ &m) {
    py::class_<DirectHamiltonian>(m,"_DirectHamiltonian")
        .def(py::init<const Array &,const Array &,BasisSnapshot,
             size_t,int,double,double,int,size_t,size_t,size_t,size_t,size_t,size_t,int,bool,size_t,bool,size_t,size_t,bool,size_t,size_t,size_t,bool,bool,size_t,size_t,size_t,bool,int,std::vector<int>,bool>())
        .def("apply",&DirectHamiltonian::apply,py::arg("coefficients"),py::arg("generator_tol")=0.,
             py::arg("generator_screen")=0,py::arg("unscreened")=false)
        .def("diagonal",&DirectHamiltonian::diagonal)
        .def("evaluate_pair_entry",&DirectHamiltonian::evaluate_pair_entry)
        .def_property_readonly("basis",&DirectHamiltonian::basis)
        .def_property_readonly("info",&DirectHamiltonian::info);
    m.def("one_electron_product",&one_electron_product,py::arg("one"),py::arg("labels"),
          py::arg("two_s"),py::arg("coefficients"),py::arg("nthreads")=1,
          py::arg("max_tensor_entries")=std::numeric_limits<size_t>::max());
    m.def("select_one_electron",&select_one_electron,py::arg("one"),py::arg("labels"),
          py::arg("two_s"),py::arg("weights"),py::arg("eps"),py::arg("nthreads")=1,
          py::arg("max_tensor_entries")=std::numeric_limits<size_t>::max(),
          py::arg("max_new")=std::numeric_limits<size_t>::max());
    py::class_<ConnectionCache, std::shared_ptr<ConnectionCache>>(m, "ConnectionCache")
        .def(py::init<const Array &, const Array &, size_t, bool, size_t>(),
             py::arg("one_mo"), py::arg("two_mo"), py::arg("maximum"), py::arg("guga") = false,
             py::arg("max_transfer_bounds") = 0)
        .def("clear", &ConnectionCache::clear)
        .def("info", &ConnectionCache::info)
        .def("search_info", &ConnectionCache::search_info)
        .def("transfer_info", &ConnectionCache::transfer_info);
    m.def("connected_configurations", [](const Config &source, size_t maximum) {
        std::set<Config> result;
        { py::gil_scoped_release release; result = connected(source, maximum); }
        return std::vector<Config>(result.begin(), result.end());
    }, py::arg("occupations"), py::arg("max_configurations") = 100000);
    m.def("configuration_bound", [](const Array &one, const Array &two, const Config &bra,
                                    const Config &ket, size_t maximum,
                                    std::shared_ptr<ConnectionCache> cache, bool guga) {
        int k = validate_integrals(one, two);
        if (cache) cache->validate(one, two);
        validate_config(bra); validate_config(ket);
        if (bra.size() != static_cast<size_t>(k) || ket.size() != bra.size() ||
            std::accumulate(bra.begin(), bra.end(), 0) !=
            std::accumulate(ket.begin(), ket.end(), 0))
            throw std::invalid_argument("configuration dimensions/electron numbers differ");
        try {
            py::gil_scoped_release release;
            return prepare(bra, ket, one.data(), two.data(), k, maximum, cache, guga)->bound;
        } catch (const std::length_error &error) { throw BudgetError(error.what()); }
    }, py::arg("one_mo"), py::arg("two_mo"), py::arg("bra"), py::arg("ket"),
       py::arg("max_connection_terms") = 1000000, py::arg("cache") = nullptr, py::arg("guga") = false);
    m.def("prescreen_bound", [](const Array &one, const Array &two, const Config &bra,
                                    const Config &ket) {
        int k = validate_integrals(one, two);
        validate_config(bra); validate_config(ket);
        if (bra.size() != static_cast<size_t>(k) || ket.size() != bra.size() ||
            std::accumulate(bra.begin(), bra.end(), 0) !=
            std::accumulate(ket.begin(), ket.end(), 0))
            throw std::invalid_argument("configuration dimensions/electron numbers differ");
        try {
            py::gil_scoped_release release;
            return prescreen_bound(bra, ket, one.data(), two.data(), k);
        } catch (const std::length_error &error) { throw BudgetError(error.what()); }
    }, py::arg("one_mo"), py::arg("two_mo"), py::arg("bra"), py::arg("ket"));
    m.def("select_hci", &select, py::arg("one_mo"), py::arg("two_mo"), py::arg("labels"),
          py::arg("two_s"), py::arg("coeffs"), py::arg("eps"), py::arg("screening") = true,
          py::arg("max_candidates") = 100000, py::arg("max_csfs") = 100000,
          py::arg("max_configurations") = 100000, py::arg("max_bond_elements") = 1000000,
          py::arg("max_tensor_entries") = 1000000, py::arg("max_connection_terms") = 1000000,
          py::arg("cache") = nullptr, py::arg("nthreads") = 1,
          py::arg("prescreen") = true, py::arg("compiled") = true, py::arg("guga") = false,
          py::arg("ordered_search") = false, py::arg("max_ordered_bytes") = 16777216, py::arg("guga_mode") = 0,
          py::arg("walk_engine") = 0, py::arg("max_walk_bytes") = 4194304,
          py::arg("selection_rule") = 0, py::arg("parent_fraction") = 1.0, py::arg("parallel_shared") = false,
          py::arg("return_groups") = false);
    m.def("build_csr", &build_csr, py::arg("one_mo"), py::arg("two_mo"), py::arg("labels"),
          py::arg("two_s"), py::arg("nrow"), py::arg("ncol"), py::arg("symmetric") = true,
          py::arg("start") = 0, py::arg("max_entries") = 2000000,
          py::arg("max_bond_elements") = 1000000, py::arg("max_tensor_entries") = 1000000,
          py::arg("max_connection_terms") = 1000000,
          py::arg("cache") = nullptr, py::arg("nthreads") = 1, py::arg("compiled") = true, py::arg("guga") = false,
          py::arg("matrix_index") = 0, py::arg("max_index_bytes") = 67108864,
          py::arg("collect_reuse") = false, py::arg("max_reuse_patterns") = 10000, py::arg("guga_mode") = 0,
          py::arg("coupling_reuse") = false, py::arg("max_reuse_bytes") = 16777216,
          py::arg("packed_reuse") = false, py::arg("parallel_shared") = false,
          py::arg("parallel_private") = false, py::arg("matrix_cutoff") = 1e-10,
          py::arg("fast_coupling") = false, py::arg("reuse_per_worker") = std::numeric_limits<size_t>::max());
}
} // namespace scsf

/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#pragma once
#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <stdexcept>
#include <string>
#ifdef __linux__
#include <sched.h>
#include <cstring>
#include <cerrno>
#endif

namespace scsf {
// A retained team for repeated coarse jobs. The caller participates. Every
// generation has a completion barrier, including exceptions; captured buffers
// cannot expire while any worker still uses the callback. The owning operator
// serializes calls.
class ThreadTeam {
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable ready_, finished_;
    std::atomic<bool> stop_{false};
    size_t generation_ = 0, completed_ = 0;
    bool shutdown_ = false;
    std::exception_ptr failure_;
    std::function<void(size_t)> callback_;
    size_t active_ = 0;
    std::vector<int> cpus_;
    // Only the participating caller must recover its original affinity.
    // Background threads keep their assigned CPU between generations.
    struct Placement {
#ifdef __linux__
        cpu_set_t previous{};
        bool restore=false;
#endif
        Placement(const std::vector<int> &cpus,size_t worker) {
            if (cpus.empty()) return;
#ifdef __linux__
            if (worker==0) {
                if (sched_getaffinity(0,sizeof(previous),&previous))
                    throw std::runtime_error("cannot read caller CPU affinity");
                restore=true;
            }
            cpu_set_t target;CPU_ZERO(&target);CPU_SET(cpus.at(worker),&target);
            if (sched_setaffinity(0,sizeof(target),&target))
                throw std::runtime_error(std::string("cannot pin worker: ")+std::strerror(errno));
#else
            throw std::invalid_argument("worker CPU placement requires Linux");
#endif
        }
        ~Placement() {
#ifdef __linux__
            if (restore) sched_setaffinity(0,sizeof(previous),&previous);
#endif
        }
    };
    void execute(size_t worker,bool &pinned) {
        if (worker>=active_ || stop_.load(std::memory_order_relaxed)) return;
        try {
            if (worker==0) {
                Placement placement(cpus_,worker);
                callback_(worker);
            } else {
                if (!pinned) {
                    Placement placement(cpus_,worker);
                    pinned=true;
                }
                callback_(worker);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!failure_) failure_ = std::current_exception();
            stop_.store(true,std::memory_order_relaxed);
        }
    }
    void run(size_t worker) {
        size_t seen = 0;
        bool pinned = cpus_.empty();
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_.wait(lock,[&] { return shutdown_ || generation_ != seen; });
            if (shutdown_) return;
            seen = generation_;
            lock.unlock();
            execute(worker,pinned);
            lock.lock();
            ++completed_;
            finished_.notify_one();
        }
    }
public:
    explicit ThreadTeam(size_t workers,std::vector<int> cpus={}) : cpus_(std::move(cpus)) {
        if (!workers || (!cpus_.empty() && cpus_.size()<workers))
            throw std::invalid_argument("invalid worker team size/CPU list");
#ifdef __linux__
        cpu_set_t allowed;
        if (!cpus_.empty()) {
            if (sched_getaffinity(0,sizeof(allowed),&allowed)) throw std::runtime_error("cannot read allowed CPUs");
            for (int cpu:cpus_) if (cpu<0 || cpu>=CPU_SETSIZE || !CPU_ISSET(cpu,&allowed))
                throw std::invalid_argument("worker CPU is outside the allowed CPU set");
        }
#else
        if (!cpus_.empty()) throw std::invalid_argument("worker CPU placement requires Linux");
#endif
        try {
            for (size_t i = 1; i < workers; ++i) threads_.emplace_back([this,i] { run(i); });
        } catch (...) {
            { std::lock_guard<std::mutex> lock(mutex_); shutdown_ = true; }
            ready_.notify_all();
            for (auto &thread : threads_) thread.join();
            throw;
        }
    }
    ~ThreadTeam() {
        { std::lock_guard<std::mutex> lock(mutex_); shutdown_ = true; }
        ready_.notify_all();
        for (auto &thread : threads_) thread.join();
    }
    // Each active index belongs to the same physical thread on every call.
    void apply_workers(size_t active,std::function<void(size_t)> job) {
        if (active>threads_.size()+1) throw std::invalid_argument("too many active workers");
        if (!active) return;
        if (threads_.empty()) {
            Placement placement(cpus_,0);
            job(0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            callback_ = std::move(job);
            active_ = active;
            completed_ = 0;
            stop_.store(false,std::memory_order_relaxed);
            failure_ = nullptr;
            ++generation_;
        }
        ready_.notify_all();
        bool pinned = false;
        execute(0,pinned);
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock,[&] { return completed_ == threads_.size(); });
        callback_ = {};
        if (failure_) std::rethrow_exception(failure_);
    }
};
} // namespace scsf

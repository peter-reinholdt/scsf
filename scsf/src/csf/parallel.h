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

namespace scsf {
// A retained team for repeated coarse jobs. The caller participates. Every
// generation has a completion barrier, including exceptions; captured buffers
// cannot expire while any worker still uses the callback. Calls are serialized
// by the owning operator. No nested team or MKL worker region is entered.
class ThreadTeam {
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable ready_, finished_;
    std::atomic<size_t> next_{0};
    std::atomic<bool> stop_{false};
    size_t generation_ = 0, completed_ = 0, count_ = 0;
    bool shutdown_ = false;
    std::exception_ptr failure_;
    std::function<void(size_t)> callback_;
    void execute() {
        try {
            while (!stop_.load(std::memory_order_relaxed)) {
                size_t task = next_.fetch_add(1,std::memory_order_relaxed);
                if (task >= count_) break;
                callback_(task);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!failure_) failure_ = std::current_exception();
            stop_.store(true,std::memory_order_relaxed);
        }
    }
    void run() {
        size_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_.wait(lock,[&] { return shutdown_ || generation_ != seen; });
            if (shutdown_) return;
            seen = generation_;
            lock.unlock();
            execute();
            lock.lock();
            ++completed_;
            finished_.notify_one();
        }
    }
public:
    explicit ThreadTeam(size_t workers) {
        try {
            for (size_t i = 1; i < workers; ++i) threads_.emplace_back([this] { run(); });
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
    void apply(size_t tasks, std::function<void(size_t)> job) {
        if (threads_.empty() || !tasks) {
            for (size_t i = 0; i < tasks; ++i) job(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            callback_ = std::move(job);
            count_ = tasks;
            completed_ = 0;
            next_.store(0,std::memory_order_relaxed);
            stop_.store(false,std::memory_order_relaxed);
            failure_ = nullptr;
            ++generation_;
        }
        ready_.notify_all();
        execute();
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock,[&] { return completed_ == threads_.size(); });
        callback_ = {};
        if (failure_) std::rethrow_exception(failure_);
    }
};
} // namespace scsf

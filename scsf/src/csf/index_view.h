/* This file is part of SCSF, licensed under GNU GPL version 3 or later. */
#pragma once
#include <cstddef>
#include <vector>

namespace scsf {
// Borrowed immutable local-to-global CSF indices. The owning basis outlives
// every view, index and worker; no public CI-vector reordering is required.
class IndexView {
    const size_t *data_ = nullptr;
    size_t size_ = 0;
public:
    IndexView() = default;
    IndexView(const size_t *data, size_t size) : data_(data), size_(size) {}
    IndexView(const std::vector<size_t> &indices) : IndexView(indices.data(), indices.size()) {}
    const size_t *begin() const { return data_; }
    const size_t *end() const { return size_ ? data_+size_ : data_; }
    size_t size() const { return size_; }
    bool empty() const { return !size_; }
    size_t operator[](size_t i) const { return data_[i]; }
    size_t front() const { return data_[0]; }
    size_t back() const { return data_[size_-1]; }
};
} // namespace scsf

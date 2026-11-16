/* This file is part of SCSF, licensed under GNU GPL version 3 or later. */
#pragma once
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <vector>

namespace scsf {
// Exact doubled-spin vertices. Optional checkpoints store the entering spin
// plus up/down masks for each sixteen-orbital segment. All masks fit positive
// int values; no word-sized orbital limit or narrow entering-spin limit exists.
class SpinVertices {
    std::vector<int> data_;
    bool compressed() const { return !data_.empty() && data_[0]<0; }
    static int count(unsigned value) {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_popcount(value);
#else
        int result=0;
        for (;value;value&=value-1) ++result;
        return result;
#endif
    }
public:
    bool is_compressed() const { return compressed(); }
    template<bool Checkpoint> int get(size_t vertex) const {
        if constexpr (Checkpoint) return (*this)[vertex];
        return data_[vertex];
    }
    bool empty() const { return size()==0; }
    size_t size() const { return compressed() ? size_t(-data_[0]) : data_.size(); }
    size_t capacity() const { return data_.capacity(); } // allocated int slots
    void reserve(size_t n) { data_.reserve(n); }
    void push_back(int value) { data_.push_back(value); }
    int operator[](size_t vertex) const {
        if (!compressed()) return data_[vertex];
        size_t base=1+3*(vertex/16);
        unsigned mask=(1u<<(vertex%16))-1;
        return data_[base]+count(unsigned(data_[base+1])&mask)-count(unsigned(data_[base+2])&mask);
    }
    int back() const { return (*this)[size()-1]; }
    void compress() {
        if (compressed() || data_.empty() || data_.size()>size_t(std::numeric_limits<int>::max())) return;
        size_t slots=1+3*((data_.size()-1)/16+1);
        if (slots>=data_.size()) return; // exact full-table fallback for tiny paths
        std::vector<int> packed(slots,0);
        // Full paths always begin at vacuum spin zero. A negative first slot
        // therefore identifies compression without enlarging each Path object.
        packed[0]=-int(data_.size());
        for (size_t i=0;i<data_.size();++i) {
            size_t base=1+3*(i/16);
            if (i%16==0) packed[base]=data_[i];
            if (i+1<data_.size()) {
                int change=data_[i+1]-data_[i];
                if (change>0) packed[base+1]|=1<<(i%16);
                else if (change<0) packed[base+2]|=1<<(i%16);
            }
        }
        data_.swap(packed);
    }
};
} // namespace scsf

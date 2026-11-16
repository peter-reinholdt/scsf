/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#include "csf.h"
#include "parallel.h"
#include <cstdint>
#include <pybind11/stl.h>
#include <algorithm>

namespace scsf {
namespace {
namespace py = pybind11;
using Index = std::int64_t;
using Dense = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Indices = py::array_t<Index, py::array::c_style | py::array::forcecast>;

Dense real_array(const py::object &value) {
    auto array = py::array::ensure(value);
    if (!array || array.dtype().kind() == 'c')
        throw py::type_error("Native CSR products require real arrays");
    return Dense(array);
}

class CSRProducts {
    // Own CSR buffers; exports are copies. Native symmetric operators append
    // rows to these vectors, with geometric capacity growth like SparseOp.
    // Snapshot modes retain owned immutable arrays.
    std::vector<double> values_;
    std::vector<Index> indices_, indptr_;
    Index nrow_, ncol_;
    int nthreads_;
    std::string mode_;
    bool symmetric_;
    struct Partition {
        Index begin, end;
    };
    std::vector<std::unique_ptr<Partition>> partitions_;
    mutable std::mutex mutex_;
    std::unique_ptr<ThreadTeam> team_;
    size_t append_count_ = 0, reallocations_ = 0, relocated_values_ = 0;
    std::vector<std::vector<double>> scratch_;

    template<class F> auto read(F action) const {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return action();
    }

    void native_rows(Index begin, Index end, const double *x, double *y,
                     Index columns) const {
        for (Index r = begin; r < end; ++r) {
            if (columns == 1) {
                double sum = 0., xr = symmetric_ ? x[r] : 0.;
                for (Index p = indptr_[r]; p < indptr_[r+1]; ++p) {
                    Index j = indices_[p];
                    double a = values_[p];
                    sum += a*x[j];
                    if (symmetric_ && j != r) y[j] += a*xr;
                }
                y[r] += sum;
            } else {
                for (Index p = indptr_[r]; p < indptr_[r+1]; ++p) {
                    Index j = indices_[p];
                    double a = values_[p];
                    for (Index c = 0; c < columns; ++c) y[r*columns+c] += a*x[j*columns+c];
                    if (symmetric_ && j != r)
                        for (Index c = 0; c < columns; ++c) y[j*columns+c] += a*x[r*columns+c];
                }
            }
        }
    }

    void apply(const double *x, double *y, Index columns) {
        const size_t size = static_cast<size_t>(nrow_)*columns;
        if (!size) return;
        if (partitions_.empty()) {
            std::fill(y, y+size, 0.0);
            return;
        }
        if (mode_ == "native-gather") {
            team_->apply(partitions_.size(),[&](size_t i) {
                const auto &part = *partitions_[i];
                std::fill(y+part.begin*columns,y+part.end*columns,0.);
                native_rows(part.begin,part.end,x,y,columns);
            });
            return;
        }
        if (partitions_.size() == 1) {
            std::fill(y, y+size, 0.);
            native_rows(0, nrow_, x, y, columns);
            return;
        }
        // Retain workers and buffers; reduce output tiles in partition order.
        if (!team_) team_=std::make_unique<ThreadTeam>(nthreads_);
        scratch_.resize(partitions_.size());
        team_->apply(partitions_.size(),[&](size_t i) {
            auto &buffer=scratch_[i];
            if (buffer.size()!=size) std::vector<double>(size).swap(buffer);
            else std::fill(buffer.begin(),buffer.end(),0.);
            const auto &part=*partitions_[i];
            native_rows(part.begin,part.end,x,buffer.data(),columns);
        });
        team_->apply((size+1023)/1024,[&](size_t tile) {
            for (size_t at=tile*1024;at<std::min(size,(tile+1)*1024);++at) {
                double value=scratch_[0][at];
                for (size_t i=1;i<scratch_.size();++i) value+=scratch_[i][at];
                y[at]=value;
            }
        });
        return;
    }

public:
    CSRProducts(const Dense &values, const py::array &indices, const py::array &indptr,
                Index nrow, Index ncol, bool symmetric, int nthreads, const std::string &mode)
        : nrow_(nrow), ncol_(ncol), nthreads_(nthreads), mode_(mode), symmetric_(symmetric) {
        if (nrow < 0 || ncol < 0 || (symmetric && nrow != ncol))
            throw std::invalid_argument("invalid CSR dimensions or symmetric rectangle");
        if (nthreads < 1 || nthreads > 256)
            throw std::invalid_argument("nthreads must be between 1 and 256");
        if (mode != "native" && mode != "native-gather")
            throw std::invalid_argument("product mode must be native or native-gather");
        // Reject lossy floating-point index casts at the native boundary.
        if (indices.dtype().kind() != 'i' || indptr.dtype().kind() != 'i')
            throw std::invalid_argument("CSR indices and indptr must be signed integers");
        Indices cols(indices), ptr(indptr);
        if (values.ndim() != 1 || cols.ndim() != 1 || ptr.ndim() != 1 ||
            values.size() != cols.size() || ptr.size()-1 != nrow)
            throw std::invalid_argument("inconsistent CSR array sizes");
        if (ptr.data()[0] != 0 || ptr.data()[nrow] != values.size())
            throw std::invalid_argument("CSR row pointers must span the stored entries");
        for (Index r = 0; r < nrow; ++r) {
            Index begin = ptr.data()[r], end = ptr.data()[r+1], previous = -1;
            if (begin < 0 || end < begin || end > values.size())
                throw std::invalid_argument("invalid CSR row pointers");
            for (Index p = begin; p < end; ++p) {
                Index c = cols.data()[p];
                if (c <= previous || c >= ncol || (symmetric && c > r))
                    throw std::invalid_argument("CSR columns must be sorted, unique, and in bounds; "
                                                "symmetric storage must be lower triangular");
                if (!std::isfinite(values.data()[p]))
                    throw std::invalid_argument("CSR values must be finite");
                previous = c;
            }
        }
        values_.assign(values.data(), values.data()+values.size());
        indices_.assign(cols.data(), cols.data()+cols.size());
        indptr_.assign(ptr.data(), ptr.data()+ptr.size());
        if (mode == "native-gather" && symmetric_) {
            py::gil_scoped_release release;
            // Expand the owned snapshot once; Python still exports lower CSR.
            // Row ownership removes reflected scatter, P full-size outputs,
            // and the serial P-by-N reduction from every product. Memory is
            // O(nnz+N), independent of the thread count.
            std::vector<Index> offsets(nrow_+1,0);
            for (Index r = 0; r < nrow_; ++r) {
                offsets[r+1] += indptr_[r+1]-indptr_[r];
                for (Index at = indptr_[r]; at < indptr_[r+1]; ++at)
                    if (indices_[at] != r) ++offsets[indices_[at]+1];
            }
            for (Index r = 0; r < nrow_; ++r) {
                if (offsets[r+1] > std::numeric_limits<Index>::max()-offsets[r])
                    throw std::length_error("full CSR index capacity exceeded");
                offsets[r+1] += offsets[r];
            }
            std::vector<Index> cursor(offsets), full_columns(offsets.back());
            std::vector<double> full_values(offsets.back());
            for (Index r = 0; r < nrow_; ++r) for (Index at = indptr_[r]; at < indptr_[r+1]; ++at) {
                Index c = indices_[at], position = cursor[r]++;
                full_values[position] = values_[at];
                full_columns[position] = c;
                if (c != r) {
                    position = cursor[c]++;
                    full_values[position] = values_[at];
                    full_columns[position] = r;
                }
            }
            values_.swap(full_values);
            indices_.swap(full_columns);
            indptr_.swap(offsets);
            symmetric_ = false;
        }
        if (values_.empty()) return;
        py::gil_scoped_release release;
        size_t count = std::min<size_t>(nthreads_, std::max<size_t>(1,
            values_.size()/(mode == "native-gather" ? 200000 : 1000000)));
        Index begin = 0;
        for (size_t i = 0; i < count; ++i) {
            Index end = i+1 == count ? nrow_ :
                std::lower_bound(indptr_.begin(), indptr_.end(),
                    (values_.size()*(i+1))/count)-indptr_.begin();
            if (end <= begin) continue;
            auto part = std::make_unique<Partition>();
            part->begin = begin;
            part->end = end;
            partitions_.push_back(std::move(part));
            begin = end;
        }
        if (mode == "native-gather") team_ = std::make_unique<ThreadTeam>(partitions_.size());
    }

    CSRProducts(const CSRProducts &) = delete;
    CSRProducts &operator=(const CSRProducts &) = delete;

    void append_rows(const Dense &values, const py::array &indices, const py::array &indptr,
                     Index nrow) {
        if (indices.dtype().kind() != 'i' || indptr.dtype().kind() != 'i')
            throw std::invalid_argument("CSR indices and indptr must be signed integers");
        Indices cols(indices), ptr(indptr);
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        if (mode_ != "native" || !symmetric_)
            throw std::invalid_argument("row append requires native symmetric CSR storage");
        if (nrow < nrow_ || nrow == std::numeric_limits<Index>::max() ||
            values.ndim() != 1 || cols.ndim() != 1 || ptr.ndim() != 1 ||
            values.size() != cols.size() || ptr.size() != nrow-nrow_+1)
            throw std::invalid_argument("invalid appended CSR dimensions");
        if (ptr.data()[0] != 0 || ptr.data()[ptr.size()-1] != values.size())
            throw std::invalid_argument("appended CSR row pointers must span the new entries");
        if (values_.size() > static_cast<size_t>(std::numeric_limits<Index>::max())-
                             static_cast<size_t>(values.size()))
            throw std::length_error("CSR index capacity exceeded");
        for (Index r = 0; r < nrow-nrow_; ++r) {
            Index begin = ptr.data()[r], end = ptr.data()[r+1], previous = -1;
            if (begin < 0 || end < begin || end > values.size())
                throw std::invalid_argument("invalid appended CSR row pointers");
            for (Index p = begin; p < end; ++p) {
                Index c = cols.data()[p];
                if (c <= previous || c > nrow_+r || !std::isfinite(values.data()[p]))
                    throw std::invalid_argument("appended CSR columns must be sorted, unique, lower triangular and finite");
                previous = c;
            }
        }
        if (nrow == nrow_) return;
        const size_t old_size = values_.size(), total = old_size+values.size();
        // Prepare all potentially throwing metadata before changing logical
        // contents. Binary searches span old row pointers and the new tail,
        // without allocating or copying a combined pointer array.
        auto offset = [&](Index r) -> Index {
            return r <= nrow_ ? indptr_[r] : old_size+ptr.data()[r-nrow_];
        };
        std::vector<std::unique_ptr<Partition>> next;
        if (total) {
            size_t count = std::min<size_t>(nthreads_, std::max<size_t>(1,total/1000000));
            next.reserve(count);
            Index begin = 0;
            for (size_t i = 0; i < count; ++i) {
                Index end = nrow;
                if (i+1 != count) {
                    Index lo = 0, hi = nrow+1, target = (total*(i+1))/count;
                    while (lo < hi) {
                        Index mid = lo+(hi-lo)/2;
                        if (offset(mid) < target) lo = mid+1;
                        else hi = mid;
                    }
                    end = lo;
                }
                if (end <= begin) continue;
                auto part = std::make_unique<Partition>();
                part->begin = begin; part->end = end;
                next.push_back(std::move(part));
                begin = end;
            }
        }
        auto reserve = [&](auto &buffer, size_t required) {
            if (required <= buffer.capacity()) return;
            size_t growth = buffer.capacity()/2+1;
            size_t capacity = buffer.capacity() > buffer.max_size()-growth ?
                              buffer.max_size() : buffer.capacity()+growth;
            buffer.reserve(std::max(required,capacity));
            ++reallocations_;
        };
        size_t previous_capacity = values_.capacity();
        reserve(values_, total);
        if (values_.capacity() != previous_capacity) relocated_values_ += old_size;
        reserve(indices_, total);
        reserve(indptr_, static_cast<size_t>(nrow)+1);
        // No allocation or other throwing operation remains. Failed reserves
        // above can alter capacity, but leave the old matrix usable and exact.
        values_.insert(values_.end(), values.data(), values.data()+values.size());
        indices_.insert(indices_.end(), cols.data(), cols.data()+cols.size());
        for (Index r = 1; r < ptr.size(); ++r) indptr_.push_back(old_size+ptr.data()[r]);
        nrow_ = ncol_ = nrow;
        partitions_.swap(next);
        ++append_count_;
    }

    py::array export_array(int field) const {
        py::array result;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            if (field == 0) {
                { py::gil_scoped_acquire acquire; result = Dense(values_.size()); }
                std::copy(values_.begin(),values_.end(),static_cast<double *>(result.mutable_data()));
            } else {
                const auto &buffer = field == 1 ? indices_ : indptr_;
                { py::gil_scoped_acquire acquire; result = Indices(buffer.size()); }
                std::copy(buffer.begin(),buffer.end(),static_cast<Index *>(result.mutable_data()));
            }
        }
        return result;
    }

    Dense diagonal() const {
        Dense result;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            { py::gil_scoped_acquire acquire; result = Dense(std::min(nrow_,ncol_)); }
            for (Index r = 0; r < result.size(); ++r) {
                auto begin = indices_.begin()+indptr_[r], end = indices_.begin()+indptr_[r+1];
                auto p = std::lower_bound(begin,end,r);
                result.mutable_data()[r] = p != end && *p == r ? values_[p-indices_.begin()] : 0.;
            }
        }
        return result;
    }

    double element(Index r, Index c) const {
        return read([&] {
            if (r < 0 || c < 0 || r >= nrow_ || c >= ncol_)
                throw std::out_of_range("CSR index out of bounds");
            if (symmetric_ && r < c) std::swap(r,c);
            auto begin = indices_.begin()+indptr_[r], end = indices_.begin()+indptr_[r+1];
            auto p = std::lower_bound(begin,end,c);
            return p != end && *p == c ? values_[p-indices_.begin()] : 0.;
        });
    }

    Dense product(const Dense &x, bool vector) {
        Dense y;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            if (x.ndim() != (vector ? 1 : 2) || x.shape(0) != ncol_)
                throw std::invalid_argument("trial array has incorrect dimensions");
            for (py::ssize_t i = 0; i < x.size(); ++i)
                if (!std::isfinite(x.data()[i]))
                    throw std::invalid_argument("trial array must be finite");
            Index columns = vector ? 1 : x.shape(1);
            std::vector<py::ssize_t> shape{nrow_};
            if (!vector) shape.push_back(columns);
            { py::gil_scoped_acquire acquire; y = Dense(shape); }
            apply(x.data(), y.mutable_data(), columns);
        }
        return y;
    }

    size_t storage_bytes() const {
        return read([&] { return values_.size()*sizeof(double)+(indices_.size()+indptr_.size())*sizeof(Index); });
    }
    auto shape() const { return read([&] { return std::make_pair(nrow_,ncol_); }); }
    size_t nnz() const { return read([&] { return values_.size(); }); }
    auto storage_info() const {
        return read([&] { return std::map<std::string,size_t>{
            {"capacity_bytes",values_.capacity()*sizeof(double)+(indices_.capacity()+indptr_.capacity())*sizeof(Index)},
            {"append_count",append_count_},{"reallocations",reallocations_},{"relocated_values",relocated_values_},
            {"retained_output_scratch_bytes",[&] { size_t bytes=0;for (const auto &s:scratch_) bytes+=s.capacity()*sizeof(double);return bytes; }()},
            {"retained_worker_team",bool(team_)}}; });
    }
    size_t active_threads() const { return read([&]() -> size_t { return values_.empty() ? 0 : partitions_.empty() ? nthreads_ : partitions_.size(); }); }
    size_t output_scratch_bytes_per_column() const {
        return read([&]() -> size_t { return mode_ != "native-gather" && partitions_.size() > 1 ?
               partitions_.size()*static_cast<size_t>(nrow_)*sizeof(double) : 0; });
    }
};
} // namespace

void bind_products(py::module_ &m) {
    py::class_<CSRProducts>(m, "_CSRProducts")
        .def(py::init([](const py::object &data, const py::array &indices,
                         const py::array &indptr, Index nrow, Index ncol,
                         bool symmetric, int nthreads, const std::string &mode) {
            return std::make_unique<CSRProducts>(real_array(data), indices, indptr,
                                                 nrow, ncol, symmetric, nthreads, mode);
        }), py::arg("data"), py::arg("indices"), py::arg("indptr"),
             py::arg("nrow"), py::arg("ncol"), py::arg("symmetric"), py::arg("nthreads") = 1,
             py::arg("mode") = "native")
        .def("matvec", [](CSRProducts &op, const py::object &x) {
            return op.product(real_array(x), true);
        })
        .def("matmat", [](CSRProducts &op, const py::object &x) {
            return op.product(real_array(x), false);
        })
        .def("append_rows", [](CSRProducts &op, const py::object &data, const py::array &indices,
                               const py::array &indptr, Index nrow) {
            op.append_rows(real_array(data),indices,indptr,nrow);
        }, py::arg("data"), py::arg("indices"), py::arg("indptr"), py::arg("nrow"))
        .def("data", [](const CSRProducts &op) { return op.export_array(0); })
        .def("indices", [](const CSRProducts &op) { return op.export_array(1); })
        .def("indptr", [](const CSRProducts &op) { return op.export_array(2); })
        .def("diagonal", &CSRProducts::diagonal)
        .def("get_element", &CSRProducts::element)
        .def_property_readonly("shape", &CSRProducts::shape)
        .def_property_readonly("nnz", &CSRProducts::nnz)
        .def_property_readonly("storage_info", &CSRProducts::storage_info)
        .def_property_readonly("storage_bytes", &CSRProducts::storage_bytes)
        .def_property_readonly("active_threads", &CSRProducts::active_threads)
        .def_property_readonly("output_scratch_bytes_per_column", &CSRProducts::output_scratch_bytes_per_column);
}
} // namespace scsf

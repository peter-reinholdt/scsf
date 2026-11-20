/* SCSF; GPL-3.0-or-later. */
#include "structured_auxiliary.h"
#include "structured_direct.h"
#include "structured_grouped.h"
#include "cih_wide.h"
#include "rdm.h"
#include "selection.h"
#include "shared_generator.h"
#include <optional>
#include <pybind11/stl.h>

namespace scsf {
namespace {
size_t checked_structured_workers(size_t count) {
    if(!count || count>256)throw std::invalid_argument("structured worker count must be in [1,256]");
    return count;
}
enum class Kernel { nminus2, cih, nminus1, particle_hole };

class StructuredHamiltonian {
    Array one_,two_;
    std::optional<Array> core_;
    structured::Primary primary_;
    structured::Workers workers_;
    std::unique_ptr<structured::Auxiliary> auxiliary_;
    std::unique_ptr<structured::MatrixDirect> direct_;
    std::unique_ptr<structured::GroupedUGA> grouped_;
    std::unique_ptr<structured::WideCIH> wide_cih_;
    size_t work_limit_,setup_ns_=0;
    Kernel kernel_;
    structured::SpatialERI integrals_{};
    std::vector<double> diagonal_;
    bool diagonal_ready_=false;
    mutable std::mutex mutex_;
public:
    StructuredHamiltonian(const Array &one,const Array &two,std::optional<Array> core,std::shared_ptr<const Basis> basis,int mode,size_t threads,
        size_t metadata,size_t coefficients,size_t work,size_t pair_limit,std::vector<int> cpus,bool private_output,
        size_t cache_limit,const StructuredHamiltonian *previous,bool slot_sweep):
        one_(one),two_(two),core_(std::move(core)),primary_([&] {
            pybind11::gil_scoped_release release;
            return structured::Primary(std::move(basis));
        }()),
        workers_(checked_structured_workers(threads),std::move(cpus)),work_limit_(work),kernel_(static_cast<Kernel>(mode)) {
        if(mode<0 || mode>3)throw std::invalid_argument("invalid Hamiltonian kernel");
        int k=primary_.k;
        size_t pairs=structured::orbital_pair(k,0);
        using structured::ERIFormat;
        auto format=core_ ? ERIFormat::thc : two.ndim()==1 ? ERIFormat::packed :
            two.ndim()==2 ? ERIFormat::cholesky : ERIFormat::dense;
        bool eri_shape=false;
        size_t rank=two.ndim()==2 ? size_t(two.shape(1)) : 0;
        switch(format) {
            case ERIFormat::dense:
                eri_shape=two.ndim()==4 && two.shape(0)==k && two.shape(1)==k && two.shape(2)==k && two.shape(3)==k;
                break;
            case ERIFormat::packed:
                eri_shape=size_t(two.shape(0))==structured::orbital_pair(pairs,0);
                break;
            case ERIFormat::cholesky:
                eri_shape=size_t(two.shape(0))==pairs;
                break;
            case ERIFormat::thc:
                eri_shape=two.ndim()==2 && two.shape(0)==k && core_->ndim()==2 &&
                    size_t(core_->shape(0))==rank && size_t(core_->shape(1))==rank;
        }
        if(one.ndim()!=2 || one.shape(0)!=k || one.shape(1)!=k || !eri_shape || (private_output && kernel_!=Kernel::nminus2))
            throw std::invalid_argument("invalid structured integral dimensions");
        integrals_={two.data(),core_ ? core_->data() : nullptr,size_t(k),rank,format};
        auto start=structured::Clock::now();pybind11::gil_scoped_release release;
        if(previous && (previous->kernel_!=kernel_ || previous->one_.data()!=one_.data() ||
                previous->two_.data()!=two_.data() || previous->core_.has_value()!=core_.has_value() ||
                (core_ && previous->core_->data()!=core_->data()) || previous->primary_.s!=primary_.s ||
                previous->primary_.n!=primary_.n))previous=nullptr;
        if(kernel_==Kernel::nminus2 || kernel_==Kernel::nminus1)auxiliary_=std::make_unique<structured::Auxiliary>(primary_,one.data(),integrals_,
            kernel_==Kernel::nminus2 ? 1 : 0,workers_,metadata,coefficients,pair_limit,private_output,
            previous ? previous->auxiliary_.get() : nullptr);
        else {
            if(kernel_==Kernel::cih)try {
                if(k<=64 && primary_.n<=structured::OneBank::mask_electron_limit) {
                    grouped_=std::make_unique<structured::GroupedUGA>(primary_,one.data(),integrals_,workers_,
                        metadata,coefficients,cache_limit,slot_sweep);
                    if(previous && previous->grouped_) {
                        std::lock_guard<std::mutex> lock(previous->mutex_);
                        grouped_->reuse_plans(*previous->grouped_);
                    }
                } else
                    wide_cih_=std::make_unique<structured::WideCIH>(primary_,one.data(),integrals_,workers_,metadata,coefficients);
            } catch(const BudgetError &) {
                // Fall back to the exact row product if templates exceed their budgets.
            }
            if(!grouped_ && !wide_cih_)direct_=std::make_unique<structured::MatrixDirect>(
                primary_,one.data(),integrals_,workers_,metadata,coefficients,cache_limit,kernel_==Kernel::cih);
        }
        setup_ns_=structured::ns(start);
    }
    Stats info()const {
        Stats s=auxiliary_ ? auxiliary_->info() : grouped_ ? grouped_->info() : wide_cih_ ? wide_cih_->info() : direct_->info();
        s["basis"]=primary_.size();s["configurations"]=primary_.groups.size();s["native_setup_ns"]=setup_ns_;
        s["workers"]=workers_.count();s["max_product_scratch_bytes"]=work_limit_;
        s["hamiltonian_offdiagonal_bytes"]=0;s["exact_product"]=1;
        using structured::ERIFormat;
        s["factored_eri"]=integrals_.format==ERIFormat::cholesky || integrals_.format==ERIFormat::thc;
        s["eri_storage_bytes"]=size_t(two_.nbytes())+(core_ ? core_->nbytes() : 0);
        if(s["factored_eri"])s["eri_factor_rank"]=integrals_.rank;
        const auto data=primary_.native->info();
        s.insert(data.begin(),data.end());
        s["basis_snapshot_bytes"]=data.at("rank_map_bytes")+data.at("occupation_bytes")
            +data.at("public_permutation_bytes")+data.at("spin_rank_bytes");
        s["decoded_occupation_bytes"]=primary_.configuration_bytes();
        s["primary_rank_bytes"]=primary_.groups.capacity()*sizeof(structured::Primary::Group);
        return s;
    }
    pybind11::tuple select_hci(const Array &weights,double eps,bool complete,
            std::shared_ptr<HCITransferBounds> bounds) {
        if(weights.ndim()!=1 || size_t(weights.size())!=primary_.size() || !std::isfinite(eps) || eps<0)
            throw std::invalid_argument("invalid native HCI weights or threshold");
        std::shared_ptr<Basis> next;
        std::vector<size_t> remap;
        Stats stats;
        {
            pybind11::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            for(size_t i=0;i<primary_.size();++i)
                if(!std::isfinite(weights.data()[i]) || weights.data()[i]<0.)
                    throw std::invalid_argument("HCI weights must be finite and nonnegative");
            auto [additions,data]=hci(primary_,workers_,one_.data(),integrals_,weights.data(),eps,
                complete,bounds.get());
            stats=std::move(data);
            if(!additions.empty()) {
                auto start=structured::Clock::now();
                std::tie(next,remap)=primary_.native->extend(additions);
                stats["basis_extend_ns"]=structured::ns(start);
            }
            stats["added"]=next ? next->size()-primary_.size() : 0;
        }
        pybind11::array_t<size_t> indices(remap.size());
        std::copy(remap.begin(),remap.end(),indices.mutable_data());
        return pybind11::make_tuple(next,indices,stats);
    }
    pybind11::tuple generator_probe(const Array &x,size_t link_bytes,size_t workspace_bytes,
                                  size_t coefficient_bytes) const {
        if(x.ndim()!=1 || size_t(x.size())!=primary_.size())
            throw std::invalid_argument("shared generator probe needs a native-row vector");
        Array y(x.size());
        Stats stats;
        {
            pybind11::gil_scoped_release release;
            std::fill_n(y.mutable_data(),y.size(),0.);
            stats=structured::shared_generator_product(primary_,one_.data(),integrals_,x.data(),y.mutable_data(),
                                               link_bytes,workspace_bytes,coefficient_bytes);
        }
        return pybind11::make_tuple(y,stats);
    }
    pybind11::tuple rdms(const Array &x,size_t output_bytes,size_t link_bytes,
                        size_t workspace_bytes,size_t coefficient_bytes) const {
        if(x.ndim()!=1 || size_t(x.size())!=primary_.size())
            throw std::invalid_argument("density matrices need a native-row coefficient vector");
        const size_t k=primary_.k,kk=k*k;
        if(k && (kk>output_bytes/sizeof(double) || kk>(output_bytes/sizeof(double)-kk)/kk))
            throw BudgetError("density matrices exceed output budget");
        if(!std::all_of(x.data(),x.data()+x.size(),[](double value){return std::isfinite(value);}))
            throw std::invalid_argument("density matrix coefficients must be finite");
        const pybind11::ssize_t dim=k;
        Array one({dim,dim}),two({dim,dim,dim,dim});
        Stats stats;
        {
            pybind11::gil_scoped_release release;
            std::fill_n(one.mutable_data(),one.size(),0.);
            std::fill_n(two.mutable_data(),two.size(),0.);
            stats=structured::occupation_rdms(primary_,x.data(),one.mutable_data(),two.mutable_data(),
                link_bytes,workspace_bytes,coefficient_bytes);
        }
        return pybind11::make_tuple(one,two,stats);
    }
    void reuse_diagonal(StructuredHamiltonian &old,const pybind11::array_t<size_t,pybind11::array::c_style> &map) {
        if(this==&old || one_.data()!=old.one_.data() ||
            two_.data()!=old.two_.data() || primary_.size()<old.primary_.size() || map.ndim()!=1 ||
            (map.size() && size_t(map.size())!=old.primary_.size()))
            throw std::invalid_argument("incompatible diagonal generation");
        pybind11::gil_scoped_release release;
        std::scoped_lock lock(mutex_,old.mutex_);
        if(old.diagonal_.empty())return;
        std::vector<double> copy(primary_.size(),std::numeric_limits<double>::quiet_NaN());
        for(size_t i=0;i<old.diagonal_.size();++i)copy.at(map.size() ? map.data()[i] : i)=old.diagonal_[i];
        diagonal_=std::move(copy);
        diagonal_ready_=false;
    }
    Array diagonal() {
        size_t n=primary_.size();
        Array result(n);
        {
            pybind11::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            if(!diagonal_ready_) {
                std::vector<double> coulomb_exchange;
                {
                    size_t k=primary_.k,kk=k*k;
                    coulomb_exchange.resize(2*kk);
                    workers_.run(k,[&](size_t p,size_t) {
                        for(size_t r=0;r<k;++r) {
                            coulomb_exchange[p*k+r]=integrals_(p,p,r,r);
                            coulomb_exchange[kk+p*k+r]=integrals_(p,r,p,r);
                        }
                    });
                }
                diagonal_.resize(n,std::numeric_limits<double>::quiet_NaN());
                workers_.run(primary_.groups.size(),[&](size_t g,size_t) {
                    const auto &group=primary_.groups[g];
                    size_t local=0;
                    while(local<group.size() && !std::isnan(diagonal_[group.row(local)]))++local;
                    if(local==group.size())return;
                    if(grouped_ || wide_cih_) {
                        // Diagonal reuse is most common for whole occupation
                        // blocks. Recompute a partially changed block as a unit.
                        const auto *coulomb=coulomb_exchange.data();
                        const auto *exchange=coulomb?coulomb+size_t(primary_.k)*primary_.k:nullptr;
                        if(grouped_)grouped_->diagonal(g,one_.data(),coulomb,exchange,diagonal_.data());
                        else wide_cih_->diagonal(g,one_.data(),coulomb,exchange,diagonal_.data());
                        return;
                    }
                    auto c=guga_diagonal_connection(*group.cfg,one_.data(),coulomb_exchange.data());
                    for(;local<group.size();++local) {
                        auto &value=diagonal_[group.row(local)];
                        if(!std::isnan(value))continue;
                        const Path path(primary_.label(g,local),primary_.k,primary_.s,0);
                        value=guga_element(path,path,c);
                    }
                },16);
                diagonal_ready_=true;
            }
            std::copy(diagonal_.begin(),diagonal_.end(),result.mutable_data());
        }
        return result;
    }
    pybind11::tuple apply(const Array &x) {
        size_t n=primary_.size();
        if(x.ndim()!=2 || size_t(x.shape(0))!=n)throw std::invalid_argument("trial block shape mismatch");
        for(pybind11::ssize_t i=0;i<x.size();++i)if(!std::isfinite(x.data()[i]))throw std::invalid_argument("nonfinite trial block");
        size_t cols=x.shape(1);Array result({pybind11::ssize_t(n),pybind11::ssize_t(cols)});Stats stats;
        auto start=structured::Clock::now();
        {
            pybind11::gil_scoped_release release;std::lock_guard<std::mutex> lock(mutex_);
            std::fill(result.mutable_data(),result.mutable_data()+n*cols,0.);
            const size_t entries=auxiliary_ ? auxiliary_->entries() : grouped_ ? grouped_->entries() : wide_cih_ ? wide_cih_->entries() : direct_->entries();
            if(entries>no_row/sizeof(double))throw BudgetError("structured product scratch size overflow");
            size_t per_column=entries*sizeof(double);
            const bool over_budget=per_column>work_limit_;
            const bool batched=over_budget && auxiliary_ && auxiliary_->can_batch(work_limit_);
            const bool serial=over_budget && !batched;
            if(over_budget)per_column=0;
            size_t width=over_budget ? std::min(size_t(1),cols) : per_column ? std::min(cols,work_limit_/per_column) : cols;
            stats["serial_streaming_fallback"]=serial;
            if(cols && !width)throw BudgetError("structured scratch cap cannot fit one RHS column");
            stats["product_array_bytes"]=per_column*width;
            for(size_t first=0;first<cols;first+=width) {
                size_t count=std::min(width,cols-first);const double *input=x.data();double *output=result.mutable_data();
                std::vector<double> input_tile,output_tile;
                if(count!=cols) {
                    input_tile.resize(n*count);output_tile.assign(n*count,0.);
                    for(size_t i=0;i<n;++i)for(size_t v=0;v<count;++v)input_tile[i*count+v]=x.data()[i*cols+first+v];
                    input=input_tile.data();output=output_tile.data();
                }
                if(batched)auxiliary_->apply_batched(input,output,work_limit_,stats);
                else if(auxiliary_)auxiliary_->apply(input,count,output,stats,serial);
                else if(grouped_)grouped_->apply(input,count,output,stats,serial);
                else if(wide_cih_)wide_cih_->apply(input,count,output,stats,serial);
                else direct_->apply(input,count,output,stats,serial);
                if(count!=cols)for(size_t i=0;i<n;++i)for(size_t v=0;v<count;++v)result.mutable_data()[i*cols+first+v]=output[i*count+v];
                ++stats["rhs_tiles"];
            }
            stats["native_total_ns"]=structured::ns(start);
        }
        return pybind11::make_tuple(result,stats);
    }
};
}
void bind_structured(pybind11::module_ &m) {
    namespace py=pybind11;
    py::class_<Basis,std::shared_ptr<Basis>>(m,"_OccupationBasis")
        .def(py::init([](int k,int n,int s,const std::vector<std::string> &labels) {
            py::gil_scoped_release release;
            return std::make_shared<Basis>(k,n,s,labels);
        }))
        .def("__len__",&Basis::size)
        .def("public_label",&Basis::public_label)
        .def("native_label",&Basis::native_label)
        .def_property_readonly("info",&Basis::info)
        .def_property_readonly("public_rows",[](const Basis &b) {
            const auto &rows=b.public_rows();
            if(rows.empty())return py::array_t<size_t>(0);
            py::array_t<size_t> a(rows.size(),rows.data(),py::cast(&b,py::return_value_policy::reference));
            a.attr("setflags")(false);
            return a;
        });
    py::class_<StructuredHamiltonian>(m,"_StructuredHamiltonian")
        .def(py::init([](const Array &one,const Array &two,std::optional<Array> core,
                std::shared_ptr<const Basis> basis,int mode,size_t threads,size_t metadata,size_t coefficients,
                size_t work,size_t pair_limit,std::vector<int> cpus,bool private_output,size_t cache_limit,const StructuredHamiltonian *previous,bool slot_sweep) {
            if(!basis)throw std::invalid_argument("native occupation basis is required");
            return std::make_unique<StructuredHamiltonian>(one,two,std::move(core),std::move(basis),mode,threads,
                metadata,coefficients,work,pair_limit,std::move(cpus),private_output,cache_limit,previous,slot_sweep);
        }))
        .def("select_hci",&StructuredHamiltonian::select_hci)
        .def("rdms",&StructuredHamiltonian::rdms)
        .def("_generator_probe",&StructuredHamiltonian::generator_probe)
        .def("reuse_diagonal",&StructuredHamiltonian::reuse_diagonal)
        .def("apply",&StructuredHamiltonian::apply)
        .def("diagonal",&StructuredHamiltonian::diagonal)
        .def_property_readonly("info",&StructuredHamiltonian::info);
}
} // namespace scsf

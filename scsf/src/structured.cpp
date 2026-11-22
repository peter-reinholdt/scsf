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
    std::shared_ptr<HCITransferBounds> bounds_;
    size_t memory_limit_,setup_ns_=0,peak_memory_=0;
    Kernel kernel_;
    structured::SpatialERI integrals_{};
    std::vector<double> diagonal_;
    bool diagonal_ready_=false;
    mutable std::mutex mutex_;

    Stats kernel_info() const {
        return auxiliary_ ? auxiliary_->info() : grouped_ ? grouped_->info() : wide_cih_ ? wide_cih_->info() : direct_->info();
    }
    size_t bound_bytes() const {return bounds_ ? bounds_->bytes() : 0;}
    void drop_workspace() const {
        if(grouped_)grouped_->release_workspace();
        if(auxiliary_)auxiliary_->release_workspace();
    }
    void record_peak(size_t bytes,const char *phase) {
        memory_left(memory_limit_,bytes,phase);
        peak_memory_=std::max(peak_memory_,bytes);
    }
public:
    StructuredHamiltonian(const Array &one,const Array &two,std::optional<Array> core,std::shared_ptr<const Basis> basis,int mode,size_t threads,
        size_t memory,std::vector<int> cpus,bool private_output,std::shared_ptr<HCITransferBounds> bounds,
        const StructuredHamiltonian *previous,bool reuse,bool slot_sweep):
        one_(one),two_(two),core_(std::move(core)),primary_([&] {
            pybind11::gil_scoped_release release;
            return structured::Primary(std::move(basis));
        }()),
        workers_(checked_structured_workers(threads),std::move(cpus)),bounds_(std::move(bounds)),memory_limit_(memory),kernel_(static_cast<Kernel>(mode)) {
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
        std::unique_lock<std::mutex> previous_lock;
        size_t old_metadata=0,old_coefficients=0;
        if(previous) {
            previous_lock=std::unique_lock<std::mutex>(previous->mutex_);
            previous->drop_workspace();
            const auto old=previous->kernel_info();
            old_metadata=old.at("metadata_bytes");
            old_coefficients=old.at("coefficient_bytes");
            peak_memory_=previous->peak_memory_;
            if(previous->kernel_!=kernel_ || previous->one_.data()!=one_.data() ||
                previous->two_.data()!=two_.data() || previous->core_.has_value()!=core_.has_value() ||
                (core_ && previous->core_->data()!=core_->data()) || previous->primary_.s!=primary_.s ||
                previous->primary_.n!=primary_.n)reuse=false;
        }
        const size_t tables=bound_bytes(), available=memory_left(memory_limit_,tables,"HCI transfer tables");
        const size_t old_kernel=checked_add(old_metadata,old_coefficients);
        const size_t fresh=memory_left(available,old_kernel,"existing Hamiltonian operator");
        if(kernel_==Kernel::nminus2 || kernel_==Kernel::nminus1) {
            const auto *prior=reuse && previous ? previous->auxiliary_.get() : nullptr;
            // N−1 may hold a reference to the old ERI layout instead of
            // allocating it again. It still counts in the new operator.
            const size_t shared_eri=kernel_==Kernel::nminus1 && prior ? prior->auxiliary_integral_bytes() : 0;
            auxiliary_=std::make_unique<structured::Auxiliary>(primary_,one.data(),integrals_,
                kernel_==Kernel::nminus2 ? 1 : 0,workers_,checked_add(fresh,shared_eri),private_output,prior);
            const auto next=auxiliary_->info();
            const bool shared=next.at("reused_removal_topology");
            const size_t new_metadata=std::max(next.at("metadata_bytes"),next.at("metadata_build_bytes"));
            size_t peak=shared ? old_kernel : checked_add(old_kernel,checked_add(new_metadata,next.at("coefficient_bytes")));
            if(!shared && next.count("reused_auxiliary_integrals") && next.at("reused_auxiliary_integrals"))
                peak-=next.at("auxiliary_integral_bytes");
            record_peak(checked_add(tables,peak),"auxiliary Hamiltonian setup");
        } else {
            size_t shared_coefficients=0;
            if(kernel_==Kernel::cih) {
                if(k<=64 && primary_.n<=structured::OneBank::mask_electron_limit) {
                    const auto *prior=reuse && previous ? previous->grouped_.get() : nullptr;
                    if(prior && !prior->can_share_bank(primary_))prior=nullptr;
                    if(prior)shared_coefficients=old_coefficients;
                    grouped_=std::make_unique<structured::GroupedUGA>(primary_,one.data(),integrals_,workers_,
                        checked_add(fresh,shared_coefficients),slot_sweep,prior);
                } else
                    wide_cih_=std::make_unique<structured::WideCIH>(primary_,one.data(),integrals_,workers_,fresh);
            } else {
                direct_=std::make_unique<structured::MatrixDirect>(primary_,one.data(),integrals_,workers_,fresh);
            }
            const auto next=kernel_info();
            const auto build=next.find("metadata_build_bytes");
            const size_t metadata=build==next.end() ? next.at("metadata_bytes") : build->second;
            const size_t new_coefficients=next.at("coefficient_bytes")-shared_coefficients;
            record_peak(checked_add(tables,checked_add(old_kernel,checked_add(metadata,new_coefficients))),
                        "Hamiltonian setup");
            if(grouped_ && reuse && previous && previous->grouped_)
                grouped_->reuse_plans(*previous->grouped_);
        }
        setup_ns_=structured::ns(start);
    }
    Stats info()const {
        std::lock_guard<std::mutex> lock(mutex_);
        Stats s=kernel_info();
        s["basis"]=primary_.size();s["configurations"]=primary_.groups.size();s["native_setup_ns"]=setup_ns_;
        s["workers"]=workers_.count();
        s["max_mem"]=memory_limit_;
        s["hci_table_bytes"]=bound_bytes();
        s["tracked_memory_bytes"]=checked_add(bound_bytes(),checked_add(s.at("metadata_bytes"),s.at("coefficient_bytes")));
        if(grouped_ || auxiliary_) {
            s["retained_workspace_bytes"]=grouped_ ? grouped_->workspace_bytes() : auxiliary_->workspace_bytes();
            s["tracked_memory_bytes"]=checked_add(s["tracked_memory_bytes"],s["retained_workspace_bytes"]);
        }
        s["peak_tracked_memory_bytes"]=std::max(peak_memory_,s["tracked_memory_bytes"]);
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
    void release_product_workspace() {
        pybind11::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        drop_workspace();
    }
    pybind11::tuple select_hci(const Array &weights,double eps,bool complete,
            double stop_growth_fraction) {
        if(weights.ndim()!=1 || size_t(weights.size())!=primary_.size() || !std::isfinite(eps) || eps<0)
            throw std::invalid_argument("invalid native HCI weights or threshold");
        if(!(stop_growth_fraction>=0. && stop_growth_fraction<=1.))
            throw std::invalid_argument("stop_growth_fraction must be in [0, 1]");
        std::shared_ptr<Basis> next;
        std::vector<size_t> remap;
        Stats stats;
        {
            pybind11::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            for(size_t i=0;i<primary_.size();++i)
                if(!std::isfinite(weights.data()[i]) || weights.data()[i]<0.)
                    throw std::invalid_argument("HCI weights must be finite and nonnegative");
            drop_workspace();
            const auto current=kernel_info();
            const size_t kernel_bytes=checked_add(current.at("metadata_bytes"),current.at("coefficient_bytes"));
            const size_t tables=memory_left(memory_limit_,kernel_bytes,"Hamiltonian storage during HCI selection");
            auto [additions,data]=hci(primary_,workers_,one_.data(),integrals_,weights.data(),eps,
                complete,bounds_.get(),tables);
            record_peak(checked_add(kernel_bytes,bound_bytes()),"HCI transfer tables");
            stats=std::move(data);
            stats["hci_table_bytes"]=bound_bytes();
            const size_t proposed=stats.at("proposed_csfs");
            if(proposed && double(proposed)/primary_.size()>stop_growth_fraction) {
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
            {
                std::lock_guard<std::mutex> lock(mutex_);
                drop_workspace();
            }
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
        std::vector<double> copy;
        if(!map.size()) {
            copy.reserve(primary_.size());
            copy.insert(copy.end(),old.diagonal_.begin(),old.diagonal_.end());
            copy.resize(primary_.size(),std::numeric_limits<double>::quiet_NaN());
        } else {
            copy.resize(primary_.size(),std::numeric_limits<double>::quiet_NaN());
            for(size_t i=0;i<old.diagonal_.size();++i)copy.at(map.data()[i])=old.diagonal_[i];
        }
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
            const size_t entries=auxiliary_ ? auxiliary_->entries() : grouped_ ? grouped_->entries() :
                wide_cih_ ? wide_cih_->entries() : direct_->entries();
            const auto current=kernel_info();
            const size_t retained=checked_add(bound_bytes(),checked_add(current.at("metadata_bytes"),current.at("coefficient_bytes")));
            const size_t available=memory_left(memory_limit_,retained,"Hamiltonian storage");
            if(cols && entries>available/sizeof(double)/cols)
                throw BudgetError("Hamiltonian product arrays for "+std::to_string(cols)+
                                  " RHS columns exceed remaining --max-mem ("+std::to_string(available)+" bytes)");
            const size_t arrays=entries*cols*sizeof(double);
            if(cols!=1)drop_workspace();
            record_peak(checked_add(retained,arrays),"Hamiltonian product arrays");
            stats["product_array_bytes"]=arrays;
            std::fill(result.mutable_data(),result.mutable_data()+n*cols,0.);
            if(cols) {
                if(auxiliary_)auxiliary_->apply(x.data(),cols,result.mutable_data(),stats);
                else if(grouped_) {
                    const size_t cached=current.at("plan_cache_bytes");
                    try {
                        grouped_->apply(x.data(),cols,result.mutable_data(),stats,checked_add(cached,available-arrays));
                    } catch (...) {
                        record_peak(retained+arrays+(grouped_->cached_bytes()-cached),"CIH plans and product arrays");
                        throw;
                    }
                }
                else if(wide_cih_)wide_cih_->apply(x.data(),cols,result.mutable_data(),stats);
                else direct_->apply(x.data(),cols,result.mutable_data(),stats);
                stats["rhs_tiles"]=1;
            }
            if(grouped_)
                record_peak(checked_add(retained,checked_add(arrays,grouped_->cached_bytes()-current.at("plan_cache_bytes"))),
                            "CIH plans and product arrays");
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
                std::shared_ptr<const Basis> basis,int mode,size_t threads,size_t memory,
                std::vector<int> cpus,bool private_output,std::shared_ptr<HCITransferBounds> bounds,
                const StructuredHamiltonian *previous,bool reuse,bool slot_sweep) {
            if(!basis)throw std::invalid_argument("native occupation basis is required");
            return std::make_unique<StructuredHamiltonian>(one,two,std::move(core),std::move(basis),mode,threads,
                memory,std::move(cpus),private_output,std::move(bounds),previous,reuse,slot_sweep);
        }))
        .def("select_hci",&StructuredHamiltonian::select_hci,pybind11::arg("weights"),
             pybind11::arg("eps"),pybind11::arg("complete"),pybind11::arg("stop_growth_fraction")=0.)
        .def("rdms",&StructuredHamiltonian::rdms)
        .def("_generator_probe",&StructuredHamiltonian::generator_probe)
        .def("reuse_diagonal",&StructuredHamiltonian::reuse_diagonal)
        .def("apply",&StructuredHamiltonian::apply)
        .def("release_product_workspace",&StructuredHamiltonian::release_product_workspace)
        .def("diagonal",&StructuredHamiltonian::diagonal)
        .def_property_readonly("info",&StructuredHamiltonian::info);
}
} // namespace scsf

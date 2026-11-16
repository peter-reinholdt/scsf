/* This file is part of SCSF, licensed under GNU GPL version 3 or later. */
#include "selected_basis.h"
#include "pair_descriptor.h"
#include "selected_index.h"
#include <pybind11/stl.h>
#include <algorithm>
#include <atomic>
#include <numeric>
#include <set>

namespace py = pybind11;
namespace scsf {
using namespace pybind11::literals;
void validate_config(const Config &config) {
    for (int n : config)
        if (n < 0 || n > 2) throw std::invalid_argument("occupations must be 0,1,2");
}

Groups groups(const std::vector<std::string> &labels, int k, int two_s,
              size_t start, size_t end) {
    Groups result;
    int electrons = -1;
    std::set<std::string> unique;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i].size() != static_cast<size_t>(k))
            throw std::invalid_argument("CSF label length differs from orbital count");
        Config config = configuration(labels[i], two_s);
        int n = std::accumulate(config.begin(), config.end(), 0);
        if (electrons < 0) electrons = n;
        if (n != electrons) throw std::invalid_argument("CSFs must share electron number");
        if (!unique.insert(labels[i]).second)
            throw std::invalid_argument("duplicate CSF label");
        if (start <= i && i < end) result[config].push_back(i);
    }
    return result;
}


SelectedBasis::SelectedBasis(int k, int n, int spin, std::vector<std::string> labels, size_t generation, bool packed)
    : context_{k,n,spin,generation,0}, labels_(std::move(labels)),packed_(packed),ncsf_(labels_.size()) {
    if (k<0 || n<0 || static_cast<long long>(n)>2LL*k || spin<0 ||
        spin>std::min(static_cast<long long>(n),2LL*k-n) || (n-spin)%2)
        throw std::invalid_argument("incompatible CSF basis sector");
    auto owned=scsf::groups(labels_,k,spin,0,labels_.size());
    if (!owned.empty() && std::accumulate(owned.begin()->first.begin(),owned.begin()->first.end(),0)!=n)
        throw std::invalid_argument("CSFs have the wrong electron number");
    indices_.reserve(labels_.size());
    configuration_ids_.resize(labels_.size());
    offsets_.reserve(owned.size()+1);
    order_.reserve(owned.size());
    ordinal_.reserve(owned.size());
    for (const auto &group:owned) {
        size_t id=order_.size(), offset=indices_.size();
        offsets_.push_back(offset);
        indices_.insert(indices_.end(),group.second.begin(),group.second.end());
        for (size_t csf:group.second) configuration_ids_[csf]=id;
        auto at=groups_.emplace(group.first,IndexView(indices_.data()+offset,group.second.size())).first;
        order_.push_back(&*at);
        ordinal_.emplace(&*at,id);
    }
    offsets_.push_back(indices_.size());
    if (packed_) {
        local_ranks_.resize(ncsf_);
        spin_offsets_.reserve(order_.size()+1);
        size_t bits=0;
        for (const auto *group:order_) {
            spin_offsets_.push_back(bits);
            size_t singles=std::count(group->first.begin(),group->first.end(),1);
            if (singles && group->second.size()>(std::numeric_limits<size_t>::max()-bits)/singles)
                throw std::length_error("packed spin-path size overflow");
            bits+=singles*group->second.size();
        }
        spin_offsets_.push_back(bits);
        spin_words_.assign(bits/64+size_t(bits%64!=0),0);
        for (size_t g=0;g<order_.size();++g) {
            size_t bit=spin_offsets_[g],rank=0;
            for (size_t csf:order_[g]->second) {
                local_ranks_[csf]=rank++;
                for (char step:labels_[csf]) if (step=='u' || step=='d') {
                    if (step=='d') spin_words_[bit/64]|=std::uint64_t(1)<<(bit%64);
                    ++bit;
                }
            }
        }
        // Drop the source labels, including their allocation. Compact mode
        // must not silently retain a second copy of the basis representation.
        std::vector<std::string>().swap(labels_);
    }
    // Snapshot IDs are process-local identity, never a hash of labels or a
    // restart identity. Equal generation counters on unrelated bases are safe.
    static std::atomic<std::uint64_t> next{1};
    context_.snapshot_id=next.fetch_add(1);
}

const std::string &SelectedBasis::label(size_t csf,std::string &scratch) const {
    if (csf>=ncsf_) throw std::out_of_range("CSF index outside basis");
    if (!packed_) return labels_[csf];
    size_t g=configuration_ids_[csf];
    const auto &config=order_[g]->first;
    size_t singles=std::count(config.begin(),config.end(),1);
    size_t bit=spin_offsets_[g]+local_ranks_[csf]*singles;
    scratch.resize(config.size());
    for (size_t p=0;p<config.size();++p) {
        if (config[p]==1) {
            scratch[p]=(spin_words_[bit/64]>>(bit%64))&1 ? 'd' : 'u';
            ++bit;
        } else scratch[p]=config[p]==2 ? '2' : '0';
    }
    return scratch;
}
std::vector<std::string> SelectedBasis::labels() const {
    std::vector<std::string> result;result.reserve(ncsf_);
    std::string scratch;
    for (size_t i=0;i<ncsf_;++i) result.push_back(label(i,scratch));
    return result;
}
std::string SelectedBasis::spin_path(size_t csf) const {
    std::string result,scratch;
    for (char step:label(csf,scratch)) if (step=='u' || step=='d') result.push_back(step);
    return result;
}
Stats SelectedBasis::storage_info() const {
    size_t label_bytes=labels_.capacity()*sizeof(std::string);
    for (const auto &s:labels_) label_bytes+=s.capacity()+1;
    return {{"packed_spin_paths",packed_},{"retained_label_bytes_estimate",label_bytes},
        {"spin_word_bytes",spin_words_.capacity()*sizeof(std::uint64_t)},
        {"spin_position_bytes",(local_ranks_.capacity()+spin_offsets_.capacity())*sizeof(size_t)}};
}
size_t SelectedBasis::bytes_estimate() const {
    size_t bytes=sizeof(SelectedBasis)+labels_.capacity()*sizeof(std::string)+
        (indices_.capacity()+configuration_ids_.capacity()+offsets_.capacity()+
         local_ranks_.capacity()+spin_offsets_.capacity())*sizeof(size_t)+spin_words_.capacity()*sizeof(std::uint64_t)+
        order_.capacity()*sizeof(const Group *)+ordinal_.bucket_count()*sizeof(void *)+
        ordinal_.size()*4*sizeof(void *);
    for (const auto &s:labels_) bytes+=s.capacity()+1;
    for (const auto &g:groups_) bytes+=sizeof(Group)+4*sizeof(void *)+g.first.capacity()*sizeof(int);
    return bytes;
}

void bind_selected_basis(py::module_ &m) {
    py::class_<SelectedBasis,std::shared_ptr<SelectedBasis>>(m,"_SelectedBasis")
        .def(py::init<int,int,int,std::vector<std::string>,size_t,bool>(),
             py::arg("nbasis"),py::arg("nelec"),py::arg("two_s"),py::arg("labels"),py::arg("generation")=0,py::arg("packed")=false)
        .def_property_readonly("context",[](const SelectedBasis &b) {
            const auto &c=b.context();
            return py::dict("nbasis"_a=c.nbasis,"nelec"_a=c.nelec,"two_s"_a=c.two_s,
                "orbital_order"_a=c.orbital_order,"phase_convention_id"_a=c.phase_convention,
                "symmetry_convention_id"_a=c.symmetry_convention,"basis_generation"_a=c.generation,
                "snapshot_id"_a=c.snapshot_id);
        })
        .def_property_readonly("labels",[](const SelectedBasis &b) { return py::tuple(py::cast(b.labels())); })
        .def_property_readonly("configuration_indices",&SelectedBasis::indices)
        .def_property_readonly("configuration_offsets",&SelectedBasis::offsets)
        .def_property_readonly("configuration_ids",&SelectedBasis::configuration_ids)
        .def_property_readonly("bytes_estimate",&SelectedBasis::bytes_estimate)
        .def_property_readonly("storage_info",&SelectedBasis::storage_info)
        .def("spin_path",&SelectedBasis::spin_path)
        .def("occupations",[](const SelectedBasis &b,size_t id) { return b.order().at(id)->first; })
        .def("pair_descriptor",[](const SelectedBasis &b,size_t i,size_t j) {
            const auto &a=b.order().at(i)->first,&ket=b.order().at(j)->first;
            PairDescriptor pair(a,ket,i,j);
            std::vector<int> holes,particles,physical(a.size());
            for (size_t p=0;p<a.size();++p) {
                for (int n=ket[p];n>a[p];--n) holes.push_back(int(p));
                for (int n=a[p];n>ket[p];--n) particles.push_back(int(p));
            }
            std::iota(physical.begin(),physical.end(),0);
            std::vector<std::array<int,2>> one;
            std::vector<std::array<int,4>> two;
            pair.one_body_patterns([&](int p,int q) { one.push_back({p,q}); });
            pair.two_body_patterns([&](int q,int s,int p,int r) { two.push_back({q,s,p,r}); });
            return py::dict("I"_a=i,"J"_a=j,"transfer_rank"_a=pair.transfer_rank,
                "connected"_a=pair.connected(),"holes"_a=holes,"particles"_a=particles,
                "reduced_to_physical_orbitals"_a=physical,"one_body_patterns"_a=one,"two_body_patterns"_a=two);
        })
        .def("neighbors",[](const SelectedBasis &b,size_t id,int mode,size_t budget) {
            const auto &source=b.order().at(id)->first;
            ConfigurationIndex<SelectedBasis::Groups> index(b.groups(),mode,budget);
            Stats stats;
            std::vector<size_t> result;
            for (auto group:index.query(source,std::numeric_limits<size_t>::max(),stats))
                result.push_back(b.ordinals().at(group));
            return result;
        },py::arg("configuration_id"),py::arg("mode")=2,py::arg("max_bytes")=64*1024*1024);
}
} // namespace scsf

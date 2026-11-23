/* SCSF; GPL-3.0-or-later. Spin-scalar products over occupation blocks. */
#include "structured_internal.h"
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <complex>

namespace scsf {
namespace py = pybind11;
namespace {
using Complex = std::complex<double>;
using ComplexArray = py::array_t<Complex, py::array::c_style>;

bool finite(Complex value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

void validate(const ComplexArray &one, const std::shared_ptr<const Basis> &basis, size_t workers) {
    if (!basis || one.ndim()!=2 || one.shape(0)!=basis->orbitals() || one.shape(1)!=basis->orbitals())
        throw std::invalid_argument("one-electron operator shape must match the orbital count");
    if (!workers || workers>256)
        throw std::invalid_argument("worker count must be in [1,256]");
    for (py::ssize_t i=0;i<one.size();++i)
        if (!finite(one.data()[i])) throw std::invalid_argument("one-electron operator must be finite");
}

std::vector<size_t> public_indices(const Basis &basis) {
    const auto &forward=basis.public_rows();
    std::vector<size_t> inverse(forward.size());
    for (size_t i=0;i<forward.size();++i) inverse[forward[i]]=i;
    return inverse;
}

std::vector<Path> prepare_paths(const structured::Primary &primary) {
    std::vector<Path> paths;
    paths.reserve(primary.size());
    for (size_t group=0;group<primary.groups.size();++group)
        for (size_t local=0;local<primary.groups[group].size();++local)
            paths.emplace_back(primary.label(group,local),primary.k,primary.s,2);
    return paths;
}

class PropertyTransfers {
    using Terms=std::array<std::vector<int>,OrbitalSymmetry::extent>;
    std::vector<Terms> terms_;
public:
    PropertyTransfers(const structured::Primary &p,const Complex *one,bool adjoint):terms_(p.k) {
        const auto &sym=p.native->symmetry();
        for(int i=0;i<p.k;++i)for(int j=0;j<p.k;++j) {
            const size_t at=adjoint ? size_t(i)*p.k+j : size_t(j)*p.k+i;
            if(i!=j && one[at]!=Complex{})terms_[i][sym.transfer(i,j)].push_back(j);
        }
    }
    template<class Visit>void visit(int orbital,unsigned allowed,Visit callback) const {
        for(;allowed;allowed&=allowed-1) {
            const unsigned g=__builtin_ctz(allowed);
            for(int other:terms_[orbital][g])callback(other);
        }
    }
};

// Destination-owned tiles apply each directed E_pq block once. Public row
// numbers are used at the vector boundary so a coefficient prefix can be
// read without padding or scattering between occupation tiles.
ComplexArray one_electron_product(const ComplexArray &one,
        std::shared_ptr<const Basis> basis, const ComplexArray &x,
        size_t nthreads) {
    validate(one,basis,nthreads);
    if (x.ndim()!=2 || size_t(x.shape(0))>basis->size())
        throw std::invalid_argument("one-electron coefficients need a selected-basis prefix block");
    for (py::ssize_t i=0;i<x.size();++i)
        if (!finite(x.data()[i])) throw std::invalid_argument("one-electron coefficients must be finite");
    ComplexArray output({static_cast<py::ssize_t>(basis->size()),x.shape(1)});
    const size_t columns=x.shape(1), source_size=x.shape(0);
    if (!columns || !basis->size()) return output;

    auto *y=output.mutable_data();
    const auto *input=x.data();
    const auto *integrals=one.data();
    py::gil_scoped_release release;
    const structured::Primary primary(std::move(basis));
    const size_t k=primary.k;
    const auto paths=prepare_paths(primary);
    const PropertyTransfers transfers(primary,integrals,true);
    const auto indices=public_indices(*primary.native);
    auto public_row=[&](size_t row) {return indices.empty() ? row : indices[row];};
    unsigned sectors=0;
    for(const auto &group:primary.groups) {
        for(size_t i=0;i<group.size();++i)if(public_row(group.row(i))<source_size) {
            sectors|=1u<<group.block->irrep;
            break;
        }
    }
    struct Tile {size_t group,begin,end;};
    std::vector<Tile> tiles;
    for (size_t group=0;group<primary.groups.size();++group)
        for (size_t i=0;i<primary.groups[group].size();i+=32)
            tiles.push_back({group,i,std::min(i+32,primary.groups[group].size())});
    structured::Workers workers(std::min(nthreads,tiles.size()),{});
    workers.run(tiles.size(),[&](size_t task,size_t) {
        const auto [group,begin,end]=tiles[task];
        const auto &bra=primary.groups[group];
        Complex diagonal{};
        for (size_t p=0;p<k;++p) diagonal+=double((*bra.cfg)[p])*integrals[p*k+p];
        for (size_t i=begin;i<end;++i) {
            const size_t row=public_row(bra.row(i));
            for (size_t c=0;c<columns;++c)
                y[row*columns+c]=row<source_size ? diagonal*input[row*columns+c] : Complex{};
        }
        Config source=*bra.cfg;
        unsigned allowed=0;
        for(auto remaining=sectors;remaining;remaining&=remaining-1)
            allowed|=1u<<(bra.block->irrep^unsigned(__builtin_ctz(remaining)));
        for (size_t p=0;p<k;++p) {
            if (!source[p]) continue;
            --source[p];
            transfers.visit(p,allowed,[&](size_t q) {
                const auto integral=integrals[p*k+q];
                if(source[q]==2)return;
                ++source[q];
                const size_t id=primary.native->find(Occupation(source));
                --source[q];
                if(id==no_row)return;
                const auto &ket=primary.groups[id];
                for (size_t i=begin;i<end;++i) {
                    const size_t bra_row=bra.row(i), out=public_row(bra_row)*columns;
                    for (size_t j=0;j<ket.size();++j) {
                        const size_t ket_row=ket.row(j), in=public_row(ket_row);
                        if (in>=source_size) continue;
                        const auto value=integral*guga_generator(paths[bra_row],paths[ket_row],p,q);
                        for (size_t c=0;c<columns;++c) y[out+c]+=value*input[in*columns+c];
                    }
                }
            });
            ++source[p];
        }
    });
    return output;
}

py::tuple select_one_electron(const ComplexArray &one, std::shared_ptr<const Basis> basis,
        const Array &weights, double eps, size_t nthreads) {
    validate(one,basis,nthreads);
    if (weights.ndim()!=1 || size_t(weights.size())!=basis->size() || !std::isfinite(eps) || eps<0)
        throw std::invalid_argument("invalid one-electron selection coefficients or threshold");
    for (py::ssize_t i=0;i<weights.size();++i)
        if (!std::isfinite(weights.data()[i]) || weights.data()[i]<0)
            throw std::invalid_argument("one-electron selection weights must be finite and nonnegative");
    std::set<std::string> accepted;
    Stats stats{{"matrix_elements",0},{"configuration_pairs",0}};
    {
        py::gil_scoped_release release;
        const structured::Primary primary(std::move(basis));
        const auto paths=prepare_paths(primary);
        const auto indices=public_indices(*primary.native);
        const auto *integrals=one.data();
        const PropertyTransfers transfers(primary,integrals,false);
        const auto *coefficients=weights.data();
        auto weight=[&](size_t row) {return coefficients[indices.empty() ? row : indices[row]];};
        const size_t count=std::min(nthreads,std::max(size_t(1),primary.groups.size()));
        structured::Workers workers(count,{});
        std::vector<std::set<std::string>> additions(count);
        std::vector<Stats> counters(count);
        workers.run(primary.groups.size(),[&](size_t job,size_t worker) {
            const auto &source=primary.groups[job];
            std::vector<Path> active_paths;
            std::vector<double> active_weights;
            for (size_t i=0;i<source.size();++i) {
                const double w=weight(source.row(i));
                if (w==0.) continue;
                active_paths.push_back(paths[source.row(i)]);
                active_weights.push_back(w);
            }
            if (active_paths.empty()) return;
            auto &local=additions[worker];
            auto &census=counters[worker];
            Config target=*source.cfg;
            const int target_irrep=primary.native->target_irrep();
            const unsigned allowed=target_irrep<0 ? (1u<<OrbitalSymmetry::extent)-1
                : 1u<<(unsigned(target_irrep)^source.block->irrep);
            for (int q=0;q<primary.k;++q) {
                if (!target[q]) continue;
                --target[q];
                transfers.visit(q,allowed,[&](int p) {
                    const auto integral=integrals[p*primary.k+q];
                    if(target[p]==2)return;
                    ++target[p];
                    ++census["configuration_pairs"];
                    const auto key=Occupation(target);
                    const size_t id=primary.native->find(key);
                    const auto *block=id==no_row ? nullptr : &primary.native->blocks()[id];
                    if (!block || !block->complete()) {
                        auto evaluate=[&](const std::string &label) {
                            if (local.count(label) || (block &&
                                block->contains(primary.native->spin(block->open).rank(label)))) return;
                            const Path bra(label,primary.k,primary.s,2);
                            for (size_t i=0;i<active_paths.size();++i) {
                                ++census["matrix_elements"];
                                if (std::abs(integral*guga_generator(bra,active_paths[i],p,q))*active_weights[i]>eps) {
                                    local.insert(label);
                                    break;
                                }
                            }
                        };
                        Connection channel;
                        channel.generators.push_back({1.,p,q,0,0,false});
                        const bool connected=key.singles()>=12 && guga_connected_paths(target,active_paths,channel,
                            4*1024*1024/count,evaluate,census);
                        if (!connected) {
                            const SpinRanks ranks(key.singles(),primary.s);
                            for (size_t rank=0;rank<ranks.size();++rank) evaluate(ranks.label(target,rank));
                        }
                    }
                    --target[p];
                });
                ++target[q];
            }
        });
        for (auto &local:additions) accepted.merge(local);
        for (const auto &local:counters) for (const auto &[name,value]:local) stats[name]+=value;
        stats["workers"]=count;
        stats["added"]=accepted.size();
    }
    return py::make_tuple(std::vector<std::string>(accepted.begin(),accepted.end()),stats);
}
} // namespace

void bind_one_electron(py::module_ &m) {
    m.def("one_electron_product",&one_electron_product);
    m.def("select_one_electron",&select_one_electron);
}
} // namespace scsf

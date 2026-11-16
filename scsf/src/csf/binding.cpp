/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */
#include "csf.h"
#include "selected_basis.h"
#include "selected_index.h"
#include <pybind11/stl.h>
#include <chrono>

namespace py = pybind11;
PYBIND11_MODULE(_csf, m) {
    m.doc() = "SCSF native CSF couplings, selection, CSR construction, and native CSR products";
    py::register_exception<scsf::BudgetError>(m, "BudgetError", PyExc_MemoryError);
    m.def("hamiltonian_block", &scsf::block, py::arg("one_mo"), py::arg("two_mo"),
          py::arg("rows"), py::arg("cols"), py::arg("two_s"),
          py::arg("max_bond_elements") = 1000000,
          py::arg("max_tensor_entries") = 1000000,
          py::arg("max_connection_terms") = 1000000, py::arg("cache") = nullptr,
          py::arg("compiled") = true, py::arg("guga") = false, py::arg("guga_mode") = 0);
    scsf::bind_selected_basis(m);
    // Diagnostic for exact neighbor checks, including deliberately constant
    // fingerprints. Production callers always use the full 64-bit hash.
    m.def("_configuration_index_probe", [](scsf::BasisSnapshot basis,int mode,size_t maximum,
                                           unsigned bits,bool return_partners,size_t repeats) {
        using namespace scsf;
        checked_basis(basis);
        Stats stats;std::vector<std::vector<size_t>> result;
        double setup,query;
        {
            py::gil_scoped_release release;
            auto start=std::chrono::steady_clock::now();
            ConfigurationIndex<SelectedBasis::Groups> index(basis->groups(),mode,maximum,bits);
            setup=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            stats["mode"]=index.mode();stats["bytes"]=index.bytes();stats["entries"]=index.entries();
            start=std::chrono::steady_clock::now();
            for (size_t repeat=0;repeat<repeats;++repeat) for (const auto *g:basis->order()) {
                auto found=index.query(g->first,std::numeric_limits<size_t>::max(),stats);
                stats["neighbors"]+=found.size();
                if (return_partners) {
                    result.emplace_back();
                    for (auto *p:found) result.back().push_back(basis->ordinals().at(p));
                    std::sort(result.back().begin(),result.back().end());
                }
            }
            query=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        }
        return py::make_tuple(result,stats,setup,query);
    },py::arg("basis"),py::arg("mode"),py::arg("max_bytes"),py::arg("hash_bits")=64,
       py::arg("return_partners")=true,py::arg("repeats")=1);
    scsf::bind_sci(m);
    scsf::bind_guga(m);
    scsf::bind_walks(m);
    scsf::bind_products(m);
}

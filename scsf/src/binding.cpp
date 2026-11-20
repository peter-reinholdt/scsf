/* SCSF; GPL-3.0-or-later. */
#include "csf.h"

namespace py = pybind11;
namespace scsf { void bind_structured(py::module_ &); }

PYBIND11_MODULE(_csf, m) {
    m.doc() = "SCSF native occupation-block Hamiltonian and property products";
    py::register_exception<scsf::BudgetError>(m, "BudgetError", PyExc_MemoryError);
    py::class_<scsf::HCITransferBounds, std::shared_ptr<scsf::HCITransferBounds>>(m, "_HCITransferBounds")
        .def(py::init<const scsf::Array &, const scsf::Array &, size_t>());
    scsf::bind_one_electron(m);
    scsf::bind_structured(m);
}

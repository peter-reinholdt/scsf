/* This file is part of SCSF, licensed under GNU GPL version 3 or later. */
#pragma once
#include "csf.h"
#include <limits>

namespace scsf {
using Groups = std::map<Config, std::vector<size_t>>;
using Stats = std::map<std::string, size_t>;

// Legacy append/rectangular selection clients use owning group lists. Direct
// operators use SelectedBasis::Groups, whose views share one flat allocation.
Groups groups(const std::vector<std::string> &, int k, int two_s, size_t start, size_t end);
void validate_config(const Config &);

struct BasisContext {
    int nbasis, nelec, two_s;
    size_t generation;
    std::uint64_t snapshot_id;
    static constexpr const char *orbital_order = "input";
    static constexpr const char *phase_convention = "cg-real-alpha-beta-v1";
    static constexpr const char *symmetry_convention = "none";
};

class SelectedBasis {
public:
    using Groups = std::map<Config, IndexView>;
    using Group = Groups::value_type;
private:
    BasisContext context_;
    std::vector<std::string> labels_;
    // Open-shell down steps, packed consecutively within each configuration.
    // No per-CSF pointer/offset is needed: local rank and open-shell count give
    // the exact bit offset. This is optional and replaces retained labels.
    bool packed_ = false;
    size_t ncsf_ = 0;
    std::vector<size_t> local_ranks_, spin_offsets_;
    std::vector<std::uint64_t> spin_words_;
    std::vector<size_t> indices_, configuration_ids_, offsets_;
    Groups groups_;
    std::vector<const Group *> order_;
    std::unordered_map<const Group *,size_t> ordinal_;
public:
    SelectedBasis(int nbasis, int nelec, int two_s,
                  std::vector<std::string> labels, size_t generation = 0, bool packed = false);
    SelectedBasis(const SelectedBasis &) = delete;
    SelectedBasis &operator=(const SelectedBasis &) = delete;
    const BasisContext &context() const { return context_; }
    size_t size() const { return ncsf_; }
    bool packed() const { return packed_; }
    std::vector<std::string> labels() const;
    const std::string &label(size_t csf, std::string &scratch) const;
    const auto &groups() const { return groups_; }
    const auto &order() const { return order_; }
    const auto &ordinals() const { return ordinal_; }
    const auto &indices() const { return indices_; }
    const auto &offsets() const { return offsets_; }
    const auto &configuration_ids() const { return configuration_ids_; }
    IndexView local_indices(size_t id) const { return order_.at(id)->second; }
    std::string spin_path(size_t csf) const;
    size_t bytes_estimate() const;
    Stats storage_info() const;
};
using BasisSnapshot = std::shared_ptr<const SelectedBasis>;
inline BasisSnapshot checked_basis(BasisSnapshot basis) {
    if (!basis) throw std::invalid_argument("a selected basis snapshot is required");
    return basis;
}
void bind_selected_basis(pybind11::module_ &);
} // namespace scsf

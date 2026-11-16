/* This file is part of SCSF, licensed under GNU GPL version 3 or later. */
#pragma once
#include "index_view.h"
#include <utility>

namespace scsf {
// Native block-application contract. Evaluators perform coupling generation /
// integral contraction; the driver owns preparation, vector/error buffers,
// screening and scheduling. All callbacks are statically dispatched and can
// inline. A future backend may supply a tiled block implementation instead.
// r/t always denote the original row/column lists, even in source traversal.
struct SelectedBlock {
    IndexView rows, columns;
    bool transpose = false;
    bool triangular = false;
    bool include_diagonal = false;
};

template<class Evaluate> class EntryCouplingBackend {
    Evaluate evaluate_;
    template<class Accept, class Publish>
    void apply(const SelectedBlock &block, Accept accept, Publish publish) {
        for (size_t r=0;r<block.rows.size();++r)
            for (size_t t=0;t<block.columns.size();++t) {
                size_t i=block.transpose ? block.columns[t] : block.rows[r];
                size_t j=block.transpose ? block.rows[r] : block.columns[t];
                if ((!block.include_diagonal && i==j) || (block.triangular && j>i)) continue;
                if (accept(r,t,i,j)) publish(i,j,evaluate_pair_entry(r,t,i,j));
            }
    }
public:
    explicit EntryCouplingBackend(Evaluate evaluate) : evaluate_(std::move(evaluate)) {}
    double evaluate_pair_entry(size_t r, size_t t, size_t i, size_t j) {
        return evaluate_(r,t,i,j);
    }
    template<class Accept, class Publish>
    void apply_same_configuration(const SelectedBlock &block, Accept accept, Publish publish) {
        apply(block,accept,publish);
    }
    template<class Accept, class Publish>
    void apply_configuration_pair(const SelectedBlock &block, Accept accept, Publish publish) {
        apply(block,accept,publish);
    }
};
template<class Evaluate> EntryCouplingBackend<Evaluate> entry_backend(Evaluate evaluate) {
    return EntryCouplingBackend<Evaluate>(std::move(evaluate));
}
template<class Backend, class Accept, class Publish>
void apply_selected_block(Backend &backend, bool same_configuration,
                          const SelectedBlock &block, Accept accept, Publish publish) {
    if (same_configuration) backend.apply_same_configuration(block,accept,publish);
    else backend.apply_configuration_pair(block,accept,publish);
}
} // namespace scsf

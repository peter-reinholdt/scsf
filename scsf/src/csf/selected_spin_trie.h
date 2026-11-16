/* This file is part of SCSF, licensed under GNU GPL version 3 or later. */
#pragma once
#include "csf.h"
#include <limits>

namespace scsf {
// One binary node per selected open-shell prefix; closed sites do not branch.
// Leaves are local selected indices, never a complete spin-class enumeration.
struct SelectedSpinTrie {
    static constexpr std::uint32_t none=std::numeric_limits<std::uint32_t>::max();
    struct Node {
        std::array<std::uint32_t,2> child{none,none};
        std::uint32_t leaf=none, first=none;
        int spin=0;
    };
    std::vector<int> sites;
    std::vector<Node> nodes;
    size_t bytes() const {
        return sizeof(*this)+sites.capacity()*sizeof(int)+nodes.capacity()*sizeof(Node);
    }
    static std::unique_ptr<SelectedSpinTrie> build(const Config &,IndexView,
        const std::vector<Path> &,size_t maximum);

};

struct SpinTraversalStats {
    size_t blocks=0, nodes=0, pruned=0, leaves=0, channels=0, scratch_peak=0;
    size_t hybrid_short_blocks=0,hybrid_split_blocks=0,hybrid_short_terms=0,hybrid_long_terms=0;
    size_t weighted_blocks=0,weighted_nodes=0,weighted_hits=0,weighted_pruned=0,weighted_channels=0;
    size_t weighted_buffer_peak=0;
};
struct WeightedSpinScratch {
    std::vector<double> stack,cache;
    std::vector<const Path *> previous;
    std::vector<unsigned char> live;
    size_t bytes() const { return (stack.capacity()+cache.capacity())*sizeof(double)+
        previous.capacity()*sizeof(const Path *)+live.capacity(); }
};
struct HybridSpinScratch {
    Connection short_terms,long_terms;
    GugaLeanPlan short_plan;
    GugaBatchScratch values;
    size_t bytes() const { return (short_terms.generators.capacity()+long_terms.generators.capacity())*sizeof(GugaTerm)+
        short_plan.capacity()*sizeof(GugaLeanTerm)+values.bytes(); }
};
bool selected_guga_batch(const Config &,const Config &,const SelectedSpinTrie &,
    const std::vector<Path> &,IndexView,IndexView,const std::vector<double> &,
    bool,const Connection &,size_t,GugaBatchScratch &,SpinTraversalStats &);

bool weighted_guga_apply(const Config &,const Config &,const SelectedSpinTrie &,const SelectedSpinTrie &,
    const std::vector<Path> &,IndexView,IndexView,const std::vector<double> &,bool,const Connection &,
    const double *,double *,size_t,size_t,WeightedSpinScratch &,SpinTraversalStats &);
// 0: use the scalar fallback; 1: a complete temporary rectangle.
int hybrid_guga_apply(const Config &,const Config &,const SelectedSpinTrie &,
    const std::vector<Path> &,IndexView,IndexView,const std::vector<std::uint64_t> &,
    const std::vector<double> &,bool,const Connection &,const GugaLeanPlan &,GugaMemo &,bool,
    int,size_t,GugaBatchScratch &,HybridSpinScratch &,SpinTraversalStats &);
} // namespace scsf

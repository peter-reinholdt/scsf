/* This file is part of SCSF, licensed under GNU GPL version 3 or later. */
#include "selected_spin_trie.h"
#include <algorithm>

namespace scsf {
std::unique_ptr<SelectedSpinTrie> SelectedSpinTrie::build(const Config &cfg,IndexView indices,
        const std::vector<Path> &paths,size_t maximum) {
    if (indices.size()>none || maximum<sizeof(SelectedSpinTrie)+sizeof(Node)) return {};
    auto t=std::make_unique<SelectedSpinTrie>();
    size_t singles=std::count(cfg.begin(),cfg.end(),1);
    if (singles>(maximum-sizeof(*t)-sizeof(Node))/sizeof(int)) return {};
    t->sites.reserve(singles);
    for (size_t p=0;p<cfg.size();++p) if (cfg[p]==1) t->sites.push_back(p);
    size_t fixed=t->bytes(),limit=std::min(size_t(none),(maximum-fixed)/sizeof(Node));
    if (!limit) return {};
    t->nodes.reserve(std::min(size_t(8),limit));t->nodes.emplace_back();
    for (size_t local=0;local<indices.size();++local) {
        const auto &label=paths[indices[local]].label;
        std::uint32_t at=0;
        t->nodes[0].first=std::min(t->nodes[0].first,std::uint32_t(local));
        for (size_t d=0;d<singles;++d) {
            bool down=label[t->sites[d]]=='d';
            auto next=t->nodes[at].child[down];
            if (next==none) {
                if (t->nodes.size()==limit) return {};
                if (t->nodes.size()==t->nodes.capacity())
                    t->nodes.reserve(std::min(limit,2*t->nodes.capacity()));
                Node node;node.spin=t->nodes[at].spin+(down ? -1 : 1);
                next=t->nodes.size();t->nodes[at].child[down]=next;t->nodes.push_back(node);
            }
            at=next;auto &node=t->nodes[at];
            node.first=std::min(node.first,std::uint32_t(local));
        }
        t->nodes[at].leaf=local;
    }
    if (t->bytes()>maximum) return {};
    return t;
}


} // namespace scsf

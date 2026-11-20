/* SCSF; GPL-3.0-or-later. Strict HCI over native occupation blocks. */
#pragma once
#include "configuration_connectivity.h"
#include "structured_internal.h"
#include "structured_eri.h"

namespace scsf {

inline bool screened(double bound,double weight,double eps,int k) {
    if (bound==0.) return true;
    const long double product=static_cast<long double>(bound)*weight;
    const long double margin=512.L*std::numeric_limits<double>::epsilon()*(k+1)*(product+eps);
    return product+margin<eps;
}

inline std::pair<Basis::Additions,Stats> hci(const structured::Primary &p,structured::Workers &workers,
        const double *h,const structured::SpatialERI &eri,const double *weights,double eps,bool complete,
        HCITransferBounds *bounds) {
    const auto &basis=*p.native;
    const auto start=structured::Clock::now();
    if (bounds) bounds->prepare_transfer_bounds(p.k);
    const bool ordered=bounds && bounds->prepare_ordered_transfers(p.k);
    struct Shard {
        std::mutex mutex;
        std::atomic<bool> ready{false};
        struct Hash { size_t operator()(const Occupation &key) const { return key.hash(); } };
        // As in Basis::Additions, an empty rank vector selects the whole configuration.
        std::unordered_map<Occupation,std::vector<size_t>,Hash> targets;
    };
    std::array<Shard,64> shards;

    std::vector<std::unique_ptr<SpinRanks>> ranks(basis.maximum_open()+1);
    std::vector<std::pair<double,size_t>> sources;
    sources.reserve(p.groups.size());
    for (size_t id=0;id<p.groups.size();++id) {
        const auto &block=basis.blocks()[id];
        double maximum=0.;
        for (size_t i=0;i<block.size();++i) maximum=std::max(maximum,weights[block.offset+i]);
        if (maximum==0.) continue;
        sources.emplace_back(maximum,id);
        for (int m=std::max(p.s,block.open-4);m<=std::min(basis.maximum_open(),block.open+4);m+=2)
            if (!ranks[m]) ranks[m]=std::make_unique<SpinRanks>(m,p.s);
    }
    std::sort(sources.begin(),sources.end(),[](const auto &a,const auto &b) {
        return a.first>b.first || (a.first==b.first && a.second<b.second);
    });

    struct Counts {size_t visits=0, screened=0, elements=0, candidates=0;};
    std::vector<Counts> counts(workers.count());
    workers.run(sources.size(),[&](size_t source,size_t worker) {
        auto [maximum,id]=sources[source];
        auto &stats=counts[worker];
        const auto &ket=*p.groups[id].cfg;
        const auto &block=basis.blocks()[id];
        std::vector<std::pair<double,Path>> parents;
        parents.reserve(block.size());
        for (size_t i=0;i<block.size();++i) {
            const double weight=weights[block.offset+i];
            if (weight==0.) continue;
            parents.emplace_back(weight,Path(basis.spin(block.open).label(ket,block.rank(i)),p.k,p.s,2));
        }
        std::stable_sort(parents.begin(),parents.end(),[](const auto &a,const auto &b){return a.first>b.first;});
        Connection terms;
        auto visit=[&](const std::array<int,2> &holes,const std::array<int,2> &particles,int order) {
            ++stats.visits;
            if (bounds && !(ordered && order==2)) {
                const double bound=bounds->selection_bound(ket,holes,particles,order,p.k);
                if (screened(bound,maximum,eps,p.k)) {++stats.screened;return;}
            }
            Occupation key=block.key;
            key.transfer(holes,particles,order);
            const auto &spin=ranks[key.singles()];
            if (!spin || !spin->size()) return;
            const size_t hash=key.hash();
            const size_t other=basis.find(key,hash);
            const auto *internal=other==no_row ? nullptr : &basis.blocks()[other];
            if (internal && internal->complete()) return;

            auto &shard=shards[hash%shards.size()];
            std::vector<size_t> already;
            const size_t missing=spin->size()-(internal ? internal->size() : 0);
            if(shard.ready.load(std::memory_order_acquire)) {
                std::lock_guard<std::mutex> lock(shard.mutex);
                auto it=shard.targets.find(key);
                if(it!=shard.targets.end()) {
                    if(complete || it->second.size()==missing)return;
                    already=it->second;
                }
            }
            stats.candidates+=missing-already.size();
            if (complete && internal)
                throw std::logic_error("configuration HCI requires a complete source space");

            const auto bra=endpoint(ket,holes,particles,order);
            PairDescriptor pair(bra,ket,0,0,holes,particles,order);
            ranked_connection(pair,h,eri,true,terms);
            if(terms.generators.empty())return;
            // The prepared bound covers the *sum* of all generator paths.
            // Sort the source ranks once so one suffix rejection covers every
            // weaker parent, before any candidate path is unranked.
            auto last=parents.end();
            while(last!=parents.begin() && screened(terms.term_bound,(last-1)->first,eps,p.k))--last;
            if(last==parents.begin()) {++stats.screened;return;}
            std::vector<size_t> additions;
            for (size_t rank=0;rank<spin->size();++rank) {
                if ((internal && internal->contains(rank)) || std::binary_search(already.begin(),already.end(),rank))
                    continue;
                const Path path(spin->label(bra,rank),p.k,p.s,2);
                for(auto candidate_parent=parents.begin();candidate_parent!=last;++candidate_parent) {
                    const auto &[weight,parent]=*candidate_parent;
                    ++stats.elements;
                    if (!(std::abs(guga_element(path,parent,terms)*weight)>eps)) continue;
                    additions.push_back(rank);
                    break;
                }
                if (complete && !additions.empty()) break;
            }
            if (additions.empty()) return;
            std::lock_guard<std::mutex> lock(shard.mutex);
            auto &target=shard.targets.try_emplace(std::move(key)).first->second;
            shard.ready.store(true,std::memory_order_release);
            if (complete) return;
            std::vector<size_t> combined;
            std::set_union(target.begin(),target.end(),additions.begin(),additions.end(),
                           std::back_inserter(combined));
            target=std::move(combined);
        };

        if (!ordered) {
            transfers(ket,visit);
            return;
        }
        visit({}, {},0);
        for (int q=0;q<p.k;++q) {
            if (!ket[q]) continue;
            for (int a=0;a<p.k;++a) if (a!=q && ket[a]<2) visit({q,0},{a,0},1);
            for (int s=q;s<p.k;++s) {
                if (ket[s]<(q==s ? 2 : 1)) continue;
                auto [item,last]=bounds->ordered_transfers(q,s,p.k);
                for (;item!=last;++item) {
                    if (screened(item->bound,maximum,eps,p.k)) break;
                    if (ket[item->p]==2 || ket[item->r]>=(item->p==item->r ? 1 : 2)) continue;
                    visit({q,s},{item->p,item->r},2);
                }
            }
        }
    },4);
    Basis::Additions result;
    for (auto &shard:shards) for (auto &[cfg,selected]:shard.targets)
        result.emplace(cfg.decode(p.k),std::move(selected));
    Stats stats{{"ordered_search",ordered},
                {"source_configurations",sources.size()},{"selection_ns",structured::ns(start)}};
    for (const auto &c:counts) {
        stats["candidate_rank_visits"]+=c.candidates;
        stats["transfer_visits"]+=c.visits;
        stats["screened_pairs"]+=c.screened;
        stats["matrix_elements"]+=c.elements;
    }
    return {std::move(result),std::move(stats)};
}

} // namespace scsf

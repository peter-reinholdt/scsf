/* SCSF; GPL-3.0-or-later. Spin-free generator channels by intermediate occupation. */
#pragma once
#include "cih_diagonal.h"

namespace scsf::v2 {

class GeneratorPostings {
    const structured::Primary &primary_;
    structured::OneBank bank_;
    size_t allocated_=0,records_=0;

public:
    struct Channel {size_t source;int create,destroy;};
    std::map<Config,std::vector<Channel>> groups;

    GeneratorPostings(const structured::Primary &p,size_t link_bytes,size_t coefficient_bytes,
                      const double *active=nullptr)
        :primary_(p),bank_(cih_template_electrons(p),p.s,coefficient_bytes) {
        auto stash=[&](const Config &cfg,Channel c) {
            auto [it,inserted]=groups.try_emplace(cfg);
            if(inserted)allocated_=checked_add(allocated_,sizeof(decltype(groups)::value_type)
                +5*sizeof(void *)+cfg.size()*sizeof(int));
            auto &channels=it->second;
            const size_t before=channels.capacity();
            channels.push_back(c);
            allocated_=checked_add(allocated_,(channels.capacity()-before)*sizeof(Channel));
            if(allocated_>link_bytes)throw BudgetError("generator postings exceed metadata budget");
            ++records_;
        };
        for(size_t src=0;src<p.groups.size();++src) {
            const auto &group=p.groups[src];
            const auto &ket=*group.cfg;
            if(active) {
                const double *values=active+group.block->offset;
                if(std::none_of(values,values+group.size(),[](double c){return c!=0.;}))continue;
            }
            Config mid=ket;
            for(int q=0;q<p.k;++q)if(ket[q]) {
                stash(ket,{src,q,q});
                --mid[q];
                for(int r=0;r<p.k;++r)if(r!=q && ket[r]<2) {
                    if(bank_.get(ket,r,q).data.empty())continue;
                    ++mid[r];
                    stash(mid,{src,r,q});
                    --mid[r];
                }
                ++mid[q];
            }
        }
    }

    size_t rows(const Channel &c) const {
        const auto &source=primary_.groups[c.source];
        if(c.create==c.destroy)return source.block->dimension;
        return bank_.get(*source.cfg,c.create,c.destroy).rows;
    }

    template<bool Adjoint>
    void contract(const Channel &c,const double *x,double *y) const {
        const auto &source=primary_.groups[c.source];
        auto add=[&](size_t row,size_t local,double weight) {
            const size_t index=source.row(local);
            if constexpr(Adjoint)y[index]+=weight*x[row];
            else y[row]+=weight*x[index];
        };
        if(c.create==c.destroy) {
            for(size_t j=0;j<source.size();++j)add(source.rank(j),j,(*source.cfg)[c.create]);
        } else {
            const auto &matrix=bank_.get(*source.cfg,c.create,c.destroy);
            for(size_t row=0;row<matrix.rows;++row)
                for(size_t j=0;j<source.size();++j)add(row,j,matrix.at(row,source.rank(j)));
        }
    }

    Stats info() const {
        return {{"intermediate_occupations",groups.size()},{"generator_channels",records_},
                {"posting_bytes",allocated_},{"coefficient_bytes",bank_.bytes()}};
    }
};
} // namespace scsf::v2

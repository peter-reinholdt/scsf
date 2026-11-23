/* SCSF; GPL-3.0-or-later. Spin-free generator channels by intermediate occupation. */
#pragma once
#include "cih_diagonal.h"

namespace scsf::structured {

class GeneratorPostings {
    const Primary &primary_;
    OneBank bank_;
    std::vector<unsigned char> support_;
    size_t allocated_=0,records_=0;

    const Matrix &coupling(const Primary::Group &group,int create,int destroy) const {
        const auto &cfg=*group.cfg;
        const auto &key=group.block->key;
        const int pi=key.singles_before(create)+(destroy<create && cfg[destroy]!=1);
        const int qi=key.singles_before(destroy)+(create<destroy && cfg[create]!=1);
        return bank_.get(group.block->open,cfg[create],cfg[destroy],pi,qi);
    }

public:
    struct Channel {size_t source;int create,destroy;};
    enum class Order {natural,directed,sector};
    std::map<Config,std::vector<Channel>> groups;

    GeneratorPostings(const Primary &p,size_t link_bytes,size_t coefficient_bytes,
                      const double *active=nullptr,const double *other=nullptr,size_t total_bytes=no_row,
                      Order order=Order::natural)
        :primary_(p),bank_(cih_template_electrons(p),p.s,std::min(coefficient_bytes,total_bytes)) {
        link_bytes=std::min(link_bytes,memory_left(total_bytes,bank_.bytes(),"generator coefficients"));
        if(active) {
            memory_left(link_bytes,p.groups.size(),"generator coefficient support");
            support_.resize(p.groups.size());
            allocated_=support_.capacity();
        }
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
                const auto nonzero=[&](const double *values) {
                    values+=group.block->offset;
                    return std::any_of(values,values+group.size(),[](double c){return c!=0.;});
                };
                const bool right=nonzero(active),left=other ? nonzero(other) : right;
                support_[src]=unsigned(right)+2*unsigned(left);
                if(!support_[src])continue;
            }
            Config mid=ket;
            for(int q=0;q<p.k;++q)if(ket[q]) {
                stash(ket,{src,q,q});
                --mid[q];
                for(int r=0;r<p.k;++r)if(r!=q && ket[r]<2) {
                    if(coupling(group,r,q).data.empty())continue;
                    ++mid[r];
                    stash(mid,{src,r,q});
                    --mid[r];
                }
                ++mid[q];
            }
        }
        if(order==Order::directed)for(auto &[cfg,channels]:groups)
            std::sort(channels.begin(),channels.end(),[](const Channel &a,const Channel &b) {
                return std::tie(a.destroy,a.create)<std::tie(b.destroy,b.create);
            });
        else if(order==Order::sector && p.native->symmetry().nontrivial())for(auto &[cfg,channels]:groups)
            std::sort(channels.begin(),channels.end(),[&](const Channel &a,const Channel &b) {
                return std::tie(p.groups[a.source].block->irrep,a.source,a.destroy,a.create)<
                    std::tie(p.groups[b.source].block->irrep,b.source,b.destroy,b.create);
            });
    }

    unsigned support(size_t source) const {return support_.empty() ? 3 : support_[source];}

    size_t rows(const Channel &c) const {
        const auto &source=primary_.groups[c.source];
        return c.create==c.destroy ? source.block->dimension : coupling(source,c.create,c.destroy).rows;
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
            return;
        }
        const auto &matrix=coupling(source,c.create,c.destroy);
        if(source.canonical_full()) {
            for(size_t row=0;row<matrix.rows;++row)
                for(size_t j=matrix.ptr[row];j<matrix.ptr[row+1];++j)add(row,matrix.indices[j],matrix.data[j]);
        } else {
            for(size_t row=0;row<matrix.rows;++row)
                for(size_t j=0;j<source.size();++j)add(row,j,matrix.at(row,source.rank(j)));
        }
    }

    Stats info() const {
        return {{"intermediate_occupations",groups.size()},{"generator_channels",records_},
                {"posting_bytes",allocated_},{"coefficient_bytes",bank_.bytes()}};
    }
};
} // namespace scsf::structured

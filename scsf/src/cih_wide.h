/* SCSF; GPL-3.0-or-later. Grouped CIH with unrestricted orbital addresses. */
#pragma once
#include "cih_diagonal.h"

namespace scsf::structured {

class WideCIH {
    struct Posting {size_t group;int orbital;};
    struct Scratch {
        std::vector<double> forward,backward;
        size_t groups=0,links=0;
    };
    const Primary &p_;
    Workers &workers_;
    SpatialERI eri_;
    OneBank bank_;
    std::vector<double> one_;
    std::map<Config,std::vector<Posting>> skeletons_;
    std::vector<Scratch> scratch_;
    size_t metadata_bytes_=0,postings_=0;

    void single(size_t src,size_t dst,const Matrix &m,double value,const double *x,size_t width,double *y) const {
        const auto &ket=p_.groups[src],&bra=p_.groups[dst];
        for(size_t i=0;i<bra.size();++i)
            for(size_t j=0;j<ket.size();++j) {
                const double c=value*m.at(bra.rank(i),ket.rank(j));
                for(size_t v=0;v<width;++v) {
                    y[bra.row(i)*width+v]+=c*x[ket.row(j)*width+v];
                    y[ket.row(j)*width+v]+=c*x[bra.row(i)*width+v];
                }
            }
    }

    void source(size_t src,const double *x,size_t width,double *y,Scratch &scratch) const {
        const auto &group=p_.groups[src];
        const auto &ket=*group.cfg;
        const int k=p_.k;
        double constant=0.;
        for(int p=0;p<k;++p)if(ket[p]) {
            constant+=one_[size_t(p)*k+p]*ket[p];
            for(int q=0;q<k;++q)if(ket[q])constant+=.5*eri_(p,p,q,q)*ket[p]*ket[q];
        }
        for(size_t i=0;i<group.size();++i)
            for(size_t v=0;v<width;++v)y[group.row(i)*width+v]+=constant*x[group.row(i)*width+v];
        Config removed=ket,mid=ket,residue=ket;
        for(int s=0;s<k;++s)if(ket[s]) {
            removed=ket;
            --removed[s];
            const auto &single_targets=skeletons_.at(removed);
            for(const auto &post:single_targets) {
                if(post.group<=src || post.orbital==s)continue;
                const int r=post.orbital;
                const size_t rs=size_t(s)*k+r;
                const auto &target=*p_.groups[post.group].cfg;
                double value=one_[size_t(r)*k+s];
                for(int q=0;q<k;++q) {
                    const int n=(size_t(q)*(k+1)<rs ? target : ket)[q];
                    if(n)value+=n*eri_(r,s,q,q);
                }
                if(value!=0.)single(src,post.group,bank_.get(ket,r,s),value,x,width,y);
            }

            for(int r=0;r<k;++r)if(r!=s && ket[r]<2) {
                const Matrix &first=bank_.get(ket,r,s);
                if(!first.rows)continue;
                mid=removed;
                ++mid[r];
                bool active=false;
                const size_t rs=size_t(s)*k+r;
                for(int q=0;q<=s;++q)if(mid[q]) {
                    residue=mid;
                    --residue[q];
                    const auto match=skeletons_.find(residue);
                    if(match==skeletons_.end())continue;
                    for(const auto &post:match->second) {
                        const int p=post.orbital;
                        const size_t pq=size_t(q)*k+p;
                        if(post.group<src || p==q || pq>rs)continue;
                        double value=eri_(p,q,r,s);
                        if(value==0.)continue;
                        if(!active) {
                            active=true;
                            ++scratch.groups;
                            scratch.forward.assign(first.rows*width,0.);
                            scratch.backward.assign(first.rows*width,0.);
                            for(size_t i=0;i<first.rows;++i)
                                for(size_t j=0;j<group.size();++j)
                                    for(size_t v=0;v<width;++v)
                                        scratch.forward[i*width+v]+=first.at(i,group.rank(j))*x[group.row(j)*width+v];
                        }
                        ++scratch.links;
                        if(pq==rs)value*=.5;
                        const auto &target=p_.groups[post.group];
                        const auto &second=bank_.get(mid,p,q);
                        for(size_t i=0;i<target.size();++i)
                            for(size_t j=0;j<first.rows;++j) {
                                const double m=value*second.at(target.rank(i),j);
                                for(size_t v=0;v<width;++v) {
                                    const size_t t=target.row(i)*width+v;
                                    y[t]+=m*scratch.forward[j*width+v];
                                    if(post.group!=src)scratch.backward[j*width+v]+=m*x[t];
                                }
                            }
                    }
                }
                if(active)for(size_t i=0;i<first.rows;++i)
                    for(size_t j=0;j<group.size();++j)
                        for(size_t v=0;v<width;++v)
                            y[group.row(j)*width+v]+=first.at(i,group.rank(j))*scratch.backward[i*width+v];
            }
        }
    }

public:
    WideCIH(const Primary &p,const double *h,SpatialERI eri,
            Workers &workers,size_t budget)
        : p_(p),workers_(workers),eri_(eri),bank_(cih_template_electrons(p),p.s,budget,true),scratch_(workers.count()) {
        const size_t k=p.k;
        const size_t metadata=memory_left(budget,bank_.bytes(),"wide CIH spin coefficients");
        if(k && k>metadata/sizeof(double)/k)throw BudgetError("wide CIH one-electron table exceeds metadata budget");
        one_.resize(k*k);
        metadata_bytes_=one_.capacity()*sizeof(double);
        for(size_t g=0;g<p.groups.size();++g) {
            Config cfg=*p.groups[g].cfg;
            for(size_t q=0;q<k;++q)if(cfg[q]) {
                --cfg[q];
                auto [slot,inserted]=skeletons_.try_emplace(cfg);
                if(inserted)metadata_bytes_=checked_add(metadata_bytes_,sizeof(decltype(skeletons_)::value_type)
                    +5*sizeof(void *)+k*sizeof(int));
                auto &range=slot->second;
                const auto capacity=range.capacity();
                range.push_back({g,static_cast<int>(q)});
                metadata_bytes_=checked_add(metadata_bytes_,(range.capacity()-capacity)*sizeof(Posting));
                if(metadata_bytes_>metadata)throw BudgetError("wide CIH skeletons exceed metadata budget");
                ++cfg[q];
                ++postings_;
            }
        }
        for(size_t q=0;q<k;++q)for(size_t p0=0;p0<k;++p0) {
            double value=h[p0*k+q];
            for(size_t l=0;l<k;++l) {
                const size_t left=l*k+p0,right=q*k+l;
                const double v=eri_(p0,l,l,q);
                value-=.5*v;
                if(left<right)value-=.5*v;
                if(left>right)value+=.5*v;
            }
            one_[p0*k+q]=value;
        }
    }

    void diagonal(size_t group,const double *h,const double *coulomb,const double *exchange,double *result) const {
        cih_diagonal(p_.groups[group],bank_,h,coulomb,exchange,p_.k,result);
    }

    size_t entries() const {return workers_.count()>1 ? workers_.count()*p_.size() : 0;}

    Stats info() const {
        return {{"metadata_bytes",metadata_bytes_},{"coefficient_bytes",bank_.bytes()},
                {"wide_grouped_uga",1},{"residue_links",postings_},{"plan_cache_bytes",0},
                {"workspace_entries_per_rhs",entries()}};
    }

    void apply(const double *x,size_t width,double *y,Stats &stats) {
        const size_t stride=p_.size()*width;
        const bool reduce=workers_.count()>1;
        std::vector<double> private_output(reduce ? workers_.count()*stride : 0,0.);
        for(auto &scratch:scratch_)scratch.groups=scratch.links=0;
        auto run=[&](size_t g,size_t w) {
            source(g,x,width,reduce ? private_output.data()+w*stride : y,scratch_[w]);
        };
        workers_.run(p_.groups.size(),run,4);
        if(reduce)workers_.run(p_.size(),[&](size_t i,size_t) {
            for(size_t w=0;w<workers_.count();++w)
                for(size_t v=0;v<width;++v)y[i*width+v]+=private_output[w*stride+i*width+v];
        },128);
        for(const auto &scratch:scratch_) {
            stats["generator_groups"]+=scratch.groups;
            stats["generator_links"]+=scratch.links;
            stats["worker_temporary_bytes"]+=(scratch.forward.capacity()+scratch.backward.capacity())*sizeof(double);
        }
    }
};
} // namespace scsf::structured

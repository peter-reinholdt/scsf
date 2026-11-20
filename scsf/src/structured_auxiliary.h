/* SCSF; GPL-3.0-or-later. Complete number-changing auxiliary-space products. */
#pragma once
#include "structured_internal.h"
#include "structured_eri.h"

namespace scsf { namespace structured {

class Auxiliary {
    const Primary &p_;
    const double *h_;
    SpatialERI integrals_;
    int mode_;
    bool private_output_;
    Workers &workers_;
    struct Topology {
        RemovalBank bank;
        std::unique_ptr<RemovalLinks> one,two;
        std::unique_ptr<Residues> auxiliary_queries;
        std::map<int,std::unique_ptr<OneBank>> one_banks;
        std::vector<const OneBank *> family_banks;
        std::vector<std::pair<int,int>> orbital_pairs;
        std::vector<double> pair_integrals;
        std::vector<size_t> pair_rows,pair_columns;
        std::vector<double> pair_features;
        size_t feature_count=0,metadata_bytes=0,build_bytes=0,one_configs=0,two_configs=0;
        Topology(const Primary &primary,size_t coefficients):bank(primary,coefficients/2){}
    };
    std::shared_ptr<Topology> topology_;
    std::vector<std::vector<double>> temporary_,mixed_,projected_;
    bool reused_topology_=false;
    size_t incremental_groups_=0,overlap_bytes_=0;

    static bool append_prefix(const Primary &p,const Auxiliary *previous) {
        if(!previous || previous->p_.groups.size()>p.groups.size())return false;
        const auto &old=previous->p_.native->blocks(),&next=p.native->blocks();
        for(size_t g=0;g<old.size();++g)if(!(old[g].key==next[g].key))return false;
        return true;
    }
    double eri(int p,int q,int r,int s) const {
        return integrals_(p,r,q,s);
    }
    double pair_value(size_t a,size_t b,const SpatialERI &integrals) const {
        auto x=topology_->orbital_pairs[a],y=topology_->orbital_pairs[b];
        int p=x.first%p_.k,q=x.second%p_.k,r=y.first%p_.k,s=y.second%p_.k;
        if(p_.s==0 && x.first<p_.k && y.first<p_.k) {
            // Fold the equal, omitted alpha-beta orientation at both ends.
            double direct=integrals(p,r,q,s);
            if(p==q && r==s)return direct;
            if(p==q || r==s)return 2*direct;
            return 2*(direct+integrals(p,s,q,r));
        }
        return ((x.first/p_.k==y.first/p_.k && x.second/p_.k==y.second/p_.k) ? integrals(p,r,q,s) : 0.)-
               ((x.first/p_.k==y.second/p_.k && x.second/p_.k==y.first/p_.k) ? integrals(p,s,q,r) : 0.);
    }
    bool same_spin(size_t channel) const {
        auto pair=topology_->orbital_pairs[channel];
        return pair.first/p_.k==pair.second/p_.k;
    }
    void prepare_pair_integrals(size_t bytes,size_t construction_bytes) {
        const size_t m=topology_->orbital_pairs.size();
        if(!m)return;
        if(integrals_.format==ERIFormat::thc && integrals_.rank) {
            const size_t rank=integrals_.rank;
            const size_t average=topology_->two->parents.size()/topology_->two->blocks.size();
            // The pair contraction uses two applications of the Khatri–Rao
            // features. Only keep them if they reduce the typical block width.
            if(rank<=average/2/rank && rank<=bytes/8/m/rank) {
                topology_->feature_count=rank*rank;
                topology_->pair_features.resize(m*topology_->feature_count);
                workers_.run(m,[&](size_t a,size_t) {
                    auto [i,j]=topology_->orbital_pairs[a];
                    int p=i%p_.k,q=j%p_.k;
                    int exchange=same_spin(a) ? -1 : p_.s==0 && p!=q ? 1 : 0;
                    const double *x=integrals_.data+p*rank,*y=integrals_.data+q*rank;
                    double *row=topology_->pair_features.data()+a*topology_->feature_count;
                    for(size_t u=0;u<rank;++u)for(size_t v=0;v<rank;++v)
                        row[u*rank+v]=x[u]*y[v]+exchange*y[u]*x[v];
                },8);
                return;
            }
        }
        // Each auxiliary magnetic family conserves the removed spin
        // projection. Cross-sector rows would only store structural zeros.
        std::array<std::vector<size_t>,3> sectors;
        for(size_t a=0;a<m;++a) {
            auto [i,j]=topology_->orbital_pairs[a];
            sectors[i/p_.k+j/p_.k].push_back(a);
        }
        size_t elements=0;
        for(auto &sector:sectors) {
            if(!sector.empty() && sector.size()>(bytes/8-elements)/sector.size())return;
            elements+=sector.size()*sector.size();
        }
        const size_t table_bytes=elements*8;
        const size_t index_bytes=2*m*sizeof(size_t);
        if(index_bytes>bytes-table_bytes)return;
        SpatialERI setup_integrals=integrals_;
        std::vector<double> transformed;
        const size_t pairs=integrals_.k*(integrals_.k+1)/2;
        if(integrals_.format==ERIFormat::thc && integrals_.rank &&
                integrals_.rank<=(construction_bytes-table_bytes-index_bytes)/8/pairs) {
            transformed.resize(pairs*integrals_.rank);
            workers_.run(integrals_.k,[&](size_t p,size_t) {
                for(size_t q=0;q<=p;++q)
                    integrals_.transform_density(p,q,transformed.data()+orbital_pair(p,q)*integrals_.rank);
            });
            setup_integrals.transformed=transformed.data();
            topology_->build_bytes=std::max(topology_->build_bytes,topology_->metadata_bytes+table_bytes+index_bytes+transformed.size()*8);
        }
        topology_->pair_integrals.resize(elements);
        topology_->pair_rows.resize(m);
        topology_->pair_columns.resize(m);
        size_t offset=0;
        for(const auto &sector:sectors) {
            size_t width=sector.size();
            workers_.run(width,[&](size_t local,size_t) {
                size_t a=sector[local],row=offset+local*width;
                topology_->pair_rows[a]=row;
                topology_->pair_columns[a]=local;
                for(size_t b=0;b<width;++b)topology_->pair_integrals[row+b]=pair_value(a,sector[b],setup_integrals);
            });
            offset+=width*width;
        }
    }
    void forward_block(const RemovalLinks &links,const RemovalLinks::Block &block,
                       const double *input,size_t cols,double *t) const {
        std::fill_n(t,(block.end-block.begin)*block.dim*cols,0.);
        for(size_t j=block.begin;j<block.end;++j) {
            const auto &parent=links.parents[j];const auto &c=topology_->bank.matrices[parent.matrix];const auto &g=p_.groups[parent.group];
            double *target=t+(j-block.begin)*block.dim*cols;
            if(g.canonical_full()) {
                const double *source=input+g.block->offset*cols;
                for(size_t a=0;a<c.rows;++a)for(size_t j=c.ptr[a];j<c.ptr[a+1];++j) {
                    const double *row=source+c.indices[j]*cols;
                    for(size_t v=0;v<cols;++v)target[a*cols+v]+=c.data[j]*row[v];
                }
                continue;
            }
            for(size_t a=0;a<c.rows;++a)for(size_t i=0;i<g.size();++i) {
                double value=c.at(a,g.rank(i));if(value==0.)continue;
                for(size_t v=0;v<cols;++v)target[a*cols+v]+=value*input[g.row(i)*cols+v];
            }
        }
    }
    void adjoint(const RemovalLinks &links,const double *u,size_t cols,double *output,
                 size_t first=0,size_t last=none,size_t base=0) {
        workers_.run(p_.groups.size(),[&](size_t group,size_t) {
            const auto &g=p_.groups[group];
            for(size_t j=links.reverse_ptr[group];j<links.reverse_ptr[group+1];++j) {
                size_t parent_id=links.reverse[j];
                if(parent_id<first)continue;
                if(parent_id>=last)break;
                const auto &parent=links.parents[parent_id];const auto &c=topology_->bank.matrices[parent.matrix];
                const double *source=u+(links.offsets[parent_id]-base)*cols;
                // For a singlet primary space, rotational invariance makes
                // every magnetic component of a scalar contraction equal.
                // Retain M=S only and sum the 2S+1 equal components here.
                for(size_t i=0;i<g.size();++i)for(size_t a=0;a<c.rows;++a) {
                    double value=parent.weight*c.at(a,g.rank(i));if(value==0.)continue;
                    for(size_t v=0;v<cols;++v)output[g.row(i)*cols+v]+=value*source[a*cols+v];
                }
            }
        },16);
    }
    void one_body_middle(const double *t,size_t cols,double *u) {
        workers_.run(topology_->one->blocks.size(),[&](size_t b,size_t) {
            const auto &block=topology_->one->blocks[b];
            std::fill_n(u+block.offset*cols,(block.end-block.begin)*block.dim*cols,0.);
            for(size_t a=block.begin;a<block.end;++a)for(size_t r=block.begin;r<block.end;++r) {
                const auto &pa=topology_->one->parents[a],&pr=topology_->one->parents[r];
                double value=h_[(pa.channel%p_.k)*p_.k+pr.channel%p_.k];
                double *target=u+topology_->one->offsets[a]*cols;
                const double *source=t+topology_->one->offsets[r]*cols;
                for(size_t i=0;i<block.dim*cols;++i)target[i]+=value*source[i];
            }
        },16);
    }
    // Contract one destination residue family. Source blocks are either
    // borrowed from the global forward array or regenerated on demand.
    template<class Read>
    void nminus1_destination(size_t a,size_t f,const RemovalLinks::Block &left,
                             size_t cols,Read read,double *result,std::vector<double> &transformed) {
        const auto &ca=topology_->one->residues->configs[a];
        const size_t width=left.dim*cols;
        std::fill_n(result,(left.end-left.begin)*width,0.);
        const double *local=read(left);
        for(size_t i=left.begin;i<left.end;++i) {
            int p=topology_->one->parents[i].channel%p_.k;
            double *target=result+(i-left.begin)*width;
            for(size_t j=left.begin;j<left.end;++j) {
                int r=topology_->one->parents[j].channel%p_.k;
                double value=h_[p*p_.k+r];
                for(size_t query=topology_->auxiliary_queries->reverse_ptr[a];query<topology_->auxiliary_queries->reverse_ptr[a+1];++query) {
                    int q=topology_->auxiliary_queries->queries[query].orbital;
                    value+=.5*ca[q]*eri(p,q,r,q);
                }
                const double *source=local+(j-left.begin)*width;
                for(size_t k=0;k<width;++k)target[k]+=value*source[k];
            }
        }
        for(size_t query=topology_->auxiliary_queries->reverse_ptr[a];query<topology_->auxiliary_queries->reverse_ptr[a+1];++query) {
            const auto location=topology_->auxiliary_queries->queries[query];
            const int q=location.orbital;
            for(size_t record=topology_->auxiliary_queries->ptr[location.residue];record<topology_->auxiliary_queries->ptr[location.residue+1];++record) {
                const auto &connection=topology_->auxiliary_queries->records[record];
                const size_t b=connection.parent;
                const int s=connection.p;
                if(q==s)continue;
                const size_t bi=topology_->one->block_ids[b*topology_->one->families.size()+f];
                if(bi==none)continue;
                const auto &right=topology_->one->blocks[bi];
                const auto &cb=topology_->one->residues->configs[b];
                const Matrix &matrix=topology_->family_banks[f]->get(cb,q,s);
                const double *source=read(right);
                transformed.assign((right.end-right.begin)*width,0.);
                for(size_t j=right.begin;j<right.end;++j) {
                    const double *channel=source+(j-right.begin)*right.dim*cols;
                    double *target=transformed.data()+(j-right.begin)*width;
                    for(size_t row=0;row<left.dim;++row)
                        for(size_t k=matrix.ptr[row];k<matrix.ptr[row+1];++k)for(size_t v=0;v<cols;++v)
                            target[row*cols+v]+=matrix.data[k]*channel[matrix.indices[k]*cols+v];
                }
                for(size_t i=left.begin;i<left.end;++i)for(size_t j=right.begin;j<right.end;++j) {
                    const auto &destination=topology_->one->parents[i],&parent=topology_->one->parents[j];
                    double value=.5*eri(destination.channel%p_.k,q,parent.channel%p_.k,s);
                    if(value==0.)continue;
                    const double *channel=transformed.data()+(j-right.begin)*width;
                    double *target=result+(i-left.begin)*width;
                    for(size_t k=0;k<width;++k)target[k]+=value*channel[k];
                }
            }
        }
    }
    void nminus1_middle(const double *t,size_t cols,double *u) {
        workers_.run(topology_->one->blocks.size(),[&](size_t b,size_t worker) {
            const auto &block=topology_->one->blocks[b];
            auto read=[&](const RemovalLinks::Block &source) {return t+source.offset*cols;};
            nminus1_destination(block.config,block.family,block,cols,read,u+block.offset*cols,temporary_[worker]);
        },8);
    }
    void nminus1_stream(const double *input,size_t cols,double *output,Stats &stats) {
        auto start=Clock::now();
        auto &forward=projected_[0],&destination=mixed_[0],&transformed=temporary_[0];
        for(const auto &block:topology_->one->blocks) {
            auto read=[&](const RemovalLinks::Block &source) {
                forward.resize((source.end-source.begin)*source.dim*cols);
                forward_block(*topology_->one,source,input,cols,forward.data());
                ++stats["removal_blocks_regenerated"];
                return forward.data();
            };
            destination.resize((block.end-block.begin)*block.dim*cols);
            nminus1_destination(block.config,block.family,block,cols,read,destination.data(),transformed);
            block_adjoint(*topology_->one,block,cols,destination.data(),output);
        }
        stats["streaming_ns"]+=ns(start);
    }
    void pair_middle(const RemovalLinks::Block &block,size_t width,const double *t,
                     double *target,size_t worker) {
        const size_t count=block.end-block.begin;
        std::fill_n(target,count*width,0.);
        if(topology_->feature_count) {
            auto &projected=projected_[worker];
            projected.assign(topology_->feature_count*width,0.);
            for(size_t a=0;a<count;++a) {
                const double *feature=topology_->pair_features.data()+topology_->two->parents[block.begin+a].channel*topology_->feature_count;
                for(size_t f=0;f<topology_->feature_count;++f)for(size_t i=0;i<width;++i)
                    projected[f*width+i]+=feature[f]*t[a*width+i];
            }
            // Antisymmetrizing both ends duplicates the direct-minus-exchange
            // value in the ordered grid-pair sum. Other spin families use Z.
            const double scale=same_spin(topology_->two->parents[block.begin].channel) ? .5 : 1.;
            for(size_t f=0;f<topology_->feature_count;++f)for(size_t i=0;i<width;++i)
                projected[f*width+i]*=scale*integrals_.core[f];
            for(size_t a=0;a<count;++a) {
                const double *feature=topology_->pair_features.data()+topology_->two->parents[block.begin+a].channel*topology_->feature_count;
                for(size_t f=0;f<topology_->feature_count;++f)for(size_t i=0;i<width;++i)
                    target[a*width+i]+=feature[f]*projected[f*width+i];
            }
            return;
        }
        for(size_t a=0;a<count;++a) {
            size_t ap=topology_->two->parents[block.begin+a].channel;
            const double *row=topology_->pair_integrals.empty() ? nullptr : topology_->pair_integrals.data()+topology_->pair_rows[ap];
            for(size_t c=0;c<count;++c) {
                size_t cp=topology_->two->parents[block.begin+c].channel;
                double value=row ? row[topology_->pair_columns[cp]] : pair_value(ap,cp,integrals_);
                if(value==0.)continue;
                for(size_t i=0;i<width;++i)target[a*width+i]+=value*t[c*width+i];
            }
        }
    }
    void pair_forward_middle(const double *input,size_t cols,double *u) {
        workers_.run(topology_->two->blocks.size(),[&](size_t b,size_t worker) {
            const auto &block=topology_->two->blocks[b];
            const size_t width=block.dim*cols, block_size=(block.end-block.begin)*width;
            auto &t=temporary_[worker];
            if(t.size()<block_size)t.resize(block_size);
            forward_block(*topology_->two,block,input,cols,t.data());
            pair_middle(block,width,t.data(),u+block.offset*cols,worker);
        },4);
    }
    void block_adjoint(const RemovalLinks &links,const RemovalLinks::Block &block,
                       size_t cols,const double *mixed,double *output) {
        for(size_t j=block.begin;j<block.end;++j) {
            const auto &parent=links.parents[j];
            const auto &g=p_.groups[parent.group];
            const auto &c=topology_->bank.matrices[parent.matrix];
            const double *source=mixed+(j-block.begin)*block.dim*cols;
            if(g.canonical_full()) {
                double *target=output+g.block->offset*cols;
                for(size_t a=0;a<c.rows;++a)for(size_t j=c.ptr[a];j<c.ptr[a+1];++j) {
                    double *row=target+c.indices[j]*cols;
                    const double value=parent.weight*c.data[j];
                    for(size_t v=0;v<cols;++v)row[v]+=value*source[a*cols+v];
                }
                continue;
            }
            for(size_t i=0;i<g.size();++i)for(size_t a=0;a<c.rows;++a) {
                double value=parent.weight*c.at(a,g.rank(i));
                if(value==0.)continue;
                for(size_t v=0;v<cols;++v)output[g.row(i)*cols+v]+=value*source[a*cols+v];
            }
        }
    }
    void apply_private(const double *input,size_t cols,double *output,Stats &stats,bool serial) {
        const size_t workers=serial ? 1 : workers_.count(), output_size=p_.size()*cols;
        const bool reduce=workers>1;
        auto start=Clock::now();
        std::unique_ptr<double[]> partial(reduce ? new double[workers*output_size] : nullptr);
        auto run_worker=[&](size_t worker) {
            double *local=reduce ? partial.get()+worker*output_size : output;
            if(reduce)std::fill_n(local,output_size,0.);
            auto &t=temporary_[worker],&u=mixed_[worker];
            auto process=[&](const RemovalLinks &links,bool pair) {
                constexpr size_t chunk=16;
                for(size_t first=worker*chunk;first<links.blocks.size();first+=workers*chunk) {
                    for(size_t b=first;b<std::min(first+chunk,links.blocks.size());++b) {
                        const auto &block=links.blocks[b];
                        size_t width=block.dim*cols,count=block.end-block.begin;
                        if(t.size()<count*width)t.resize(count*width);
                        if(u.size()<count*width)u.resize(count*width);
                        forward_block(links,block,input,cols,t.data());
                        if(pair)pair_middle(block,width,t.data(),u.data(),worker);
                        else {
                            std::fill_n(u.data(),count*width,0.);
                            for(size_t a=0;a<count;++a)for(size_t r=0;r<count;++r) {
                                int p=links.parents[block.begin+a].channel%p_.k;
                                int q=links.parents[block.begin+r].channel%p_.k;
                                double value=h_[p*p_.k+q];
                                for(size_t i=0;i<width;++i)u[a*width+i]+=value*t[r*width+i];
                            }
                        }
                        block_adjoint(links,block,cols,u.data(),local);
                    }
                }
            };
            process(*topology_->one,false);
            process(*topology_->two,true);
        };
        if(serial)run_worker(0);else workers_.each_worker(run_worker);
        stats["private_block_ns"]+=ns(start);
        if(reduce) {
            start=Clock::now();
            constexpr size_t tile=4096;
            workers_.run(output_size/tile + (output_size%tile>0),[&](size_t t,size_t) {
                size_t first=t*tile,last=std::min(first+tile,output_size);
                for(size_t w=0;w<workers;++w)
                    for(size_t i=first;i<last;++i)output[i]+=partial[w*output_size+i];
            });
            stats["output_reduce_ns"]+=ns(start);
        }
    }
public:
    Auxiliary(const Primary &p,const double *h,SpatialERI integrals,int mode,Workers &workers,
              size_t metadata,size_t coefficients,size_t pair_limit,bool private_output,const Auxiliary *previous=nullptr):
        p_(p),h_(h),integrals_(integrals),mode_(mode),private_output_(private_output),workers_(workers),
        temporary_(workers.count()),mixed_(workers.count()),projected_(workers.count()) {
        const bool append=mode==1 && append_prefix(p,previous) && private_output==previous->private_output_ &&
            previous->topology_->metadata_bytes<=metadata && previous->topology_->bank.bytes()<=coefficients/2;
        const size_t before=append ? previous->p_.groups.size() : 0;
        if(append && before==p.groups.size()) {
            topology_=previous->topology_;
            reused_topology_=true;
            return;
        }
        const size_t added=p.groups.size()-before;
        // Deltas reduce enumeration for small appends. A fresh build also
        // avoids reserving space for two schedules when the suffix is large.
        const bool incremental=before && p.k<=32 && added<=before/2;
        auto construct=[&](bool use_delta) {
            topology_=std::make_shared<Topology>(p,coefficients);
            if(use_delta)topology_->bank=previous->topology_->bank;
            auto links=[&](int rank,size_t allowance) -> std::unique_ptr<RemovalLinks> {
                if(!use_delta)
                    return std::make_unique<RemovalLinks>(p_,topology_->bank,rank,allowance,mode_==0,!private_output_);
                const auto &old=*previous->topology_;
                RemovalLinks delta(p_,topology_->bank,rank,allowance,false,false,before);
                auto next=std::make_unique<RemovalLinks>(rank==1 ? *old.one : *old.two,delta,
                    p_.groups.size(),allowance,!private_output_,rank==1 ? nullptr : &old.orbital_pairs,2*p_.k);
                next->compact_builder_bytes=std::max(next->compact_builder_bytes,
                    v2::checked_add(delta.bytes(),delta.compact_builder_bytes));
                return next;
            };
            topology_->one=links(1,metadata);
            topology_->metadata_bytes=topology_->one->bytes();
            topology_->build_bytes=topology_->metadata_bytes+topology_->one->compact_builder_bytes;
            topology_->one_configs=topology_->one->residue_count;
            if(mode_==0) {
                topology_->auxiliary_queries=std::make_unique<Residues>(topology_->one->residues->configs.size(),p_.k,1,metadata-topology_->metadata_bytes,
                    [&](size_t a)->const Config &{return topology_->one->residues->configs[a];});
                topology_->metadata_bytes+=topology_->auxiliary_queries->bytes();
                topology_->build_bytes=topology_->metadata_bytes;
                std::set<int> sectors;for(auto f:topology_->one->families)sectors.insert(f.spin);
                size_t allowance=(coefficients-topology_->bank.bytes())/std::max(size_t(1),sectors.size());
                for(int s:sectors)topology_->one_banks.emplace(s,std::make_unique<OneBank>(std::max(0,p_.n-1),s,allowance));
                for(auto f:topology_->one->families)topology_->family_banks.push_back(topology_->one_banks.at(f.spin).get());
                std::vector<Residues::Record>().swap(topology_->one->residues->records);
                std::vector<size_t>().swap(topology_->one->residues->ptr);
                std::vector<size_t>().swap(topology_->one->residues->reverse_ptr);
                std::vector<Residues::Query>().swap(topology_->one->residues->queries);
                topology_->metadata_bytes=topology_->one->bytes()+topology_->auxiliary_queries->bytes();
            } else {
                topology_->one->residues.reset();
                topology_->metadata_bytes=topology_->one->bytes();
                topology_->two=links(2,metadata-topology_->metadata_bytes);
                topology_->build_bytes=std::max(topology_->build_bytes,topology_->metadata_bytes+topology_->two->bytes()+topology_->two->compact_builder_bytes);
                topology_->two_configs=topology_->two->residue_count;
                topology_->two->residues.reset();
                topology_->metadata_bytes+=topology_->two->bytes();
                if(topology_->two->parents.empty())return;
                // Only parent channels enter either end of the pair contraction.
                // Compact their orbital-pair directory before constructing the
                // optional integral table (important for sparse, wide orbital sets).
                size_t modes=2*size_t(p_.k),all_pairs=modes*(modes-1)/2;
                if(all_pairs>(metadata-topology_->metadata_bytes)/sizeof(size_t))
                    throw BudgetError("pair channel directory exceeds metadata budget");
                topology_->build_bytes=std::max(topology_->build_bytes,topology_->metadata_bytes+all_pairs*sizeof(size_t));
                std::vector<size_t> channel_ids(all_pairs,none);
                for(auto &parent:topology_->two->parents)channel_ids[parent.channel]=0;
                size_t channel=0;
                for(int a=0;a<2*p_.k;++a)for(int b=a+1;b<2*p_.k;++b,++channel)
                    if(channel_ids[channel]!=none) {
                        channel_ids[channel]=topology_->orbital_pairs.size();topology_->orbital_pairs.push_back({a,b});
                    }
                for(auto &parent:topology_->two->parents)parent.channel=channel_ids[parent.channel];
                size_t orbital_bytes=topology_->orbital_pairs.capacity()*sizeof(topology_->orbital_pairs[0]);
                if(orbital_bytes>metadata-topology_->metadata_bytes)throw BudgetError("pair directory exceeds metadata budget");
                topology_->metadata_bytes+=orbital_bytes;
                prepare_pair_integrals(std::min(pair_limit,metadata-topology_->metadata_bytes),metadata-topology_->metadata_bytes);
                topology_->metadata_bytes+=(topology_->pair_integrals.capacity()+topology_->pair_features.capacity())*8
                    +(topology_->pair_rows.capacity()+topology_->pair_columns.capacity())*sizeof(size_t);
            }
        };
        if(incremental) {
            try {
                construct(true);
                incremental_groups_=added;
            } catch(const BudgetError &) {
                topology_.reset();
                construct(false);
            }
        } else construct(false);
        overlap_bytes_=previous ? v2::checked_add(previous->topology_->metadata_bytes,topology_->build_bytes) :
            topology_->build_bytes;
    }
    size_t entries() const {
        if(private_output_)return workers_.count()==1 ? 0 : workers_.count()*p_.size();
        return mode_==0 ? 2*topology_->one->entries : std::max(2*topology_->one->entries,topology_->two->entries);
    }
    bool can_batch(size_t bytes) const {
        if(mode_!=1 || private_output_)return false;
        for(auto *links:{topology_->one.get(),topology_->two.get()})
            for(const auto &block:links->blocks)
                if((block.end-block.begin)*block.dim>bytes/sizeof(double)/(links==topology_->one.get() ? 2 : 1))
                    return false;
        return true;
    }
    void apply_batched(const double *input,double *output,size_t bytes,Stats &stats) {
        auto start=Clock::now();
        for(const auto *links:{topology_->one.get(),topology_->two.get()}) {
            const bool pair=links==topology_->two.get();
            const size_t capacity=bytes/sizeof(double)/(pair ? 1 : 2);
            auto after=[&](size_t b) {
                return b==links->blocks.size() ? links->entries : links->blocks[b].offset;
            };
            for(size_t first=0;first<links->blocks.size();) {
                const size_t base=after(first);
                size_t last=first+1;
                while(last<links->blocks.size() && after(last+1)-base<=capacity)++last;
                const size_t entries=after(last)-base;
                std::unique_ptr<double[]> u(new double[entries]);
                std::unique_ptr<double[]> t(pair ? nullptr : new double[entries]);
                workers_.run(last-first,[&](size_t b,size_t worker) {
                    const auto &block=links->blocks[first+b];
                    double *target=u.get()+(block.offset-base);
                    const size_t n=block.end-block.begin;
                    if(pair) {
                        auto &forward=temporary_[worker];
                        forward.resize(n*block.dim);
                        forward_block(*links,block,input,1,forward.data());
                        pair_middle(block,block.dim,forward.data(),target,worker);
                    } else {
                        double *forward=t.get()+(block.offset-base);
                        forward_block(*links,block,input,1,forward);
                        std::fill_n(target,n*block.dim,0.);
                        for(size_t i=0;i<n;++i)for(size_t j=0;j<n;++j) {
                            const int p=links->parents[block.begin+i].channel%p_.k;
                            const int r=links->parents[block.begin+j].channel%p_.k;
                            const double scale=h_[p*p_.k+r];
                            for(size_t k=0;k<block.dim;++k)target[i*block.dim+k]+=scale*forward[j*block.dim+k];
                        }
                    }
                },8);
                adjoint(*links,u.get(),1,output,links->blocks[first].begin,links->blocks[last-1].end,base);
                stats["product_array_bytes"]=std::max(stats["product_array_bytes"],entries*sizeof(double)*(pair ? 1 : 2));
                ++stats["residue_batches"];
                first=last;
            }
        }
        stats["batched_gather_ns"]+=ns(start);
        size_t workspace=0;
        for(const auto *bank:{&temporary_,&projected_})
            for(const auto &buffer:*bank)workspace+=buffer.capacity()*sizeof(double);
        stats["worker_workspace_bytes"]=workspace;
    }
    void apply(const double *input,size_t cols,double *output,Stats &stats,bool serial=false) {
        if(serial && mode_==0)nminus1_stream(input,cols,output,stats);
        else if(private_output_ || serial)apply_private(input,cols,output,stats,serial);
        else apply_gather(input,cols,output,stats);
        size_t scratch=0;
        for(auto *bank:{&temporary_,&mixed_,&projected_})for(auto &a:*bank)scratch+=a.capacity()*8;
        stats["worker_workspace_bytes"]=scratch;
    }
    void apply_gather(const double *input,size_t cols,double *output,Stats &stats) {
        auto start=Clock::now();
        std::unique_ptr<double[]> t(new double[topology_->one->entries*cols]),u(new double[topology_->one->entries*cols]);
        workers_.run(topology_->one->blocks.size(),[&](size_t b,size_t) {
            const auto &block=topology_->one->blocks[b];
            forward_block(*topology_->one,block,input,cols,t.get()+block.offset*cols);
        },16);
        stats["forward_ns"]+=ns(start);start=Clock::now();
        if(mode_==0)nminus1_middle(t.get(),cols,u.get());else one_body_middle(t.get(),cols,u.get());
        stats["middle_ns"]+=ns(start);start=Clock::now();adjoint(*topology_->one,u.get(),cols,output);
        stats["adjoint_ns"]+=ns(start);
        if(topology_->two && topology_->two->entries) {
            t.reset();u.reset();
            start=Clock::now();u.reset(new double[topology_->two->entries*cols]);
            pair_forward_middle(input,cols,u.get());stats["pair_forward_middle_ns"]+=ns(start);
            start=Clock::now();adjoint(*topology_->two,u.get(),cols,output);stats["adjoint_ns"]+=ns(start);
        }
    }
    Stats info() const {
        Stats s={{"metadata_bytes",topology_->metadata_bytes},{"metadata_build_bytes",reused_topology_ ? 0 : topology_->build_bytes},{"reused_removal_topology",reused_topology_},{"incremental_removal_groups",incremental_groups_},{"removal_metadata_overlap_bytes",overlap_bytes_},{"coefficient_bytes",topology_->bank.bytes()},
            {"singlet_magnetic_reduction",p_.s==0},
            {"removal_classes",topology_->bank.matrices.size()},{"one_auxiliary_configurations",topology_->one_configs},
            {"one_auxiliary_blocks",topology_->one->blocks.size()},{"one_parent_channels",topology_->one->parents.size()},
            {"one_weighted_links",topology_->one->weighted_links},{"one_intermediate_entries",topology_->one->entries},
            {"product_array_bytes_per_rhs",entries()*8},{"pair_integral_bytes",topology_->pair_integrals.capacity()*8},
            {"pair_factor_bytes",topology_->pair_features.capacity()*8},{"pair_factor_columns",topology_->feature_count},
            {"private_residue_output",private_output_}};
        for(auto &entry:topology_->one_banks)s["coefficient_bytes"]+=entry.second->bytes();
        if(topology_->two) {
            s["two_auxiliary_configurations"]=topology_->two_configs;s["two_auxiliary_blocks"]=topology_->two->blocks.size();
            s["two_parent_channels"]=topology_->two->parents.size();s["two_weighted_links"]=topology_->two->weighted_links;s["two_intermediate_entries"]=topology_->two->entries;
        }
        return s;
    }
};

} }

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
    struct FoldedPair {
        unsigned matrix;
        double scale;
    };
    struct Topology {
        RemovalBank bank;
        std::unique_ptr<RemovalLinks> one,two;
        std::unique_ptr<RemovalQueries> auxiliary_queries;
        std::map<int,std::unique_ptr<OneBank>> one_banks;
        std::vector<const OneBank *> family_banks;
        std::vector<Occupation::Limb> auxiliary_masks;
        std::shared_ptr<const std::vector<double>> auxiliary_integrals;
        std::vector<size_t> local_offsets;
        std::vector<double> local_integrals;
        std::vector<std::pair<int,int>> orbital_pairs;
        std::vector<double> pair_integrals;
        std::vector<size_t> pair_rows,pair_columns;
        std::vector<double> pair_features;
        std::vector<FoldedPair> folded_pairs;
        std::vector<size_t> folded_offsets;
        std::vector<unsigned> mixed_pairs;
        size_t feature_count=0,metadata_bytes=0,build_bytes=0,one_configs=0,two_configs=0;
        Topology(const Primary &primary,size_t budget):bank(primary,budget){}
    };
    std::shared_ptr<Topology> topology_;
    std::vector<std::vector<double>> temporary_,mixed_,projected_;
    bool reused_topology_=false;
    bool reused_auxiliary_integrals_=false;
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
    void prepare_pair_integrals(size_t bytes) {
        const size_t m=topology_->orbital_pairs.size();
        if(!m)return;
        if(integrals_.format==ERIFormat::thc && integrals_.rank) {
            const size_t rank=integrals_.rank;
            const size_t average=topology_->two->parents.size()/topology_->two->blocks.size();
            // The pair contraction uses two applications of the Khatri–Rao
            // features. Only keep them if they reduce the typical block width.
            if(rank<=average/2/rank) {
                if(rank>bytes/8/m/rank)throw BudgetError("N−2 THC features exceed --max-mem");
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
            if(!sector.empty() && sector.size()>(bytes/8-elements)/sector.size())
                throw BudgetError("N−2 pair integral table exceeds --max-mem");
            elements+=sector.size()*sector.size();
        }
        const size_t table_bytes=elements*8;
        const size_t index_bytes=2*m*sizeof(size_t);
        memory_left(bytes-table_bytes,index_bytes,"N−2 pair integral index");
        SpatialERI setup_integrals=integrals_;
        std::vector<double> transformed;
        const size_t pairs=integrals_.k*(integrals_.k+1)/2;
        if(integrals_.format==ERIFormat::thc && integrals_.rank) {
            if(integrals_.rank>(bytes-table_bytes-index_bytes)/8/pairs)
                throw BudgetError("N−2 THC setup exceeds --max-mem");
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
    void fold_single_parent_pairs(size_t budget) {
        const auto &links=*topology_->two;
        if(links.blocks.size()>std::numeric_limits<unsigned>::max() ||
            topology_->pair_integrals.empty() ||
            !std::all_of(p_.groups.begin(),p_.groups.end(),[](const auto &g){return g.canonical_full();}))
            return;
        const size_t singletons=std::count_if(links.blocks.begin(),links.blocks.end(),
            [](const auto &b){return b.end-b.begin==1;});
        const size_t mixed=links.blocks.size()-singletons, groups=p_.groups.size();
        if(singletons<groups || !singletons)return;
        const size_t peak=2*(groups+1)*sizeof(size_t)+mixed*sizeof(unsigned)+2*singletons*sizeof(FoldedPair);
        memory_left(budget,peak,"N−2 folded pair schedule");
        auto &offsets=topology_->folded_offsets;
        auto &mixed_pairs=topology_->mixed_pairs;
        offsets.assign(groups+1,0);
        mixed_pairs.reserve(mixed);
        for(size_t i=0;i<links.blocks.size();++i) {
            const auto &b=links.blocks[i];
            if(b.end-b.begin==1)++offsets[links.parents[b.begin].group+1];
            else mixed_pairs.push_back(i);
        }
        std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());
        auto ends=offsets;
        std::vector<FoldedPair> terms(singletons);
        for(const auto &b:links.blocks)if(b.end-b.begin==1) {
            const auto &parent=links.parents[b.begin];
            const auto c=parent.channel;
            const double scale=topology_->pair_integrals[topology_->pair_rows[c]+topology_->pair_columns[c]];
            terms[ends[parent.group]++]={parent.matrix,links.families[b.family].weight*scale};
        }
        workers_.run(groups,[&](size_t g,size_t) {
            auto first=terms.begin()+offsets[g],last=terms.begin()+offsets[g+1],next=first;
            std::sort(first,last,[](const auto &a,const auto &b){return a.matrix<b.matrix;});
            for(auto it=first;it!=last;) {
                auto term=*it++;
                while(it!=last && it->matrix==term.matrix)term.scale+=it++->scale;
                if(term.scale!=0.)*next++=term;
            }
            ends[g]=next-terms.begin();
        },16);
        auto &folded=topology_->folded_pairs;
        size_t count=0;
        for(size_t g=0;g<groups;++g)count+=ends[g]-offsets[g];
        folded.reserve(count);
        for(size_t g=0;g<groups;++g) {
            const auto first=offsets[g];
            offsets[g]=folded.size();
            folded.insert(folded.end(),terms.begin()+first,terms.begin()+ends[g]);
        }
        offsets.back()=folded.size();
        const size_t bytes=offsets.capacity()*sizeof(size_t)+mixed_pairs.capacity()*sizeof(unsigned)+
            folded.capacity()*sizeof(FoldedPair);
        topology_->build_bytes=std::max(topology_->build_bytes,topology_->metadata_bytes+bytes+
            ends.capacity()*sizeof(size_t)+terms.capacity()*sizeof(FoldedPair));
        topology_->metadata_bytes+=bytes;
    }
    template<size_t Columns=0>
    void forward_block(const RemovalLinks &links,const RemovalLinks::Block &block,
                       const double *input,size_t cols,double *t) const {
        if constexpr(Columns)cols=Columns;
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
    void adjoint(const RemovalLinks &links,const double *u,size_t cols,double *output) {
        workers_.run(p_.groups.size(),[&](size_t group,size_t) {
            const auto &g=p_.groups[group];
            for(size_t j=links.reverse_ptr[group];j<links.reverse_ptr[group+1];++j) {
                size_t parent_id=links.reverse[j];
                const auto &parent=links.parents[parent_id];const auto &c=topology_->bank.matrices[parent.matrix];
                const double *source=u+links.offsets[parent_id]*cols;
                const unsigned weight=topology_->bank.weights[parent.matrix];
                // For a singlet primary space, rotational invariance makes
                // every magnetic component of a scalar contraction equal.
                // Retain M=S only and sum the 2S+1 equal components here.
                for(size_t i=0;i<g.size();++i)for(size_t a=0;a<c.rows;++a) {
                    double value=weight*c.at(a,g.rank(i));if(value==0.)continue;
                    for(size_t v=0;v<cols;++v)output[g.row(i)*cols+v]+=value*source[a*cols+v];
                }
            }
        },16);
    }
    void one_body_middle(const double *t,size_t cols,double *u) {
        workers_.run(topology_->one->blocks.size(),[&](size_t b,size_t) {
            const auto &block=topology_->one->blocks[b];
            std::fill_n(u+topology_->one->offset(block)*cols,(block.end-block.begin)*block.dim*cols,0.);
            for(size_t a=block.begin;a<block.end;++a)for(size_t r=block.begin;r<block.end;++r) {
                const auto &pa=topology_->one->parents[a],&pr=topology_->one->parents[r];
                double value=h_[pa.channel*p_.k+pr.channel];
                double *target=u+topology_->one->offsets[a]*cols;
                const double *source=t+topology_->one->offsets[r]*cols;
                for(size_t i=0;i<block.dim*cols;++i)target[i]+=value*source[i];
            }
        },16);
    }
    double local_integral(size_t config,int p,int r) const {
        const size_t k=p_.k,kk=k*k,pr=p*k+r;
        const auto *table=topology_->auxiliary_integrals.get();
        const auto *integrals=table ? table->data() : nullptr;
        const auto &occupations=topology_->one->residues->configs[config];
        const auto &queries=*topology_->auxiliary_queries;
        double value=h_[pr];
        for(size_t i=queries.reverse_ptr[config];i<queries.reverse_ptr[config+1];++i) {
            const size_t q=queries.queries[i].orbital;
            const double g=integrals ? integrals[(q*k+q)*kk+pr] : eri(p,q,r,q);
            value+=.5*occupations[q]*g;
        }
        return value;
    }
    void prepare_local_integrals(size_t allowance) {
        const auto &links=*topology_->one;
        const size_t offsets_bytes=links.blocks.size()*sizeof(size_t);
        if(links.blocks.empty())return;
        const size_t capacity=memory_left(allowance,offsets_bytes,"N−1 local integral offsets")/sizeof(double);
        size_t entries=0;
        for(const auto &block:links.blocks) {
            const size_t count=block.end-block.begin;
            if(count>(capacity-entries)/count)
                throw BudgetError("N−1 local integral table exceeds --max-mem");
            entries+=count*count;
        }
        auto &offsets=topology_->local_offsets;
        auto &values=topology_->local_integrals;
        offsets.resize(links.blocks.size());
        size_t start=0;
        for(size_t i=0;i<links.blocks.size();++i) {
            offsets[i]=start;
            const auto &block=links.blocks[i];
            const size_t count=block.end-block.begin;
            start+=count*count;
        }
        values.resize(entries);
        workers_.run(links.blocks.size(),[&](size_t b,size_t) {
            const auto &block=links.blocks[b];
            const size_t count=block.end-block.begin;
            const auto *parents=links.parents.data()+block.begin;
            double *block_values=values.data()+offsets[b];
            for(size_t i=0;i<count;++i)for(size_t j=0;j<count;++j)
                block_values[i*count+j]=local_integral(block.config,parents[i].channel,parents[j].channel);
        },32);
        topology_->metadata_bytes+=offsets.capacity()*sizeof(size_t)+values.capacity()*sizeof(double);
    }
    // Contract one destination residue family. Source blocks are either
    // borrowed from the global forward array or regenerated on demand.
    template<size_t Columns,class Read>
    void nminus1_destination(size_t a,size_t f,const RemovalLinks::Block &left,
                             size_t cols,Read read,double *result,std::vector<double> &transformed) {
        if constexpr(Columns)cols=Columns;
        const auto &links=*topology_->one;
        const auto &queries=*topology_->auxiliary_queries;
        const size_t k=p_.k,kk=k*k;
        const auto *layout=topology_->auxiliary_integrals.get();
        const auto *table=layout ? layout->data() : nullptr;
        const auto *left_parents=links.parents.data()+left.begin;
        const auto &bank=*topology_->family_banks[f];
        const size_t width=left.dim*cols,left_count=left.end-left.begin;
        std::fill_n(result,left_count*width,0.);
        const double *local=read(left);
        const double *local_values=topology_->local_integrals.data()+topology_->local_offsets[&left-links.blocks.data()];
        for(size_t i=0;i<left_count;++i) {
            double *target=result+i*width;
            for(size_t j=0;j<left_count;++j) {
                const double value=local_values[i*left_count+j];
                const double *source=local+j*width;
                for(size_t v=0;v<width;++v)target[v]+=value*source[v];
            }
        }
        const bool identity_directory=links.block_ids.empty();
        for(size_t query=queries.reverse_ptr[a];query<queries.reverse_ptr[a+1];++query) {
            const auto location=queries.queries[query];
            const int q=location.orbital;
            for(size_t record=queries.ptr[location.config];record<queries.ptr[location.config+1];++record) {
                const auto &connection=queries.parents[record];
                const size_t b=connection.config;
                const int s=connection.orbital;
                if(q==s)continue;
                const size_t bi=identity_directory ? b : links.block_ids[b*links.families.size()+f];
                if(bi==none)continue;
                const auto &right=links.blocks[bi];
                const auto &matrix=topology_->auxiliary_masks.empty()
                    ? bank.get(links.residues->configs[b],q,s)
                    : bank.at(bank.index(topology_->auxiliary_masks[b][0],topology_->auxiliary_masks[b][1],q,s));
                const double *source=read(right);
                const size_t count=right.end-right.begin;
                const auto *right_parents=links.parents.data()+right.begin;
                const double *integrals=table ? table+(q*k+s)*kk : nullptr;
                // With one destination channel the spin transform is consumed
                // once. Accumulate it directly into the output residue.
                if constexpr(Columns==1)if(left_count==1) {
                    const int p=left_parents[0].channel;
                    if(left.dim==1 && right.dim==1) {
                        double value=0.;
                        for(size_t j=0;j<count;++j) {
                            const int r=right_parents[j].channel;
                            const double scale=.5*(integrals ? integrals[p*k+r] : eri(p,q,r,s));
                            value+=scale*source[j];
                        }
                        result[0]+=matrix.values[0]*value;
                        continue;
                    }
                    // For an expensive spin transform, combine the orbital
                    // channels first and apply the shared matrix just once.
                    if(count>1 && matrix.data.size()+left.dim>2*right.dim) {
                        transformed.assign(right.dim,0.);
                        for(size_t j=0;j<count;++j) {
                            const int r=right_parents[j].channel;
                            const double scale=.5*(integrals ? integrals[p*k+r] : eri(p,q,r,s));
                            if(scale==0.)continue;
                            const double *channel=source+j*right.dim;
                            for(size_t i=0;i<right.dim;++i)transformed[i]+=scale*channel[i];
                        }
                        for(size_t row=0;row<left.dim;++row) {
                            double value=0.;
                            for(size_t m=matrix.ptr[row];m<matrix.ptr[row+1];++m)
                                value+=matrix.data[m]*transformed[matrix.indices[m]];
                            result[row]+=value;
                        }
                        continue;
                    }
                    for(size_t j=0;j<count;++j) {
                        const int r=right_parents[j].channel;
                        const double scale=.5*(integrals ? integrals[p*k+r] : eri(p,q,r,s));
                        if(scale==0.)continue;
                        const double *channel=source+j*right.dim;
                        for(size_t row=0;row<left.dim;++row) {
                            double value=0.;
                            for(size_t m=matrix.ptr[row];m<matrix.ptr[row+1];++m)
                                value+=matrix.data[m]*channel[matrix.indices[m]];
                            result[row]+=scale*value;
                        }
                    }
                    continue;
                }
                transformed.assign(count*width,0.);
                for(size_t j=0;j<count;++j) {
                    const double *channel=source+j*right.dim*cols;
                    double *target=transformed.data()+j*width;
                    for(size_t row=0;row<left.dim;++row)
                        for(size_t k=matrix.ptr[row];k<matrix.ptr[row+1];++k)for(size_t v=0;v<cols;++v)
                            target[row*cols+v]+=matrix.data[k]*channel[matrix.indices[k]*cols+v];
                }
                for(size_t i=0;i<left_count;++i) {
                    const int p=left_parents[i].channel;
                    double *target=result+i*width;
                    for(size_t j=0;j<count;++j) {
                        const int r=right_parents[j].channel;
                        const double value=.5*(integrals ? integrals[p*k+r] : eri(p,q,r,s));
                        if(value==0.)continue;
                        const double *channel=transformed.data()+j*width;
                        for(size_t v=0;v<width;++v)target[v]+=value*channel[v];
                    }
                }
            }
        }
    }
    void nminus1_middle(const double *t,size_t cols,double *u) {
        const auto &links=*topology_->one;
        workers_.run(links.blocks.size(),[&](size_t b,size_t worker) {
            const auto &block=links.blocks[b];
            auto read=[&](const RemovalLinks::Block &source) {return t+links.offset(source)*cols;};
            double *out=u+links.offset(block)*cols;
            if(cols==1)nminus1_destination<1>(block.config,block.family,block,cols,read,out,temporary_[worker]);
            else nminus1_destination<0>(block.config,block.family,block,cols,read,out,temporary_[worker]);
        },32);
    }
    template<size_t Width>
    void pair_contract(const RemovalLinks::Block &block,size_t width,const double *t,double *target) const {
        if constexpr(Width)width=Width;
        const size_t count=block.end-block.begin;
        const auto *parents=topology_->two->parents.data()+block.begin;
        const auto *table=topology_->pair_integrals.data();
        for(size_t a=0;a<count;++a) {
            const size_t ap=parents[a].channel;
            const double *row=table+topology_->pair_rows[ap];
            for(size_t c=0;c<count;++c) {
                const double value=row[topology_->pair_columns[parents[c].channel]];
                if(value==0.)continue;
                for(size_t i=0;i<width;++i)target[a*width+i]+=value*t[c*width+i];
            }
        }
    }
    void pair_middle(const RemovalLinks::Block &block,size_t width,const double *t,
                     double *target,size_t worker) {
        const size_t count=block.end-block.begin;
        std::fill_n(target,count*width,0.);
        if(!topology_->pair_integrals.empty()) {
            switch(width) {
                case 1: return pair_contract<1>(block,width,t,target);
                case 2: return pair_contract<2>(block,width,t,target);
                case 3: return pair_contract<3>(block,width,t,target);
                case 5: return pair_contract<5>(block,width,t,target);
                case 9: return pair_contract<9>(block,width,t,target);
                default: return pair_contract<0>(block,width,t,target);
            }
        }
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
        throw std::logic_error("N−2 pair contraction has no integral representation");
    }
    void pair_forward_middle(const double *input,size_t cols,double *u) {
        workers_.run(topology_->two->blocks.size(),[&](size_t b,size_t worker) {
            const auto &block=topology_->two->blocks[b];
            const size_t width=block.dim*cols, block_size=(block.end-block.begin)*width;
            auto &t=temporary_[worker];
            if(t.size()<block_size)t.resize(block_size);
            forward_block(*topology_->two,block,input,cols,t.data());
            pair_middle(block,width,t.data(),u+topology_->two->offset(block)*cols,worker);
        },4);
    }
    template<size_t Columns=0>
    void block_adjoint(const RemovalLinks &links,const RemovalLinks::Block &block,
                       size_t cols,const double *mixed,double *output) {
        if constexpr(Columns)cols=Columns;
        const unsigned weight=links.families[block.family].weight;
        for(size_t j=block.begin;j<block.end;++j) {
            const auto &parent=links.parents[j];
            const auto &g=p_.groups[parent.group];
            const auto &c=topology_->bank.matrices[parent.matrix];
            const double *source=mixed+(j-block.begin)*block.dim*cols;
            if(g.canonical_full()) {
                double *target=output+g.block->offset*cols;
                for(size_t a=0;a<c.rows;++a)for(size_t j=c.ptr[a];j<c.ptr[a+1];++j) {
                    double *row=target+c.indices[j]*cols;
                    const double value=weight*c.data[j];
                    for(size_t v=0;v<cols;++v)row[v]+=value*source[a*cols+v];
                }
                continue;
            }
            for(size_t i=0;i<g.size();++i)for(size_t a=0;a<c.rows;++a) {
                double value=weight*c.at(a,g.rank(i));
                if(value==0.)continue;
                for(size_t v=0;v<cols;++v)output[g.row(i)*cols+v]+=value*source[a*cols+v];
            }
        }
    }
    bool apply_single_pair(const RemovalLinks &links,const RemovalLinks::Block &block,
                           const double *input,double *output) const {
        const auto &parent=links.parents[block.begin];
        const auto &group=p_.groups[parent.group];
        if(!group.canonical_full())return false;
        const auto &matrix=topology_->bank.matrices[parent.matrix];
        const auto channel=parent.channel;
        const double scale=topology_->pair_integrals[topology_->pair_rows[channel]+topology_->pair_columns[channel]];
        if(scale==0.)return true;
        const unsigned weight=links.families[block.family].weight;
        const double *x=input+group.block->offset;
        double *y=output+group.block->offset;
        for(size_t row=0;row<matrix.rows;++row) {
            const size_t begin=matrix.ptr[row],end=matrix.ptr[row+1];
            double projected=0.;
            for(size_t j=begin;j<end;++j)projected+=matrix.data[j]*x[matrix.indices[j]];
            const double mixed=scale*projected;
            for(size_t j=begin;j<end;++j)y[matrix.indices[j]]+=(weight*matrix.data[j])*mixed;
        }
        return true;
    }
    void apply_folded_pairs(const double *input,double *output) {
        auto apply_group=[&](size_t g,size_t) {
            const auto offset=p_.groups[g].block->offset;
            const double *x=input+offset;
            double *y=output+offset;
            for(size_t i=topology_->folded_offsets[g];i<topology_->folded_offsets[g+1];++i) {
                const auto &term=topology_->folded_pairs[i];
                const auto &matrix=topology_->bank.matrices[term.matrix];
                for(size_t row=0;row<matrix.rows;++row) {
                    const size_t begin=matrix.ptr[row],end=matrix.ptr[row+1];
                    double projected=0.;
                    for(size_t j=begin;j<end;++j)projected+=matrix.data[j]*x[matrix.indices[j]];
                    const double mixed=term.scale*projected;
                    for(size_t j=begin;j<end;++j)y[matrix.indices[j]]+=matrix.data[j]*mixed;
                }
            }
        };
        workers_.run(p_.groups.size(),apply_group,16);
    }
    template<size_t Columns>
    void apply_private(const double *input,size_t cols,double *output,Stats &stats) {
        if constexpr(Columns)cols=Columns;
        const size_t workers=workers_.count(), output_size=p_.size()*cols;
        const bool reduce=workers>1;
        auto start=Clock::now();
        const bool folded=Columns==1 && !topology_->folded_offsets.empty();
        if(folded) {
            apply_folded_pairs(input,output);
            stats["folded_pair_ns"]+=ns(start);
            start=Clock::now();
        }
        std::unique_ptr<double[]> partial(reduce ? new double[workers*output_size] : nullptr);
        std::atomic<size_t> next_one{0},next_two{0};
        auto run_worker=[&](size_t worker) {
            double *local=reduce ? partial.get()+worker*output_size : output;
            if(reduce)std::fill_n(local,output_size,0.);
            auto &t=temporary_[worker],&u=mixed_[worker];
            auto process=[&](const RemovalLinks &links,bool pair,std::atomic<size_t> &next) {
                const bool compact=pair && folded;
                const size_t limit=compact ? topology_->mixed_pairs.size() : links.blocks.size();
                // Claim contiguous chunks so faster workers can take more
                // blocks while keeping the schedule reads sequential.
                constexpr size_t chunk=128;
                auto claim=[&] {return next.fetch_add(chunk,std::memory_order_relaxed);};
                for(size_t first=reduce ? claim() : 0;first<limit;
                    first=reduce ? claim() : first+chunk) {
                    for(size_t b=first;b<std::min(first+chunk,limit);++b) {
                        const auto &block=links.blocks[compact ? topology_->mixed_pairs[b] : b];
                        size_t width=block.dim*cols,count=block.end-block.begin;
                        if constexpr(Columns==1)
                            if(pair && count==1 && !topology_->pair_integrals.empty() &&
                                apply_single_pair(links,block,input,local))continue;
                        if(t.size()<count*width)t.resize(count*width);
                        if(u.size()<count*width)u.resize(count*width);
                        forward_block<Columns>(links,block,input,cols,t.data());
                        if(pair)pair_middle(block,width,t.data(),u.data(),worker);
                        else {
                            std::fill_n(u.data(),count*width,0.);
                            for(size_t a=0;a<count;++a)for(size_t r=0;r<count;++r) {
                                int p=links.parents[block.begin+a].channel;
                                int q=links.parents[block.begin+r].channel;
                                double value=h_[p*p_.k+q];
                                for(size_t i=0;i<width;++i)u[a*width+i]+=value*t[r*width+i];
                            }
                        }
                        block_adjoint<Columns>(links,block,cols,u.data(),local);
                    }
                }
            };
            process(*topology_->one,false,next_one);
            process(*topology_->two,true,next_two);
        };
        workers_.each_worker(run_worker);
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
              size_t budget,bool private_output,const Auxiliary *previous=nullptr):
        p_(p),h_(h),integrals_(integrals),mode_(mode),private_output_(private_output),workers_(workers),
        temporary_(workers.count()),mixed_(workers.count()),projected_(workers.count()) {
        const bool append=mode==1 && append_prefix(p,previous) && private_output==previous->private_output_;
        const size_t before=append ? previous->p_.groups.size() : 0;
        if(append && before==p.groups.size()) {
            topology_=previous->topology_;
            reused_topology_=true;
            return;
        }
        const size_t added=p.groups.size()-before;
        // Deltas reduce enumeration for small appends. A fresh build also
        // avoids reserving space for two schedules when the suffix is large.
        const bool incremental=before && p.k<=64 && added<=before/2;
        auto construct=[&](bool use_delta) {
            topology_=std::make_shared<Topology>(p,budget);
            if(use_delta)topology_->bank=previous->topology_->bank;
            auto links=[&](int rank,size_t allowance) -> std::unique_ptr<RemovalLinks> {
                if(!use_delta)
                    return std::make_unique<RemovalLinks>(p_,topology_->bank,rank,allowance,mode_==0,!private_output_);
                const auto &old=*previous->topology_;
                RemovalLinks delta(p_,topology_->bank,rank,allowance,false,false,before);
                auto next=std::make_unique<RemovalLinks>(rank==1 ? *old.one : *old.two,delta,
                    p_.groups.size(),allowance,!private_output_,rank==1 ? nullptr : &old.orbital_pairs,2*p_.k);
                next->compact_builder_bytes=std::max(next->compact_builder_bytes,
                    checked_add(delta.bytes(),delta.compact_builder_bytes));
                return next;
            };
            topology_->one=links(1,memory_left(budget,topology_->bank.bytes(),"removal coefficients"));
            size_t metadata=memory_left(budget,topology_->bank.bytes(),"removal coefficients");
            topology_->metadata_bytes=topology_->one->bytes();
            topology_->build_bytes=topology_->metadata_bytes+topology_->one->compact_builder_bytes;
            topology_->one_configs=topology_->one->residue_count;
            if(mode_==0) {
                std::vector<Residues::Record>().swap(topology_->one->residues->records);
                std::vector<size_t>().swap(topology_->one->residues->ptr);
                if(topology_->one->families.size()==1 && topology_->one->blocks.size()==topology_->one_configs)
                    std::vector<size_t>().swap(topology_->one->block_ids);
                topology_->metadata_bytes=topology_->one->bytes();
                const auto &configs=topology_->one->residues->configs;
                topology_->auxiliary_queries=std::make_unique<RemovalQueries>(configs,p_.k,metadata-topology_->metadata_bytes);
                topology_->build_bytes=std::max(topology_->build_bytes,
                    topology_->metadata_bytes+topology_->auxiliary_queries->build_bytes);
                topology_->metadata_bytes+=topology_->auxiliary_queries->bytes();
                std::set<int> sectors;for(auto f:topology_->one->families)sectors.insert(f.spin);
                // Removing one electron can open one shell. Keep room for
                // the doubly occupied transfer site in same-open-shell moves.
                const int template_electrons=std::min(p_.n-1,p_.maximum_selected_open()+3);
                for(int s:sectors) {
                    const size_t allowance=memory_left(metadata,topology_->metadata_bytes,"N−1 residue links");
                    auto bank=std::make_unique<OneBank>(template_electrons,s,allowance);
                    metadata-=bank->bytes();
                    topology_->one_banks.emplace(s,std::move(bank));
                }
                for(auto f:topology_->one->families)topology_->family_banks.push_back(topology_->one_banks.at(f.spin).get());
                if(p_.k<=64) {
                    memory_left(metadata-topology_->metadata_bytes,configs.size()*sizeof(Occupation::Limb),
                                "N−1 occupation masks");
                    auto &masks=topology_->auxiliary_masks;
                    masks.reserve(configs.size());
                    for(const auto &cfg:configs)masks.push_back(Occupation(cfg).first);
                    topology_->metadata_bytes+=masks.capacity()*sizeof(masks[0]);
                }
                if(integrals_.format==ERIFormat::dense && p_.k) {
                    const size_t k=p_.k,kk=k*k;
                    const size_t remaining=metadata-topology_->metadata_bytes;
                    if(kk>remaining/sizeof(double)/kk)throw BudgetError("N−1 ERI layout exceeds --max-mem");
                    const auto shared=previous ? previous->topology_->auxiliary_integrals : nullptr;
                    if(shared) {
                        memory_left(remaining,shared->capacity()*sizeof(double),"N−1 ERI layout");
                        topology_->auxiliary_integrals=shared;
                        reused_auxiliary_integrals_=true;
                    } else {
                        auto table=std::make_shared<std::vector<double>>(kk*kk);
                        workers_.run(kk,[&](size_t qs,size_t) {
                            const size_t q=qs/k,s=qs%k;
                            for(size_t p=0;p<k;++p)for(size_t r=0;r<k;++r)
                                (*table)[qs*kk+p*k+r]=integrals_.data[((p*k+q)*k+r)*k+s];
                        },8);
                        topology_->auxiliary_integrals=std::move(table);
                    }
                }
                topology_->metadata_bytes+=auxiliary_integral_bytes();
                prepare_local_integrals(metadata-topology_->metadata_bytes);
                topology_->build_bytes=std::max(topology_->build_bytes,topology_->metadata_bytes);
            } else {
                topology_->one->residues.reset();
                topology_->metadata_bytes=topology_->one->bytes();
                topology_->two=links(2,memory_left(metadata,topology_->metadata_bytes,"N−2 one-removal schedule"));
                metadata=memory_left(budget,topology_->bank.bytes(),"removal coefficients");
                topology_->build_bytes=std::max(topology_->build_bytes,topology_->metadata_bytes+topology_->two->bytes()+topology_->two->compact_builder_bytes);
                topology_->two_configs=topology_->two->residue_count;
                topology_->two->residues.reset();
                topology_->metadata_bytes+=topology_->two->bytes();
                memory_left(metadata,topology_->metadata_bytes,"N−2 removal schedules");
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
                std::vector<size_t>().swap(channel_ids);
                prepare_pair_integrals(metadata-topology_->metadata_bytes);
                topology_->metadata_bytes+=(topology_->pair_integrals.capacity()+topology_->pair_features.capacity())*8
                    +(topology_->pair_rows.capacity()+topology_->pair_columns.capacity())*sizeof(size_t);
                if(private_output_)fold_single_parent_pairs(metadata-topology_->metadata_bytes);
            }
        };
        construct(incremental);
        if(incremental)incremental_groups_=added;
        overlap_bytes_=previous ? checked_add(previous->topology_->metadata_bytes,topology_->build_bytes) :
            topology_->build_bytes;
        if(reused_auxiliary_integrals_)overlap_bytes_-=auxiliary_integral_bytes();
    }
    size_t auxiliary_integral_bytes() const {
        const auto &table=topology_->auxiliary_integrals;
        return table ? table->capacity()*sizeof(double) : 0;
    }
    size_t entries() const {
        if(private_output_)return workers_.count()==1 ? 0 : workers_.count()*p_.size();
        return mode_==0 ? 2*topology_->one->entries : std::max(2*topology_->one->entries,topology_->two->entries);
    }
    void apply(const double *input,size_t cols,double *output,Stats &stats) {
        if(private_output_) {
            if(cols==1)apply_private<1>(input,cols,output,stats);
            else apply_private<0>(input,cols,output,stats);
        } else apply_gather(input,cols,output,stats);
        size_t scratch=0;
        for(auto *bank:{&temporary_,&mixed_,&projected_})for(auto &a:*bank)scratch+=a.capacity()*8;
        stats["worker_workspace_bytes"]=scratch;
    }
    void apply_gather(const double *input,size_t cols,double *output,Stats &stats) {
        auto start=Clock::now();
        std::unique_ptr<double[]> t(new double[topology_->one->entries*cols]),u(new double[topology_->one->entries*cols]);
        workers_.run(topology_->one->blocks.size(),[&](size_t b,size_t) {
            const auto &block=topology_->one->blocks[b];
            forward_block(*topology_->one,block,input,cols,t.get()+topology_->one->offset(block)*cols);
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
        if(mode_==0) {
            s["auxiliary_integral_bytes"]=auxiliary_integral_bytes();
            s["reused_auxiliary_integrals"]=reused_auxiliary_integrals_;
            s["auxiliary_local_integral_bytes"]=topology_->local_integrals.capacity()*sizeof(double)+
                topology_->local_offsets.capacity()*sizeof(size_t);
        }
        if(topology_->two) {
            s["two_auxiliary_configurations"]=topology_->two_configs;s["two_auxiliary_blocks"]=topology_->two->blocks.size();
            s["two_parent_channels"]=topology_->two->parents.size();s["two_weighted_links"]=topology_->two->weighted_links;s["two_intermediate_entries"]=topology_->two->entries;
            s["folded_pair_terms"]=topology_->folded_pairs.size();
            s["mixed_pair_blocks"]=topology_->mixed_pairs.size();
        }
        return s;
    }
};

} }

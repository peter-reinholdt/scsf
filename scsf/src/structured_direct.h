/* SCSF; GPL-3.0-or-later. Recursive matrix, tree and particle/hole products. */
#pragma once
#include "structured_internal.h"
#include "cih_diagonal.h"
#include "structured_eri.h"

namespace scsf { namespace structured {

// Occupation-prefix tree. Leaves hold integer configuration/string addresses.
// A connected pair can add and remove at most two electrons in any subset
// of orbitals. Track both counts even when the strings are only a partition.
class OccupationTree {
    struct Node {
        std::array<int,3> child{{-1,-1,-1}};
        unsigned leaf=~0u,minimum=~0u;
    };
    std::vector<Node> nodes_;
    size_t length_=0;
public:
    template<class Getter>OccupationTree(size_t count,size_t length,size_t maximum,Getter get):length_(length) {
        nodes_.emplace_back();
        for(size_t g=0;g<count;++g) {
            auto &cfg=get(g);size_t node=0;
            nodes_[node].minimum=std::min(nodes_[node].minimum,unsigned(g));
            for(int v:cfg) {
                int next=nodes_[node].child[v];
                if(next<0) {
                    if(nodes_.size()>=maximum/sizeof(Node))throw BudgetError("occupation tree exceeds metadata budget");
                    next=nodes_.size();nodes_[node].child[v]=next;nodes_.emplace_back();
                }
                node=next;nodes_[node].minimum=std::min(nodes_[node].minimum,unsigned(g));
            }
            nodes_[node].leaf=g;
        }
        if(bytes()>maximum)throw BudgetError("occupation tree exceeds metadata budget");
    }
    template<class Callback>void visit(const Config &cfg,unsigned maximum,Callback callback) const {
        auto search=[&](auto &&self,size_t node,size_t level,int added,int removed)->void {
            const auto &n=nodes_[node];if(n.minimum>maximum)return;
            if(level==length_) {callback(n.leaf,added+removed);return;}
            const int source=cfg[level];
            for(int v=0;v<3;++v) {
                const int child=n.child[v];
                const int a=added+std::max(0,v-source),r=removed+std::max(0,source-v);
                if(child>=0 && a<=2 && r<=2)self(self,child,level+1,a,r);
            }
        };
        search(search,0,0,0,0);
    }
    size_t bytes() const {return nodes_.capacity()*sizeof(Node);}
};

// Reference-partition strings and CFG parent lists. Repeated substring queries
// are retained once per unique main string, rather than once per primary CFG.
class ParticleHole {
    std::vector<Config> main_,other_;
    std::vector<unsigned> main_id_,other_id_,parents_;
    std::vector<size_t> parent_ptr_;
    std::array<size_t,OrbitalSymmetry::extent> sector_{};
    size_t sector_count_=0;
    using NeighborBuckets=std::array<size_t,6>;
    std::vector<NeighborBuckets> neighbor_ptr_;
    std::vector<std::uint16_t> narrow_neighbors_;
    std::vector<unsigned> neighbors_;
    std::vector<unsigned char> distances_;
    std::unique_ptr<OccupationTree> tree_;
    size_t fixed_bytes_=0,main_sample_visits_=0,other_sample_visits_=0;
    size_t build_bytes_=0;
public:
    ParticleHole(const Primary &p,Workers &workers,size_t maximum) {
        if(p.groups.empty())return;
        const auto &ref=*p.groups.front().cfg;
        std::vector<int> hsites,psites;
        for(int i=0;i<p.k;++i)(ref[i]==2 ? hsites : psites).push_back(i);
        auto strings=[&](const std::vector<int> &sites,std::vector<Config> &unique,std::vector<unsigned> &ids) {
            std::map<Config,unsigned> map;
            ids.reserve(p.groups.size());
            for(auto &g:p.groups) {
                Config c;for(int site:sites)c.push_back((*g.cfg)[site]);
                auto [it,inserted]=map.emplace(c,map.size());
                if(inserted)unique.push_back(std::move(c));
                ids.push_back(it->second);
            }
        };
        strings(hsites,other_,other_id_);strings(psites,main_,main_id_);
        auto cost=[&](const auto &strings,const auto &ids) {
            auto distance=[](const Config &a,const Config &b,int cap) {
                int d=0;
                for(size_t i=0;i<a.size() && d<=cap;++i)d+=std::abs(a[i]-b[i]);
                return d;
            };
            size_t samples=std::min(size_t(256),p.groups.size()), visits=0;
            for(size_t i=0;i<samples;++i) {
                size_t a=i*(p.groups.size()/samples);
                for(size_t j=0;j<=i;++j) {
                    size_t b=j*(p.groups.size()/samples);
                    int d=distance(strings[ids[a]],strings[ids[b]],4);
                    if(d<=4)++visits;
                }
            }
            return visits;
        };
        // Estimate posting traffic using deterministic samples of occupation pairs.
        main_sample_visits_=cost(main_,main_id_);
        other_sample_visits_=cost(other_,other_id_);
        if(other_sample_visits_<main_sample_visits_) {main_.swap(other_);main_id_.swap(other_id_);}
        std::array<bool,OrbitalSymmetry::extent> present{};
        for(const auto &group:p.groups)present[group.block->irrep]=true;
        for(size_t g=0;g<present.size();++g)if(present[g])sector_[g]=sector_count_++;
        if(main_.size()>=maximum/sizeof(size_t)/sector_count_)
            throw BudgetError("particle–hole sector postings exceed metadata budget");
        parent_ptr_.assign(main_.size()*sector_count_+1,0);
        auto posting=[&](size_t g) {return main_id_[g]*sector_count_+sector_[p.groups[g].block->irrep];};
        for(size_t g=0;g<p.groups.size();++g)++parent_ptr_[posting(g)+1];
        std::partial_sum(parent_ptr_.begin(),parent_ptr_.end(),parent_ptr_.begin());
        parents_.resize(p.groups.size());auto cursor=parent_ptr_;
        for(size_t g=0;g<p.groups.size();++g)parents_[cursor[posting(g)]++]=g;
        fixed_bytes_=bytes();
        if(fixed_bytes_>maximum)throw BudgetError("particle/hole strings exceed metadata budget");
        tree_=std::make_unique<OccupationTree>(main_.size(),main_[0].size(),maximum-fixed_bytes_,
            [&](size_t i)->const Config &{return main_[i];});
        fixed_bytes_=bytes();
        if(!other_.empty() && other_.size()<=4096) {
            memory_left(maximum-fixed_bytes_,other_.size()*other_.size(),"particle–hole distance table");
            distances_.resize(other_.size()*other_.size());
            workers.run(other_.size(),[&](size_t i,size_t) {
                for(size_t j=0;j<other_.size();++j) {
                    int d=0;for(size_t k=0;k<other_[i].size();++k)d+=std::abs(other_[i][k]-other_[j][k]);
                    distances_[i*other_.size()+j]=std::min(d,5);
                }
            });
        }
        fixed_bytes_=bytes();
        const bool narrow=main_.size()-1<=std::numeric_limits<std::uint16_t>::max();
        memory_left(maximum-fixed_bytes_,main_.size()*sizeof(NeighborBuckets),"particle–hole neighbor directory");
        neighbor_ptr_.resize(main_.size());
        workers.run(main_.size(),[&](size_t i,size_t) {
            tree_->visit(main_[i],~0u,[&](unsigned,int d){++neighbor_ptr_[i][d+1];});
            auto &bucket=neighbor_ptr_[i];
            std::partial_sum(bucket.begin(),bucket.end(),bucket.begin());
        });
        size_t end=0;
        for(auto &bucket:neighbor_ptr_) {
            for(auto &offset:bucket)offset=checked_add(offset,end);
            end=bucket.back();
        }
        const size_t id_bytes=narrow ? sizeof(narrow_neighbors_[0]) : sizeof(neighbors_[0]);
        if(end>memory_left(maximum,bytes(),"particle–hole directory")/id_bytes)
            throw BudgetError("particle–hole neighbor cache exceeds --max-mem");
        auto retain=[&](auto &ids) {
            ids.resize(end);
            workers.run(main_.size(),[&](size_t i,size_t) {
                auto next=neighbor_ptr_[i];
                tree_->visit(main_[i],~0u,[&](unsigned id,int d){ids[next[d]++]=id;});
            });
        };
        if(narrow)retain(narrow_neighbors_);
        else retain(neighbors_);
        build_bytes_=bytes();
        tree_.reset();
    }
    void query(size_t group,unsigned irrep,std::vector<unsigned> &result) const {
        result.clear();if(main_.empty())return;
        unsigned a=main_id_[group],b=other_id_[group];
        const auto *distance_row=distances_.empty() ? nullptr : distances_.data()+b*other_.size();
        auto visit=[&](unsigned id,int distance) {
            const size_t posting=id*sector_count_+sector_[irrep];
            for(size_t j=parent_ptr_[posting];j<parent_ptr_[posting+1];++j) {
                unsigned g=parents_[j];if(g>group)break;
                unsigned c=other_id_[g];int d=0;
                if(distance_row)d=distance_row[c];
                else for(size_t k=0;k<other_[b].size();++k)d+=std::abs(other_[b][k]-other_[c][k]);
                if(d+distance<=4)result.push_back(g);
            }
        };
        auto visit_cached=[&](const auto &ids) {
            const auto &bucket=neighbor_ptr_[a];
            for(int d=0;d<=4;++d)
                for(size_t j=bucket[d];j<bucket[d+1];++j)visit(ids[j],d);
        };
        if(narrow_neighbors_.empty())visit_cached(neighbors_);
        else visit_cached(narrow_neighbors_);
        std::sort(result.begin(),result.end());
    }
    size_t bytes() const {
        size_t n=(main_id_.capacity()+other_id_.capacity()+parents_.capacity())*sizeof(unsigned)+
            parent_ptr_.capacity()*sizeof(size_t)+neighbor_ptr_.capacity()*sizeof(NeighborBuckets)+
            neighbors_.capacity()*sizeof(neighbors_[0])+narrow_neighbors_.capacity()*sizeof(narrow_neighbors_[0])+
            distances_.capacity();
        n+=(main_.capacity()+other_.capacity())*sizeof(Config);
        for(auto &c:main_)n+=c.capacity()*sizeof(int);
        for(auto &c:other_)n+=c.capacity()*sizeof(int);
        return n+(tree_ ? tree_->bytes() : 0);
    }
    Stats info() const {return {{"main_strings",main_.size()},{"other_strings",other_.size()},
        {"string_connections",neighbors_.size()+narrow_neighbors_.size()},
        {"neighbor_cache_bytes",neighbor_ptr_.capacity()*sizeof(NeighborBuckets)+
            neighbors_.capacity()*sizeof(neighbors_[0])+narrow_neighbors_.capacity()*sizeof(narrow_neighbors_[0])},
        {"metadata_build_bytes",build_bytes_},{"cached_strings",main_.size()},
        {"particle_sample_postings",main_sample_visits_},{"hole_sample_postings",other_sample_visits_}};}
};

class MatrixDirect {
    const Primary &p_;
    const double *h_;
    SpatialERI eri_;
    Workers &workers_;
    OneBank bank_;
    ParticleHole strings_;
    struct Scratch {
        Connection connection;
        Config intermediate,target;
        std::vector<double> block;
        std::vector<unsigned> neighbors;
        size_t pairs=0;
    };
    std::vector<Scratch> scratch_;
    size_t metadata_=0;

    struct Factor {
        const Matrix *matrix=nullptr;
        double identity=0.;
        size_t rows=0,cols=0;
        bool valid=false;
        double at(size_t i,size_t j) const {return matrix ? matrix->at(i,j) : (i==j ? identity : 0.);}
    };
    Factor factor(const Config &ket,int p,int q,Config &bra,size_t dimension) const {
        bra=ket;
        if(p==q)return {nullptr,double(ket[p]),dimension,dimension,true};
        if(ket[p]==2 || ket[q]==0)return {};
        ++bra[p];--bra[q];const auto &m=bank_.get(ket,p,q);
        return {&m,0.,m.rows,m.cols,true};
    }
    Factor factor(const Occupation::Limb &ket,int p,int q,Occupation::Limb &bra,size_t dimension) const {
        const auto pb=std::uint64_t(1)<<p,qb=std::uint64_t(1)<<q;
        bra=ket;
        if(p==q)return {nullptr,double(bool(ket[0]&pb)+bool(ket[1]&pb)),dimension,dimension,true};
        if((ket[1]&pb) || !(ket[0]&qb))return {};
        bra[0]=(ket[0]^(qb&~ket[1]))|pb;
        bra[1]=(ket[1]&~qb)|(pb&ket[0]);
        const auto &m=bank_.at(bank_.index(ket[0],ket[1],p,q));
        return {&m,0.,m.rows,m.cols,true};
    }
    template<bool Complete>
    void add_factor(const Factor &a,const Primary::Group &bra,const Primary::Group &ket,
                    double scale,std::vector<double> &block) const {
        if(!a.valid || scale==0.)return;
        const size_t m=bra.size(),n=ket.size();
        if constexpr(Complete) {
            if(a.matrix) {
                const auto &values=a.matrix->values;
                for(size_t i=0;i<m*n;++i)block[i]+=scale*values[i];
            } else for(size_t i=0;i<std::min(m,n);++i)block[i*n+i]+=scale*a.identity;
        } else for(size_t i=0;i<m;++i)for(size_t j=0;j<n;++j)
            block[i*n+j]+=scale*a.at(bra.rank(i),ket.rank(j));
    }
    template<bool Complete>
    void add_product(const Factor &a,const Factor &b,const Primary::Group &bra,const Primary::Group &ket,
                     double scale,std::vector<double> &block) const {
        if(!a.valid || !b.valid || !b.rows || !a.rows || scale==0.)return;
        if(a.cols!=b.rows)throw std::logic_error("incomplete intermediate spin manifold");
        if(!a.matrix) {add_factor<Complete>(b,bra,ket,scale*a.identity,block);return;}
        if(!b.matrix) {add_factor<Complete>(a,bra,ket,scale*b.identity,block);return;}
        const size_t m=bra.size(),n=ket.size();
        for(size_t i=0;i<m;++i) {
            const size_t row=Complete ? i : bra.rank(i);
            double *out=block.data()+i*n;
            for(size_t t=a.matrix->ptr[row];t<a.matrix->ptr[row+1];++t) {
                const size_t mid=a.matrix->indices[t];
                const double value=scale*a.matrix->data[t];
                const double *right=b.matrix->values.data()+mid*b.cols;
                for(size_t j=0;j<n;++j)out[j]+=value*right[Complete ? j : ket.rank(j)];
            }
        }
    }
    template<bool Complete,class State>
    void spin_block(const Primary::Group &bra,const Primary::Group &ket,const State &source,
                    const State &destination,State &intermediate,State &target,Scratch &s) const {
        size_t dimension=ket.block->dimension;
        for(auto &term:s.connection.generators) {
            if(!term.two_body) {
                auto f=factor(source,term.p,term.q,target,dimension);
                if(f.valid && target==destination)add_factor<Complete>(f,bra,ket,term.coefficient,s.block);
            } else {
                auto right=factor(source,term.r,term.s,intermediate,dimension);
                if(right.valid && right.rows) {
                    auto left=factor(intermediate,term.p,term.q,target,right.rows);
                    if(left.valid && target==destination)add_product<Complete>(left,right,bra,ket,term.coefficient,s.block);
                }
                if(term.q==term.r) {
                    auto correction=factor(source,term.p,term.s,target,dimension);
                    if(correction.valid && target==destination)add_factor<Complete>(correction,bra,ket,-term.coefficient,s.block);
                }
            }
        }
    }
    void pair(size_t a,size_t b,const double *x,size_t cols,double *y,Scratch &s) const {
        auto &bra=p_.groups[a];auto &ket=p_.groups[b];
        const auto &left=bra.block->key.first,&right=ket.block->key.first;
        const auto pair=p_.k<=64 ? PairDescriptor::from_masks(*bra.cfg,*ket.cfg,a,b,left[0],left[1],right[0],right[1])
            : PairDescriptor(*bra.cfg,*ket.cfg,a,b);
        ranked_connection(pair,h_,eri_,false,s.connection);
        s.block.assign(bra.size()*ket.size(),0.);
        if(a==b)for(size_t i=0;i<bra.size();++i)s.block[i*ket.size()+i]=s.connection.scalar;
        const bool complete=bra.canonical_full() && ket.canonical_full();
        auto build_tile=[&](const auto &source,const auto &destination,auto &intermediate,auto &target) {
            if(complete)spin_block<true>(bra,ket,source,destination,intermediate,target,s);
            else spin_block<false>(bra,ket,source,destination,intermediate,target,s);
        };
        if(p_.k<=64) {
            Occupation::Limb intermediate,target;
            build_tile(right,left,intermediate,target);
        } else build_tile(*ket.cfg,*bra.cfg,s.intermediate,s.target);
        if(cols==1) {
            const double *left_x=x+bra.block->offset,*right_x=x+ket.block->offset;
            double *left_y=y+bra.block->offset,*right_y=y+ket.block->offset;
            const size_t m=bra.size(),n=ket.size();
            for(size_t i=0;i<m;++i) {
                const double *row=s.block.data()+i*n;
                double sum=0.;
                for(size_t j=0;j<n;++j)sum+=row[j]*right_x[j];
                left_y[i]+=sum;
                if(a!=b)for(size_t j=0;j<n;++j)right_y[j]+=row[j]*left_x[i];
            }
        } else for(size_t i=0;i<bra.size();++i) {
            const size_t ai=bra.row(i);
            for(size_t j=0;j<ket.size();++j) {
                const size_t bi=ket.row(j);
                double value=s.block[i*ket.size()+j];
                if(value==0.)continue;
                for(size_t v=0;v<cols;++v) {
                    y[ai*cols+v]+=value*x[bi*cols+v];
                    if(a!=b)y[bi*cols+v]+=value*x[ai*cols+v];
                }
            }
        }
        ++s.pairs;
    }
public:
    MatrixDirect(const Primary &p,const double *h,SpatialERI eri,Workers &workers,size_t budget):
        p_(p),h_(h),eri_(eri),workers_(workers),bank_(cih_template_electrons(p),p.s,budget),
        strings_(p,workers,memory_left(budget,bank_.bytes(),"particle–hole spin coefficients")),
        scratch_(workers.count()) {
        size_t largest=0;
        for(const auto &group:p.groups)largest=std::max(largest,group.size());
        metadata_=strings_.bytes();
        if(largest) {
            const size_t space=memory_left(budget,checked_add(metadata_,bank_.bytes()),"particle–hole tables");
            if(largest>space/sizeof(double)/workers_.count()/largest)
                throw BudgetError("particle–hole worker spin blocks exceed --max-mem");
            for(auto &worker:scratch_) {
                worker.block.reserve(largest*largest);
                metadata_=checked_add(metadata_,worker.block.capacity()*sizeof(double));
            }
        }
    }
    size_t entries() const {return workers_.count()==1 ? 0 : workers_.count()*p_.size();}
    Stats info() const {
        Stats s=strings_.info();
        s["metadata_build_bytes"]=std::max(metadata_,s.at("metadata_build_bytes"));
        s["metadata_bytes"]=metadata_;
        s["coefficient_bytes"]=bank_.bytes();
        s["workspace_entries_per_rhs"]=entries();return s;
    }
    void apply(const double *x,size_t cols,double *y,Stats &stats) {
        const bool use_result=workers_.count()==1;
        size_t stride=p_.size()*cols;
        std::vector<double> partial(use_result ? 0 : workers_.count()*stride,0.);
        for(auto &s:scratch_)s.pairs=0;
        auto start=Clock::now();
        auto run=[&](size_t a,size_t w) {
            auto &s=scratch_[w];auto out=use_result ? y : partial.data()+w*stride;
            strings_.query(a,p_.groups[a].block->irrep,s.neighbors);
            for(unsigned b:s.neighbors)pair(a,b,x,cols,out,s);
        };
        workers_.run(p_.groups.size(),run,8);
        stats["contraction_ns"]+=ns(start);start=Clock::now();
        if(!use_result)workers_.run(p_.size(),[&](size_t i,size_t) {
            for(size_t v=0;v<cols;++v) {
                double sum=0.;for(size_t w=0;w<workers_.count();++w)sum+=partial[w*stride+i*cols+v];
                y[i*cols+v]+=sum;
            }
        },256);
        stats["reduction_ns"]+=ns(start);
        for(const auto &s:scratch_)stats["configuration_pairs"]+=s.pairs;
        size_t bytes=0;for(auto &s:scratch_)bytes+=s.neighbors.capacity()*sizeof(unsigned)+
            (s.intermediate.capacity()+s.target.capacity())*sizeof(int)+s.connection.generators.capacity()*sizeof(GugaTerm);
        stats["worker_temporary_bytes"]=bytes;
    }
};
} }

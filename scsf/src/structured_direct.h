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
    std::vector<size_t> parent_ptr_,neighbor_ptr_;
    std::vector<unsigned> neighbors_;
    std::vector<unsigned char> neighbor_distances_,distances_;
    std::unique_ptr<OccupationTree> tree_;
    size_t fixed_bytes_=0,main_sample_visits_=0,other_sample_visits_=0;
    size_t retained_=0;
public:
    ParticleHole(const Primary &p,Workers &workers,size_t maximum,size_t neighbor_limit) {
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
        parent_ptr_.assign(main_.size()+1,0);
        for(unsigned id:main_id_)++parent_ptr_[id+1];
        std::partial_sum(parent_ptr_.begin(),parent_ptr_.end(),parent_ptr_.begin());
        parents_.resize(p.groups.size());auto cursor=parent_ptr_;
        for(size_t g=0;g<p.groups.size();++g)parents_[cursor[main_id_[g]]++]=g;
        fixed_bytes_=bytes();
        if(fixed_bytes_>maximum)throw BudgetError("particle/hole strings exceed metadata budget");
        tree_=std::make_unique<OccupationTree>(main_.size(),main_[0].size(),maximum-fixed_bytes_,
            [&](size_t i)->const Config &{return main_[i];});
        fixed_bytes_=bytes();
        if(other_.size() && other_.size()<=std::min(size_t(4096),(maximum-fixed_bytes_)/other_.size())) {
            distances_.resize(other_.size()*other_.size());
            workers.run(other_.size(),[&](size_t i,size_t) {
                for(size_t j=0;j<other_.size();++j) {
                    int d=0;for(size_t k=0;k<other_[i].size();++k)d+=std::abs(other_[i][k]-other_[j][k]);
                    distances_[i*other_.size()+j]=std::min(d,5);
                }
            });
        }
        // Spend the neighbor budget on strings queried by the most primary
        // configurations. Other strings still use the occupation tree.
        fixed_bytes_=bytes();
        const auto cache_start=fixed_bytes_;
        if(main_.size()+1>std::min(neighbor_limit,maximum-fixed_bytes_)/sizeof(size_t))return;
        neighbor_ptr_.assign(main_.size()+1,0);
        workers.run(main_.size(),[&](size_t i,size_t) {
            tree_->visit(main_[i],~0u,[&](unsigned,int){++neighbor_ptr_[i+1];});
        });
        fixed_bytes_=bytes();
        const size_t capacity=std::min(maximum-fixed_bytes_,neighbor_limit-(fixed_bytes_-cache_start))/
            (sizeof(neighbors_[0])+sizeof(neighbor_distances_[0]));
        const size_t total=std::accumulate(neighbor_ptr_.begin(),neighbor_ptr_.end(),size_t(0));
        if(total>capacity) {
            std::vector<unsigned> order(main_.size());
            std::iota(order.begin(),order.end(),0);
            auto frequency=[&](unsigned id) {return parent_ptr_[id+1]-parent_ptr_[id];};
            std::sort(order.begin(),order.end(),[&](unsigned a,unsigned b) {
                const auto first=frequency(a),second=frequency(b);
                return first!=second ? first>second : a<b;
            });
            size_t remaining=capacity;
            for(unsigned id:order) {
                auto &count=neighbor_ptr_[id+1];
                if(count>remaining)count=0;
                remaining-=count;
            }
        }
        for(auto count:neighbor_ptr_)retained_+=count!=0;
        if(!retained_) {
            std::vector<size_t>().swap(neighbor_ptr_);
            return;
        }
        std::partial_sum(neighbor_ptr_.begin(),neighbor_ptr_.end(),neighbor_ptr_.begin());
        neighbors_.resize(neighbor_ptr_.back());
        neighbor_distances_.resize(neighbors_.size());
        workers.run(main_.size(),[&](size_t i,size_t) {
            size_t at=neighbor_ptr_[i];
            if(at==neighbor_ptr_[i+1])return;
            tree_->visit(main_[i],~0u,[&](unsigned id,int d) {
                neighbors_[at]=id;
                neighbor_distances_[at++]=static_cast<unsigned char>(d);
            });
        });
    }
    void query(size_t group,std::vector<unsigned> &result) const {
        result.clear();if(main_.empty())return;
        unsigned a=main_id_[group],b=other_id_[group];
        const auto *distance_row=distances_.empty() ? nullptr : distances_.data()+b*other_.size();
        auto visit=[&](unsigned id,int distance) {
            for(size_t j=parent_ptr_[id];j<parent_ptr_[id+1];++j) {
                unsigned g=parents_[j];if(g>group)break;
                unsigned c=other_id_[g];int d=0;
                if(distance_row)d=distance_row[c];
                else for(size_t k=0;k<other_[b].size();++k)d+=std::abs(other_[b][k]-other_[c][k]);
                if(d+distance<=4)result.push_back(g);
            }
        };
        if(retained_ && neighbor_ptr_[a]!=neighbor_ptr_[a+1])
            for(size_t j=neighbor_ptr_[a];j<neighbor_ptr_[a+1];++j)visit(neighbors_[j],neighbor_distances_[j]);
        else tree_->visit(main_[a],~0u,visit);
        std::sort(result.begin(),result.end());
    }
    size_t bytes() const {
        size_t n=(main_id_.capacity()+other_id_.capacity()+parents_.capacity())*sizeof(unsigned)+
            (parent_ptr_.capacity()+neighbor_ptr_.capacity())*sizeof(size_t)+neighbors_.capacity()*sizeof(neighbors_[0])+
            neighbor_distances_.capacity()+distances_.capacity();
        n+=(main_.capacity()+other_.capacity())*sizeof(Config);
        for(auto &c:main_)n+=c.capacity()*sizeof(int);
        for(auto &c:other_)n+=c.capacity()*sizeof(int);
        return n+(tree_ ? tree_->bytes() : 0);
    }
    Stats info() const {return {{"main_strings",main_.size()},{"other_strings",other_.size()},
        {"string_connections",neighbors_.size()},{"retained_string_connections",retained_!=0},{"cached_strings",retained_},
        {"particle_sample_postings",main_sample_visits_},{"hole_sample_postings",other_sample_visits_}};}
};

class MatrixDirect {
    const Primary &p_;
    const double *h_;
    SpatialERI eri_;
    Workers &workers_;
    std::unique_ptr<OneBank> bank_;
    std::unique_ptr<OccupationTree> tree_;
    std::unique_ptr<ParticleHole> strings_;
    struct Scratch {
        Connection connection;
        Config intermediate,target;
        std::vector<double> block;
        std::vector<unsigned> neighbors;
        size_t pairs=0, row_pairs=0;
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
        ++bra[p];--bra[q];const auto &m=bank_->get(ket,p,q);
        return {&m,0.,m.rows,m.cols,true};
    }
    Factor factor(const Occupation::Limb &ket,int p,int q,Occupation::Limb &bra,size_t dimension) const {
        const auto pb=std::uint64_t(1)<<p,qb=std::uint64_t(1)<<q;
        bra=ket;
        if(p==q)return {nullptr,double(bool(ket[0]&pb)+bool(ket[1]&pb)),dimension,dimension,true};
        if((ket[1]&pb) || !(ket[0]&qb))return {};
        bra[0]=(ket[0]^(qb&~ket[1]))|pb;
        bra[1]=(ket[1]&~qb)|(pb&ket[0]);
        const auto &m=bank_->at(bank_->index(ket[0],ket[1],p,q));
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
    // Evaluate one selected row pair at a time when a complete template
    // directory or a rectangular spin tile would exceed the coefficient cap.
    void row_pair(size_t a,size_t b,const double *x,size_t cols,double *y,const Connection &connection) const {
        const auto &bra=p_.groups[a],&ket=p_.groups[b];
        for(size_t i=0;i<bra.size();++i) {
            const Path left(p_.label(a,i),p_.k,p_.s,2);
            for(size_t j=0;j<ket.size();++j) {
                const Path right(p_.label(b,j),p_.k,p_.s,2);
                const double element=guga_element(left,right,connection);
                if(element==0.)continue;
                const size_t ai=bra.row(i)*cols,bi=ket.row(j)*cols;
                for(size_t v=0;v<cols;++v) {
                    y[ai+v]+=element*x[bi+v];
                    if(a!=b)y[bi+v]+=element*x[ai+v];
                }
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
        if(!bank_ || bra.size()>8192/ket.size()) {
            row_pair(a,b,x,cols,y,s.connection);
            ++s.pairs;
            ++s.row_pairs;
            return;
        }
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
    void query(size_t a,std::vector<unsigned> &out) const {
        if(strings_)strings_->query(a,out);
        else {
            out.clear();tree_->visit(*p_.groups[a].cfg,a,[&](unsigned b,int){out.push_back(b);});
            std::sort(out.begin(),out.end());
        }
    }
public:
    MatrixDirect(const Primary &p,const double *h,SpatialERI eri,Workers &workers,
                 size_t metadata,size_t coefficients,size_t neighbors,bool row_only):p_(p),h_(h),eri_(eri),workers_(workers),scratch_(workers.count()) {
        if(!row_only)try {
            bank_=std::make_unique<OneBank>(cih_template_electrons(p),p.s,coefficients);
        } catch(const BudgetError &) {
            // Evaluate the selected spin paths directly below.
        }
        if(!row_only)strings_=std::make_unique<ParticleHole>(p,workers,metadata,neighbors);
        else tree_=std::make_unique<OccupationTree>(p.groups.size(),p.k,metadata,
            [&](size_t a)->const Config &{return *p.groups[a].cfg;});
        metadata_=strings_ ? strings_->bytes() : tree_->bytes();
    }
    size_t entries() const {return workers_.count()==1 ? 0 : workers_.count()*p_.size();}
    Stats info() const {
        Stats s=strings_ ? strings_->info() : Stats{};
        s["metadata_bytes"]=metadata_;s["coefficient_bytes"]=bank_ ? bank_->bytes() : 0;
        s["row_guga_fallback"]=!bank_;
        s["workspace_entries_per_rhs"]=entries();return s;
    }
    void apply(const double *x,size_t cols,double *y,Stats &stats,bool serial=false) {
        const bool use_result=serial || (workers_.count()==1);
        size_t stride=p_.size()*cols;
        std::vector<double> partial(use_result ? 0 : workers_.count()*stride,0.);
        for(auto &s:scratch_)s.pairs=s.row_pairs=0;
        auto start=Clock::now();
        auto run=[&](size_t a,size_t w) {
            auto &s=scratch_[w];auto out=use_result ? y : partial.data()+w*stride;
            query(a,s.neighbors);
            for(unsigned b:s.neighbors)pair(a,b,x,cols,out,s);
        };
        if(serial)for(size_t a=0;a<p_.groups.size();++a)run(a,0);
        else workers_.run(p_.groups.size(),run,8);
        stats["contraction_ns"]+=ns(start);start=Clock::now();
        if(!use_result)workers_.run(p_.size(),[&](size_t i,size_t) {
            for(size_t v=0;v<cols;++v) {
                double sum=0.;for(size_t w=0;w<workers_.count();++w)sum+=partial[w*stride+i*cols+v];
                y[i*cols+v]+=sum;
            }
        },256);
        stats["reduction_ns"]+=ns(start);
        for(const auto &s:scratch_) {
            stats["configuration_pairs"]+=s.pairs;
            stats["row_guga_pairs"]+=s.row_pairs;
        }
        size_t bytes=0;for(auto &s:scratch_)bytes+=s.block.capacity()*8+s.neighbors.capacity()*sizeof(unsigned)+
            (s.intermediate.capacity()+s.target.capacity())*sizeof(int)+s.connection.generators.capacity()*sizeof(GugaTerm);
        stats["worker_temporary_bytes"]=bytes;
    }
};
} }

/* This file is part of SCSF, licensed under GNU GPL version 3 or later.
 * Included in guga.cpp after selected GUGA, inside scsf. */
namespace {
template<bool Checkpoint>
void weighted_direction(const Config &bra_cfg,const Config &ket_cfg,const SelectedSpinTrie &trie,
    const std::vector<Path> &paths,IndexView rows,IndexView cols,const std::vector<double> &active,
    bool same,const Connection &c,bool transpose,const double *input,double *output,size_t columns,
    WeightedSpinScratch &scratch,SpinTraversalStats &stats) {
    size_t width=2*columns;
    scratch.stack.resize((ket_cfg.size()+2)*width);
    scratch.cache.resize(trie.nodes.size()*width);
    scratch.previous.resize(trie.nodes.size());scratch.live.assign(trie.nodes.size(),0);
    for (size_t at=trie.nodes.size();at--;) {
        const auto &node=trie.nodes[at];
        if (node.leaf!=SelectedSpinTrie::none) scratch.live[at]=active[cols[node.leaf]]!=0.;
        for (auto child:node.child) if (child!=SelectedSpinTrie::none) scratch.live[at]|=scratch.live[child];
    }
    // Return a vector-weighted RIGHT environment for every allowed complete
    // intermediate spin. Branch sums are formed before applying the prefix.
    auto channel=[&](int p,int q,int r,int s,bool two,double coefficient) {
        if (coefficient==0.) return;
        if (two && r==s) { coefficient*=ket_cfg[s];two=false; }
        else if (two && p==q) { coefficient*=bra_cfg[p];p=r;q=s;two=false; }
        if (!two && p==q) return; // separately stored diagonal
        if (two && (!ket_cfg[s] || ket_cfg[r]==2)) return;
        int begin=std::min(p,q),end=std::max(p,q);
        if (two) { begin=std::min({begin,r,s});end=std::max({end,r,s}); }
        for (int x=0;x<int(ket_cfg.size());++x)
            if ((x<begin || x>end) && bra_cfg[x]!=ket_cfg[x]) return;
        std::fill(scratch.previous.begin(),scratch.previous.end(),nullptr);
        ++stats.weighted_channels;
        for (size_t i:rows) {
            const auto &bra=paths[i];std::uint32_t root=0;size_t depth=0;
            while (depth<trie.sites.size() && trie.sites[depth]<begin) {
                root=trie.nodes[root].child[bra.label[trie.sites[depth]]=='d'];
                if (root==SelectedSpinTrie::none) break;
                ++depth;
            }
            if (root==SelectedSpinTrie::none) continue;
            auto visit=[&](auto &&self,int x,std::uint32_t node,size_t d) -> const double * {
                ++stats.weighted_nodes;
                double *value=scratch.stack.data()+size_t(x)*width;
                std::fill(value,value+width,0.);
                if (!scratch.live[node]) { ++stats.weighted_pruned;return value; }
                const auto &n=trie.nodes[node];
                if (x>end) {
                    if (n.spin!=bra.spins.get<Checkpoint>(end+1)) return value;
                    while (d<trie.sites.size()) {
                        node=trie.nodes[node].child[bra.label[trie.sites[d]]=='d'];
                        if (node==SelectedSpinTrie::none) return value;
                        ++d;
                    }
                    size_t j=cols[trie.nodes[node].leaf];
                    if (i!=j) for (size_t v=0;v<columns;++v) value[v]=paths[j].phase*input[j*columns+v];
                    return value;
                }
                bool open=ket_cfg[x]==1;
                // A cached suffix must not depend on which diagonal leaf was
                // excluded. Cache only subtrees whose prefix differs from bra.
                bool cacheable=open && (!same || bra.label.compare(0,x,paths[cols[n.first]].label,0,x)!=0);
                if (cacheable) {
                    const Path *old=scratch.previous[node];
                    if (old && old->spins.get<Checkpoint>(x)==bra.spins.get<Checkpoint>(x) &&
                        old->label.compare(x,std::string::npos,bra.label,x,std::string::npos)==0) {
                        ++stats.weighted_hits;return scratch.cache.data()+node*width;
                    }
                }
                int before=trie.nodes[node].spin;
                bool previous_inside=two && x>std::min(r,s) && x<=std::max(r,s);
                std::array<int,2> before_spins{previous_inside ? before-1 : before,previous_inside ? before+1 : -1};
                int before_count=previous_inside ? 2 : 1;
                for (int choice=0;choice<(open ? 2 : 1);++choice) {
                    auto child=open ? n.child[choice] : node;
                    if (child==SelectedSpinTrie::none) continue;
                    int b=trie.nodes[child].spin,db=open ? choice+1 : ket_cfg[x]==2 ? 3 : 0;
                    bool inside=two && x>=std::min(r,s) && x<std::max(r,s);
                    std::array<int,2> after{inside ? b-1 : b,inside ? b+1 : -1};
                    int after_count=inside ? 2 : 1;
                    double transfer[2][2]={{0.,0.},{0.,0.}};bool nonzero=false;
                    for (int a=0;a<before_count;++a) if (before_spins[a]>=0)
                        for (int z=0;z<after_count;++z) if (after[z]>=0) {
                            double factor;
                            if (two) {
                                int middle=ket_cfg[x]+(x==r)-(x==s),change=after[z]-before_spins[a],dm;
                                if (middle==1 && std::abs(change)==1) dm=change==1 ? 1 : 2;
                                else if (middle!=1 && change==0) dm=middle==2 ? 3 : 0;
                                else continue;
                                factor=segment(dm,db,after[z],b,r,s,x,bra.guga_mode)*
                                    segment(step(bra.label[x]),dm,bra.spins.get<Checkpoint>(x+1),after[z],p,q,x,bra.guga_mode);
                            } else factor=segment(step(bra.label[x]),db,bra.spins.get<Checkpoint>(x+1),b,p,q,x,bra.guga_mode);
                            transfer[a][z]=factor;nonzero|=factor!=0.;
                        }
                    if (!nonzero) { ++stats.weighted_pruned;continue; }
                    const double *right=self(self,x+1,child,d+open);
                    for (int a=0;a<before_count;++a) for (int z=0;z<after_count;++z)
                        if (transfer[a][z]!=0.) for (size_t v=0;v<columns;++v)
                            value[a*columns+v]+=transfer[a][z]*right[z*columns+v];
                }
                if (cacheable) {
                    std::copy(value,value+width,scratch.cache.data()+node*width);
                    scratch.previous[node]=&bra;
                }
                return value;
            };
            const double *value=visit(visit,begin,root,depth);
            for (size_t v=0;v<columns;++v) output[i*columns+v]+=coefficient*bra.phase*value[v];
        }
    };
    for (const auto &t:c.generators) {
        int p=transpose ? t.q : t.p,q=transpose ? t.p : t.q;
        int r=transpose ? t.s : t.r,s=transpose ? t.r : t.s;
        channel(p,q,r,s,t.two_body,t.coefficient);
        if (t.two_body && q==r) channel(p,s,0,0,false,-t.coefficient);
    }
}
}

bool weighted_guga_apply(const Config &bra,const Config &ket,const SelectedSpinTrie &a,const SelectedSpinTrie &b,
    const std::vector<Path> &paths,IndexView rows,IndexView cols,const std::vector<double> &active,
    bool same,const Connection &c,const double *input,double *output,size_t columns,size_t maximum,
    WeightedSpinScratch &scratch,SpinTraversalStats &stats) {
    if (!columns || rows.empty() || cols.empty() || bra.size()>1024 ||
        columns>maximum/2/(bra.size()+2) || columns>maximum/2/b.nodes.size() ||
        (!same && columns>maximum/2/a.nodes.size())) return false;
    // Reserve exact requested sizes before either direction publishes sigma.
    // resize's geometric growth could otherwise exceed the per-buffer cap
    // while moving between differently sized configuration tries.
    size_t nodes=same ? b.nodes.size() : std::max(a.nodes.size(),b.nodes.size());
    size_t stack_size=2*columns*(bra.size()+2),cache_size=2*columns*nodes;
    if (scratch.stack.capacity()<stack_size) scratch.stack.reserve(stack_size);
    if (scratch.cache.capacity()<cache_size) scratch.cache.reserve(cache_size);
    if (scratch.previous.capacity()<nodes) scratch.previous.reserve(nodes);
    if (scratch.live.capacity()<nodes) scratch.live.reserve(nodes);
    size_t peak=std::max(scratch.stack.capacity(),scratch.cache.capacity());
    if (peak>maximum) return false;
    stats.weighted_buffer_peak=std::max(stats.weighted_buffer_peak,peak);
    bool checkpoints=paths[rows[0]].spins.is_compressed();
    if (checkpoints) {
        weighted_direction<true>(bra,ket,b,paths,rows,cols,active,same,c,false,input,output,columns,scratch,stats);
        if (!same) weighted_direction<true>(ket,bra,a,paths,cols,rows,active,false,c,true,input,output,columns,scratch,stats);
    } else {
        weighted_direction<false>(bra,ket,b,paths,rows,cols,active,same,c,false,input,output,columns,scratch,stats);
        if (!same) weighted_direction<false>(ket,bra,a,paths,cols,rows,active,false,c,true,input,output,columns,scratch,stats);
    }
    ++stats.weighted_blocks;return true;
}

int hybrid_guga_apply(const Config &bra,const Config &ket,const SelectedSpinTrie &b,
    const std::vector<Path> &paths,IndexView rows,IndexView cols,const std::vector<std::uint64_t> &down,
    const std::vector<double> &active,bool same,const Connection &c,const GugaLeanPlan &plan,GugaMemo &memo,
    bool accelerated,int electrons,size_t maximum,
    GugaBatchScratch &result,HybridSpinScratch &scratch,SpinTraversalStats &stats) {
    auto is_short=[&](size_t t) {
        return bra.size()<=64 && plan.size()==c.generators.size() && memo.tiny_cacheable(plan[t],paths,rows,cols,electrons);
    };
    bool any_long=false;for (size_t t=0;t<c.generators.size();++t) if (!is_short(t)) { any_long=true;break; }
    if (!any_long) { ++stats.hybrid_short_blocks;return 0; }
    auto &short_c=scratch.short_terms;auto &long_c=scratch.long_terms;
    short_c.generators.clear();long_c.generators.clear();scratch.short_plan.clear();
    for (size_t t=0;t<c.generators.size();++t) {
        if (is_short(t)) { short_c.generators.push_back(c.generators[t]);scratch.short_plan.push_back(plan[t]); }
        else long_c.generators.push_back(c.generators[t]);
    }
    if (!short_c.generators.empty() && rows.size()>maximum/cols.size()) return 0;
    ++stats.hybrid_split_blocks;stats.hybrid_short_terms+=short_c.generators.size();stats.hybrid_long_terms+=long_c.generators.size();
    if (!selected_guga_batch(bra,ket,b,paths,rows,cols,active,same,long_c,maximum,result,stats)) return 0;
    if (short_c.generators.empty()) return 1;
    if (!memo.small_batch(paths,rows,cols,down,active,same,short_c,scratch.short_plan,accelerated,maximum,scratch.values))
        throw std::logic_error("hybrid short-generator preflight mismatch");
    for (size_t p=0;p<result.values.size();++p) result.values[p]+=scratch.values.values[p];
    return 1;
}

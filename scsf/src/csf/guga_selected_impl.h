/* This file is part of SCSF, licensed under GNU GPL version 3 or later.
 * Included by guga.cpp, inside scsf, to inline the exact segment factors. */
namespace {
template<bool Checkpoint>
bool selected_batch_impl(const Config &bra_cfg,const Config &ket_cfg,const SelectedSpinTrie &trie,
    const std::vector<Path> &paths,IndexView rows,IndexView cols,const std::vector<double> &active,
    bool same,const Connection &c,size_t maximum,GugaBatchScratch &scratch,SpinTraversalStats &stats) {
    if (rows.empty() || cols.empty() || rows.size()>maximum/cols.size()) return false;
    // Keep recursive stack bounded for unusually long orbital lists.
    if (ket_cfg.size()>1024) return false;
    auto &out=scratch.values;out.assign(rows.size()*cols.size(),0.);
    std::vector<unsigned char> live(trie.nodes.size(),0);
    for (size_t at=trie.nodes.size();at--;) {
        const auto &node=trie.nodes[at];
        if (node.leaf!=SelectedSpinTrie::none) live[at]=active[cols[node.leaf]]!=0.;
        for (auto child:node.child) if (child!=SelectedSpinTrie::none) live[at]|=live[child];
    }
    stats.scratch_peak=std::max(stats.scratch_peak,live.capacity()*sizeof(live[0]));
    ++stats.blocks;
    for (size_t row=0;row<rows.size();++row) {
        size_t i=rows[row];const auto &bra=paths[i];
        auto channel=[&](int p,int q,int r,int s,bool two,double coefficient) {
            if (coefficient==0.) return;
            if (two && r==s) { coefficient*=ket_cfg[s];two=false; }
            else if (two && p==q) { coefficient*=bra_cfg[p];p=r;q=s;two=false; }
            // E_pp contributes only the already separately stored diagonal.
            if (!two && p==q) return;
            if (two && (!ket_cfg[s] || ket_cfg[r]==2)) return;
            int begin=std::min(p,q),end=std::max(p,q);
            if (two) { begin=std::min({begin,r,s});end=std::max({end,r,s}); }
            for (int x=0;x<int(ket_cfg.size());++x)
                if ((x<begin || x>end) && bra_cfg[x]!=ket_cfg[x]) return;
            std::uint32_t root=0;size_t depth=0;
            while (depth<trie.sites.size() && trie.sites[depth]<begin) {
                root=trie.nodes[root].child[bra.label[trie.sites[depth]]=='d'];
                if (root==SelectedSpinTrie::none) return;
                ++depth;
            }
            ++stats.channels;
            auto visit=[&](auto &&self,int x,std::uint32_t node,size_t d,
                           std::array<int,2> spins,std::array<double,2> weights,int count) -> void {
                ++stats.nodes;
                const auto &n=trie.nodes[node];
                if ((active[i]==0. && !live[node]) || (same && cols[n.first]>=i)) { ++stats.pruned;return; }
                if (x>end) {
                    if (n.spin!=bra.spins.get<Checkpoint>(end+1)) { ++stats.pruned;return; }
                    while (d<trie.sites.size()) {
                        node=trie.nodes[node].child[bra.label[trie.sites[d]]=='d'];
                        if (node==SelectedSpinTrie::none) { ++stats.pruned;return; }
                        ++d;
                    }
                    size_t col=trie.nodes[node].leaf,j=cols[col];
                    if (i==j || (same && j>i) || (active[i]==0. && active[j]==0.)) return;
                    ++stats.leaves;
                    out[row*cols.size()+col]+=coefficient*bra.phase*paths[j].phase*weights[0];
                    return;
                }
                bool open=ket_cfg[x]==1;
                for (int choice=0;choice<(open ? 2 : 1);++choice) {
                    auto next_node=open ? n.child[choice] : node;
                    if (next_node==SelectedSpinTrie::none) continue;
                    int b=trie.nodes[next_node].spin,db=open ? choice+1 : ket_cfg[x]==2 ? 3 : 0;
                    std::array<double,2> next{0.,0.};std::array<int,2> after{b,-1};int ncount=1;
                    if (two) {
                        bool inside=x>=std::min(r,s) && x<std::max(r,s);
                        after={inside ? b-1 : b,inside ? b+1 : -1};ncount=inside ? 2 : 1;
                        int occupation_middle=ket_cfg[x]+(x==r)-(x==s);
                        for (int a=0;a<ncount;++a) if (after[a]>=0) for (int z=0;z<count;++z) {
                            if (weights[z]==0.) continue;
                            int change=after[a]-spins[z],dm;
                            if (occupation_middle==1 && std::abs(change)==1) dm=change==1 ? 1 : 2;
                            else if (occupation_middle!=1 && change==0) dm=occupation_middle==2 ? 3 : 0;
                            else continue;
                            double right=segment(dm,db,after[a],b,r,s,x,bra.guga_mode);
                            if (right==0.) continue;
                            next[a]+=weights[z]*right*segment(step(bra.label[x]),dm,
                                bra.spins.get<Checkpoint>(x+1),after[a],p,q,x,bra.guga_mode);
                        }
                    } else next[0]=weights[0]*segment(step(bra.label[x]),db,
                        bra.spins.get<Checkpoint>(x+1),b,p,q,x,bra.guga_mode);
                    if (next[0]==0. && next[1]==0.) { ++stats.pruned;continue; }
                    self(self,x+1,next_node,d+open,after,next,ncount);
                }
            };
            visit(visit,begin,root,depth,{trie.nodes[root].spin,-1},{1.,0.},1);
        };
        for (const auto &t:c.generators) {
            channel(t.p,t.q,t.r,t.s,t.two_body,t.coefficient);
            if (t.two_body && t.q==t.r) channel(t.p,t.s,0,0,false,-t.coefficient);
        }
    }
    return true;
}
} // namespace

bool selected_guga_batch(const Config &bra,const Config &ket,const SelectedSpinTrie &trie,
    const std::vector<Path> &paths,IndexView rows,IndexView cols,const std::vector<double> &active,
    bool same,const Connection &c,size_t maximum,GugaBatchScratch &scratch,SpinTraversalStats &stats) {
    bool checkpoints=!rows.empty() && paths[rows[0]].spins.is_compressed();
    return checkpoints ? selected_batch_impl<true>(bra,ket,trie,paths,rows,cols,active,same,c,maximum,scratch,stats) :
        selected_batch_impl<false>(bra,ket,trie,paths,rows,cols,active,same,c,maximum,scratch,stats);
}

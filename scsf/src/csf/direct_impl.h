/* Direct Hamiltonian implementation, included inside sci.cpp's anonymous
 * namespace to share the validated occupation index and coupling machinery.
 * This file is part of SCSF, licensed under GNU GPL version 3 or later. */
class DirectHamiltonian {
    using Groups = SelectedBasis::Groups;
    using Group = Groups::value_type;
    using SelectedIndex = ConfigurationIndex<Groups>;
    BasisSnapshot basis_;
    Array one_, two_;
    int k_, two_s_;
    size_t threads_, tensors_, terms_, bonds_, reuse_bytes_;
    enum Mode { CompactSpin, CompactCache, Adaptive, WideSpin, SemiDirect, Tuga, Selected, Hybrid, Weighted };
    Mode mode_;
    int factor_mode_;
    bool spin_checkpoints_;
    bool reuse_, packed_, cached_paths_ = false;
    double tolerance_, cutoff_;
    const Groups &groups_;
    const std::vector<const Group *> &order_;
    std::unique_ptr<SelectedIndex> index_;
    std::vector<Path> paths_;
    std::vector<double> diagonal_;
    struct OccupationKey {
        std::array<std::uint64_t,2> words{};
        bool operator==(const OccupationKey &b) const { return words==b.words; }
    };
    struct OccupationHash {
        size_t operator()(const OccupationKey &key) const {
            auto a=key.words[0], b=key.words[1];
            return std::hash<std::uint64_t>{}(a^(b+0x9e3779b97f4a7c15ULL+(a<<6)+(a>>2)));
        }
    };
    const std::unordered_map<const Group *,size_t> &ordinal_;
    std::unordered_map<OccupationKey,size_t,OccupationHash> wide_groups_;
    std::vector<size_t> compact_ptr_;
    std::vector<std::uint32_t> compact_cols_;
    bool compact_ready_=false;
    size_t max_ordered_, setup_queries_=0;
    size_t construction_index_mode_=0,construction_index_bytes_=0,construction_index_entries_=0;
    size_t max_hamiltonian_, cached_rows_=0;
    bool profile_couplings_,collect_spin_;
    std::vector<size_t> value_offsets_;
    std::vector<double> values_;
    std::vector<std::uint64_t> path_down_;
    size_t max_batch_,min_batch_,min_batch_terms_;
    size_t path_bytes_ = 0, graph_bytes_ = 0;
    std::vector<std::unique_ptr<GugaMemo>> memo_;
    std::vector<std::unique_ptr<SymbolicTiles>> symbolic_;
    std::vector<std::unique_ptr<SelectedSpinTrie>> spin_tries_;
    size_t trie_bytes_=0,trie_nodes_=0,trie_groups_=0;
    bool hybrid_small_sector_=false;
    struct ConfigMasks { std::uint64_t singles=0,doubles=0; };
    std::vector<ConfigMasks> rank_masks_;
    std::vector<CompactSpinPath> compact_spins_;
    void prepare_compact_spins(size_t maximum) {
        static_assert(sizeof(CompactSpinPath)==48,"compact spin summary layout");
        if (!compact_mode() || !cached_paths_ || k_>64 || paths_.size()>maximum/sizeof(CompactSpinPath)) return;
        compact_spins_.resize(paths_.size());
        if (compact_spins_.capacity()>maximum/sizeof(CompactSpinPath)) {
            decltype(compact_spins_)().swap(compact_spins_);return;
        }
        for (size_t i=0;i<paths_.size();++i) {
            auto &s=compact_spins_[i];const auto &p=paths_[i];
            s.phase=p.phase;s.checkpoint=p.spins.is_compressed();
            for (int site=0;site<=k_;++site) {
                int spin=p.spins[site];
                if (spin>15) { s.fits=false;break; }
                s.words[site/16]|=std::uint64_t(spin)<<(4*(site%16));
            }
        }
    }
    void prepare_rank_masks(size_t maximum) {
        if (!compact_mode() || k_>64 || order_.size()>maximum/sizeof(ConfigMasks)) return;
        rank_masks_.resize(order_.size());
        if (rank_masks_.capacity()>maximum/sizeof(ConfigMasks)) { decltype(rank_masks_)().swap(rank_masks_);return; }
        for (size_t g=0;g<order_.size();++g) for (int p=0;p<k_;++p) {
            if (order_[g]->first[p]==1) rank_masks_[g].singles|=std::uint64_t(1)<<p;
            if (order_[g]->first[p]==2) rank_masks_[g].doubles|=std::uint64_t(1)<<p;
        }
    }
    PairDescriptor describe_pair(size_t g,size_t bg) const {
        if (rank_masks_.empty()) return PairDescriptor(order_[g]->first,order_[bg]->first,g,bg);
        const auto &a=rank_masks_[g],&b=rank_masks_[bg];
        return PairDescriptor::from_masks(order_[g]->first,order_[bg]->first,g,bg,a.singles,a.doubles,b.singles,b.doubles);
    }
    Stats last_;
    mutable std::mutex mutex_;
    std::unique_ptr<ThreadTeam> product_team_;
    size_t team_workers_=0,output_workspace_size_=0;
    std::unique_ptr<double[]> output_workspace_;
    std::vector<Connection> retained_connections_;
    std::vector<GugaPlan> retained_plans_;
    std::vector<GugaBitPlan> retained_bits_;
    std::vector<GugaLeanPlan> retained_small_;
    std::vector<GugaBatchScratch> retained_batches_;
    bool retained_product() const { return (mode_==Adaptive); }
    size_t workspace_bytes() const {
        size_t bytes=output_workspace_size_*sizeof(double);
        for (const auto &c:retained_connections_) bytes+=c.generators.capacity()*sizeof(GugaTerm);
        for (const auto &p:retained_plans_) bytes+=p.capacity()*sizeof(GugaPreparedTerm);
        for (const auto &p:retained_bits_) bytes+=p.capacity()*sizeof(GugaBitTerm);
        for (const auto &p:retained_small_) bytes+=p.capacity()*sizeof(GugaLeanTerm);
        for (const auto &b:retained_batches_) bytes+=b.bytes();
        return bytes;
    }
    template<class F> void product_for(size_t tasks,size_t workers,F job,bool fixed=false) {
        if (!retained_product()) {
            if (fixed) parallel_for(workers,workers,[&](size_t w,size_t) {
                for (size_t t=w;t<tasks;t+=workers) job(t,w);
            });
            else parallel_for(tasks,workers,job);
            return;
        }
        if (team_workers_!=workers) {
            product_team_=std::make_unique<ThreadTeam>(workers);team_workers_=workers;
        }
        std::atomic<size_t> next{0};
        product_team_->apply(workers,[&](size_t w) {
            if (fixed) { for (size_t t=w;t<tasks;t+=workers) job(t,w); }
            else for (;;) {
                size_t t=next.fetch_add(1,std::memory_order_relaxed);
                if (t>=tasks) break;
                job(t,w);
            }
        });
    }

    bool compact_mode() const { return mode_==CompactSpin || mode_==CompactCache || mode_==Adaptive; }
    bool symbolic() const { return mode_==Tuga; }

    // Reorder existing adjacency in bounded windows; retain no second edge
    // list. Full symbolic equality is still checked by the coefficient cache.

    struct GeneratorOmission { long double bound=0.;size_t removed=0; };
    static GeneratorOmission screen_generators(Connection &c,double weight,double tolerance,int policy) {
        GeneratorOmission result;size_t kept=0;
        const long double infinity=std::numeric_limits<long double>::infinity();
        const long double limit=std::nextafter(static_cast<long double>(tolerance)/weight,0.L);
        // The termwise policy only needs three thresholds per occupation block.
        double scale=tolerance/weight;
        const double two=scale/2.,four=scale/4.,six=scale/6.;
        for (size_t at=0;at<c.generators.size();++at) {
            const auto &t=c.generators[at];double magnitude=std::abs(t.coefficient);
            int norm=t.two_body ? 4+2*(t.q==t.r) : 2;
            bool drop;
            if (policy==0) {
                // Reject obviously large terms using a double precheck before
                // paying for long-double budget accumulation/outward rounding.
                drop=false;
                if (magnitude<(norm==2 ? two : norm==4 ? four : six)) {
                    long double candidate=std::nextafter(result.bound+static_cast<long double>(magnitude)*norm,infinity);
                    drop=candidate<limit;
                    if (drop) result.bound=candidate;
                }
            } else {
                drop=magnitude<(norm==2 ? two : norm==4 ? four : six);
                if (drop) result.bound+=static_cast<long double>(magnitude)*norm;
            }
            if (drop) ++result.removed;
            else { if (kept!=at) c.generators[kept]=t;++kept; }
        }
        c.generators.resize(kept);
        if (policy && result.removed) {
            // Positive-sum roundoff certificate; coefficient*norm is evaluated
            // in long double. Input doubles' subnormals are normal here.
            long double error=(result.removed+2.L)*std::numeric_limits<long double>::epsilon();
            result.bound=error<.25L ? std::nextafter(result.bound/(1.L-error),infinity) : infinity;
        }
        return result;
    }

    static OccupationKey wide_key(const Config &c) {
        OccupationKey result;
        for (size_t p=0;p<c.size();++p)
            result.words[p/32]|=std::uint64_t(c[p])<<(2*(p%32));
        return result;
    }

    std::vector<const Group *> fast_partners(const Group &a,size_t last,Stats &stats) const {
        if (index_->mode() || k_>64) return partners(a,last,stats);
        size_t occupied=0,doubles=0;
        for (int n:a.first) { occupied+=n!=0; doubles+=n==2; }
        long double removals=doubles+static_cast<long double>(occupied)*(occupied ? occupied-1 : 0)/2;
        long double upper=1.L+static_cast<long double>(occupied)*k_+removals*k_*(k_+1.L)/2;
        if (upper>=groups_.size() || upper>100000) return partners(a,last,stats);
        ++stats["configuration_queries"];
        std::vector<const Group *> result;
        auto key=wide_key(a.first);
        transfers(a.first,static_cast<size_t>(upper),[&](const auto &holes,const auto &particles,int n) {
            auto target=key;
            for (int p=0;p<n;++p) {
                target.words[holes[p]/32]-=std::uint64_t(1)<<(2*(holes[p]%32));
                target.words[particles[p]/32]+=std::uint64_t(1)<<(2*(particles[p]%32));
            }
            ++stats["configuration_lookups"];
            auto it=wide_groups_.find(target);
            if (it!=wide_groups_.end() && order_[it->second]->second.front()<=last)
                result.push_back(order_[it->second]);
        });
        return result;
    }

    std::vector<const Group *> graph_partners(size_t g,Stats &stats) const {
        auto found=fast_partners(*order_[g],std::numeric_limits<size_t>::max(),stats);
        found.erase(std::remove_if(found.begin(),found.end(),[&](const auto *p) {
            return ordinal_.at(p)>g;
        }),found.end());
        return found;
    }

    void build_compact(size_t maximum) {
        if (order_.size()>std::numeric_limits<std::uint32_t>::max() || maximum/sizeof(size_t)<order_.size()+1) return;
        compact_ptr_.resize(order_.size()+1,0);
        parallel_for(order_.size(),threads_,[&](size_t g,size_t) {
            Stats ignored;
            compact_ptr_[g+1]=graph_partners(g,ignored).size();
        });
        setup_queries_+=order_.size();
        size_t pointer_bytes=compact_ptr_.capacity()*sizeof(size_t);
        if (pointer_bytes>maximum) { decltype(compact_ptr_)().swap(compact_ptr_); return; }
        size_t limit=(maximum-pointer_bytes)/sizeof(std::uint32_t);
        for (size_t g=0;g<order_.size();++g) {
            if (compact_ptr_[g+1]>limit-compact_ptr_[g]) {
                decltype(compact_ptr_)().swap(compact_ptr_); return;
            }
            compact_ptr_[g+1]+=compact_ptr_[g];
        }
        compact_cols_.resize(compact_ptr_.back());
        if (pointer_bytes+compact_cols_.capacity()*sizeof(std::uint32_t)>maximum) {
            decltype(compact_ptr_)().swap(compact_ptr_);
            decltype(compact_cols_)().swap(compact_cols_); return;
        }
        parallel_for(order_.size(),threads_,[&](size_t g,size_t) {
            Stats ignored;
            auto found=graph_partners(g,ignored);
            if (found.size()!=compact_ptr_[g+1]-compact_ptr_[g])
                throw std::logic_error("direct compact graph count/fill mismatch");
            for (size_t p=0;p<found.size();++p)
                compact_cols_[compact_ptr_[g]+p]=static_cast<std::uint32_t>(ordinal_.at(found[p]));
        });
        setup_queries_+=order_.size();
        graph_bytes_=pointer_bytes+compact_cols_.capacity()*sizeof(std::uint32_t);
        compact_ready_=true;
    }

    size_t value_cache_bytes() const {
        return value_offsets_.capacity()*sizeof(size_t)+values_.capacity()*sizeof(double);
    }
    void build_value_cache() {
        // This explicitly semi-direct cache owns complete occupation rows.
        // Their dense rectangular blocks need no per-element column indices.
        if (mode_!=SemiDirect || !compact_ready_ || !max_hamiltonian_ || order_.size()>max_hamiltonian_/sizeof(size_t)) return;
        value_offsets_.resize(order_.size(),std::numeric_limits<size_t>::max());
        size_t metadata=value_offsets_.capacity()*sizeof(size_t);
        if (metadata>max_hamiltonian_) { decltype(value_offsets_)().swap(value_offsets_); return; }
        size_t limit=(max_hamiltonian_-metadata)/sizeof(double);
        struct Candidate { size_t g,count,degree; };
        std::vector<Candidate> candidates;
        for (size_t g=0;g<order_.size();++g) {
            size_t count=0, ma=order_[g]->second.size();
            bool fits=true;
            for (size_t at=compact_ptr_[g];at<compact_ptr_[g+1];++at) {
                size_t mb=order_[compact_cols_[at]]->second.size();
                if (mb>(limit-count)/ma) { fits=false; break; }
                count+=ma*mb;
            }
            if (fits && count) candidates.push_back({g,count,compact_ptr_[g+1]-compact_ptr_[g]});
        }
        // Prioritize connection preparation avoided per cached value. This
        // static policy is independent of trial coefficients and screening.
        std::sort(candidates.begin(),candidates.end(),[](const auto &a,const auto &b) {
            long double left=static_cast<long double>(a.degree)*b.count;
            long double right=static_cast<long double>(b.degree)*a.count;
            return left!=right ? left>right : a.g<b.g;
        });
        size_t count=0;
        std::vector<size_t> selected;
        for (const auto &c:candidates) if (c.count<=limit-count) {
            value_offsets_[c.g]=count; count+=c.count;
            selected.push_back(c.g);
        }
        if (selected.empty()) { decltype(value_offsets_)().swap(value_offsets_); return; }
        values_.resize(count,0.);
        if (value_cache_bytes()>max_hamiltonian_) {
            decltype(value_offsets_)().swap(value_offsets_); decltype(values_)().swap(values_); return;
        }
        std::vector<GugaPlan> plans(threads_);
        parallel_for(selected.size(),threads_,[&](size_t task,size_t w) {
            size_t g=selected[task], offset=value_offsets_[g];
            const auto &a=*order_[g];
            std::vector<Path> local_a;
            std::vector<const Path *> pa;
            if (!cached_paths_) pa=get_paths(a.second,local_a);
            for (size_t at=compact_ptr_[g];at<compact_ptr_[g+1];++at) {
                size_t bg=compact_cols_[at]; const auto &b=*order_[bg];
                auto c=connection(PairDescriptor(a.first,b.first,g,bg),one_.data(),two_.data(),terms_,true,false);
                prepare_guga_plan(a.first,b.first,c,plans[w]);
                std::vector<Path> local_b;
                std::vector<const Path *> pb;
                if (!cached_paths_) pb=get_paths(b.second,local_b);
                for (size_t r=0;r<a.second.size();++r) for (size_t t=0;t<b.second.size();++t) {
                    size_t i=a.second[r],j=b.second[t];
                    if (i==j || (g==bg && j>i)) continue;
                    const Path &p=cached_paths_ ? paths_[i] : *pa[r];
                    const Path &q=cached_paths_ ? paths_[j] : *pb[t];
                    check_bonds(p,q,bonds_);
                    values_[offset+r*b.second.size()+t]=memo_[w]->planned_element(p,q,c,plans[w]);
                }
                offset+=a.second.size()*b.second.size();
            }
        });
        cached_rows_=selected.size();
    }

    static size_t path_bytes(const Path &p) {
        return sizeof(Path)+p.label.capacity()+1+p.sites.capacity()*sizeof(Site)+
            (p.spins.capacity()+p.next_single.capacity()+p.closed_prefix.capacity())*sizeof(int);
    }
    std::vector<const Path *> get_paths(IndexView ids,
                                        std::vector<Path> &local) const {
        std::vector<const Path *> result;
        result.reserve(ids.size());
        if (cached_paths_) {
            for (size_t i : ids) result.push_back(&paths_[i]);
        } else {
            size_t budget = tensors_/threads_;
            local.reserve(ids.size());
            std::string scratch;
            for (size_t i : ids) local.emplace_back(basis_->label(i,scratch),k_,two_s_,budget,false,factor_mode_,spin_checkpoints_);
            for (const auto &p : local) result.push_back(&p);
        }
        return result;
    }
    std::vector<const Group *> partners(const Group &a, size_t last, Stats &stats) const {
        ++stats["configuration_queries"];
        std::vector<const Group *> found;
        if (index_->mode()) found=index_->query(a.first,last,stats);
        else {
            size_t occupied=0, doubles=0;
            for (int n : a.first) { occupied+=n!=0; doubles+=n==2; }
            long double removals=doubles+static_cast<long double>(occupied)*(occupied ? occupied-1 : 0)/2;
            long double upper=1.L+static_cast<long double>(occupied)*k_+removals*k_*(k_+1.L)/2;
            if (upper<groups_.size() && upper<=100000) {
                Config target=a.first;
                transfers(a.first,static_cast<size_t>(upper),[&](const auto &holes,const auto &particles,int n) {
                    ++stats["configuration_lookups"];
                    for (int p=0;p<n;++p) { --target[holes[p]]; ++target[particles[p]]; }
                    auto it=groups_.find(target);
                    if (it!=groups_.end()) found.push_back(&*it);
                    for (int p=0;p<n;++p) { ++target[holes[p]]; --target[particles[p]]; }
                });
            } else for (const auto &b : groups_) found.push_back(&b);
        }
        found.erase(std::remove_if(found.begin(),found.end(),[&](const auto *b) {
            if (b->second.front()>last) return true;
            if (index_->mode()==3) return false; // Hash index verified exact endpoint distance.
            int distance=0;
            for (int p=0;p<k_;++p) distance+=std::abs(a.first[p]-b->first[p]);
            return distance>4;
        }),found.end());
        return found;
    }
    Stats info_unlocked() const {
        size_t group_bytes=basis_->bytes_estimate();
        // Hash node/allocator overhead is an estimate, separately from the
        // exact vector payloads. Include the optional <=32-orbital lookup.
         {
            group_bytes+=wide_groups_.bucket_count()*sizeof(void *)+
                         wide_groups_.size()*(sizeof(OccupationKey)+3*sizeof(void *));
        }
        size_t bytes=diagonal_.capacity()*sizeof(double)+path_bytes_+group_bytes;
        size_t memo_bytes=0,symbolic_bytes=0,symbolic_peak=0,symbolic_entries=0,symbolic_coefficients=0,symbolic_scratch=0;
        for (const auto &m : memo_) memo_bytes+=m->info().at("reuse_bytes");
        for (const auto &s : symbolic_) {
            symbolic_bytes+=s->bytes();symbolic_peak+=s->peak();
            symbolic_entries+=s->entries();symbolic_coefficients+=s->coefficients();
            symbolic_scratch+=s->scratch_peak();
        }

        return {{"basis",basis_->size()},{"configurations",groups_.size()},
            {"shared_basis_bytes_estimate",basis_->bytes_estimate()},
            {"basis_snapshot_id",size_t(basis_->context().snapshot_id)},
            {"basis_generation",basis_->context().generation},{"pair_descriptor_cache_bytes",0},
            {"diagonal_bytes",diagonal_.capacity()*sizeof(double)},
            {"path_bytes",path_bytes_},{"paths_cached",cached_paths_},{"spin_checkpoints",spin_checkpoints_},
            {"packed_basis",basis_->packed()},
            {"occupation_metadata_bytes_estimate",group_bytes},
            {"matrix_index",size_t(index_->mode())},{"matrix_index_bytes",index_->bytes()},
            {"construction_index_mode",construction_index_mode_},
            {"construction_index_bytes",construction_index_bytes_},
            {"construction_index_entries",construction_index_entries_},
            {"graph_ready",compact_ready_},{"graph_bytes",graph_bytes_},
            {"compact_graph",compact_ready_},{"graph_edges",compact_cols_.size()},
            {"unique_occupation_blocks",true},
            {"graph_setup_queries",setup_queries_},{"packed_key_words",k_<=64 ? 2 : 0},

            {"retained_product_workspace_bytes",workspace_bytes()},
            {"retained_product_team_workers",team_workers_},
            {"spin_budget_bytes",reuse_bytes_},{"spin_budget_bytes_per_worker",reuse_bytes_/threads_},
            {"persistent_bytes_estimate",workspace_bytes()+bytes+index_->bytes()+graph_bytes_+memo_bytes+symbolic_bytes+trie_bytes_+rank_masks_.capacity()*sizeof(ConfigMasks)+compact_spins_.capacity()*sizeof(CompactSpinPath)+value_cache_bytes()+path_down_.capacity()*sizeof(std::uint64_t)},
            {"selected_trie_bytes",trie_bytes_},{"selected_trie_nodes",trie_nodes_},
            {"selected_trie_groups",trie_groups_},
            {"hybrid_small_sector",hybrid_small_sector_},
            {"rank_mask_bytes",rank_masks_.capacity()*sizeof(ConfigMasks)},{"ranked_preparation",compact_mode()},
            {"mask_preparation",compact_mode() && !rank_masks_.empty()},
            {"exact_specialization",compact_mode()},
            {"aligned_tiny_cache",[&] { size_t n=0;for (const auto &m:memo_) n+=m->tiny_aligned();return n; }()},
            {"tiny_cache_line_aligned",[&] { return std::all_of(memo_.begin(),memo_.end(),[](const auto &m) { return m->tiny_line_aligned(); }); }()},
            {"paired_spin_encoding",(mode_==CompactCache)},
            {"compact_spin_bytes",compact_spins_.capacity()*sizeof(CompactSpinPath)},
            {"compact_spin_paths",compact_spins_.size()},

            {"symbolic_bytes",symbolic_bytes},{"symbolic_peak_bytes",symbolic_peak},
            {"symbolic_entries",symbolic_entries},{"symbolic_coefficients",symbolic_coefficients},
            {"symbolic_scratch_peak_bytes",symbolic_scratch},
            {"memo_bytes",memo_bytes},{"hamiltonian_offdiagonal_bytes",values_.capacity()*sizeof(double)},
            {"hamiltonian_cache_bytes",value_cache_bytes()},{"hamiltonian_cache_rows",cached_rows_},
            {"semi_direct",(mode_==SemiDirect)},{"prepared_guga",true},
            {"profile_couplings",profile_couplings_},{"path_bit_bytes",path_down_.capacity()*sizeof(std::uint64_t)},
            {"bmi2_available",guga_bmi2_available()},{"bmi2_enabled",(mode_!=SemiDirect) && k_<=64 && guga_bmi2_available()},
            {"max_batch_elements",max_batch_},{"batch_min_pairs",min_batch_},{"batch_min_terms",min_batch_terms_},
            {"compact_spin_cache",(mode_!=SemiDirect) && k_<=64 && packed_},
            {"spin_cache_ways",(mode_!=SemiDirect) && k_<=64 && packed_ ? 2 : 1},
            {"spin_slot_bytes",mode_==SemiDirect ? 64 : mode_==WideSpin ? 24 : 16},
            {"prepared_term_bytes",(mode_!=WideSpin && mode_!=SemiDirect) ? sizeof(GugaLeanTerm) : sizeof(GugaPreparedTerm)+sizeof(GugaBitTerm)},
            {"generator_screening_available",(mode_!=WideSpin && mode_!=SemiDirect)},
            {"spin_statistics",(mode_==SemiDirect) || collect_spin_},
            {"memo_slots",[&] { size_t n=0; for (const auto &m:memo_) n+=m->info().at("reuse_slots"); return n; }()}};
    }
public:
    DirectHamiltonian(const Array &one,const Array &two,BasisSnapshot basis,
                      size_t threads,int mode,double tolerance,double cutoff,
                      int matrix_index,size_t max_index,size_t max_graph,size_t max_paths,
                      size_t max_tensor,size_t max_terms,size_t max_bonds,int factors,
                      bool reuse,size_t reuse_bytes,bool packed,size_t max_ordered,
                      size_t max_hamiltonian,bool profile_couplings,size_t max_batch,size_t min_batch,size_t min_batch_terms,bool collect_spin,bool spin_checkpoints,size_t max_symbolic,size_t max_trie)
        : basis_(checked_basis(std::move(basis))),one_(one),two_(two),
          k_(validate_integrals(one,two)),two_s_(basis_->context().two_s),
          threads_(threads),tensors_(max_tensor),terms_(max_terms),bonds_(max_bonds),
          reuse_bytes_(reuse_bytes),mode_(static_cast<Mode>(mode)),factor_mode_(factors),spin_checkpoints_(spin_checkpoints),reuse_(reuse),packed_(packed),
          tolerance_(tolerance),cutoff_(cutoff),groups_(basis_->groups()),order_(basis_->order()),
          ordinal_(basis_->ordinals()),max_ordered_(max_ordered),
          max_hamiltonian_(max_hamiltonian),profile_couplings_(profile_couplings),collect_spin_(collect_spin),
          max_batch_(max_batch),min_batch_(min_batch),min_batch_terms_(min_batch_terms) {
        validate_workers(threads);
        if (k_!=basis_->context().nbasis) throw std::invalid_argument("Hamiltonian and basis orbital counts differ");
        if (two_s_<0 || mode<0 || mode>8 || matrix_index<0 || matrix_index>3 || factors<0 || factors>2)
            throw std::invalid_argument("invalid direct Hamiltonian mode/sector/index");
        if (!std::isfinite(tolerance) || tolerance<0 || !std::isfinite(cutoff) || cutoff<0)
            throw std::invalid_argument("direct tolerances must be finite and nonnegative");
        for (auto a : {&one_, &two_}) for (py::ssize_t p=0;p<a->size();++p)
            if (!std::isfinite(a->data()[p])) throw std::invalid_argument("nonfinite integrals");

        py::gil_scoped_release release;
        try {
            if (mode_!=SemiDirect && k_<=64) {
                path_down_.resize(basis_->size(),0);
                std::string scratch;
                for (size_t i=0;i<basis_->size();++i) {
                    const auto &label=basis_->label(i,scratch);
                    for (int p=0;p<k_;++p) if (label[p]=='d') path_down_[i]|=std::uint64_t(1)<<p;
                }
            }
            index_=std::make_unique<SelectedIndex>(groups_,matrix_index,max_index);
            construction_index_mode_=index_->mode();construction_index_bytes_=index_->bytes();
            construction_index_entries_=index_->entries();

             {
                if (k_<=64) wide_groups_.reserve(order_.size());
                for (size_t g=0;g<order_.size();++g) {
                    if (k_<=64) wide_groups_.emplace(wide_key(order_[g]->first),g);
                }
            }

            // A capped path cache has a safe on-the-fly fallback. Charge actual
            // capacities, including Path objects, before publishing the cache.
            if (basis_->size()<=max_paths/std::max(size_t(1),sizeof(Path))) {
                paths_.reserve(basis_->size());
                size_t budget=max_tensor;
                std::string scratch;
                for (size_t i=0;i<basis_->size();++i) {
                    const auto &s=basis_->label(i,scratch);
                    size_t entries=(size_t(k_)+1)*(factors==2 ? 3 : 1);
                    // A per-call tensor work limit must not become an
                    // accidental whole-selected-space allocation limit.
                    if (entries>budget) break;
                    paths_.emplace_back(s,k_,two_s_,budget,false,factors,spin_checkpoints_);
                    path_bytes_+=path_bytes(paths_.back());
                    if (path_bytes_>max_paths) break;
                }
                cached_paths_=paths_.size()==basis_->size() && path_bytes_<=max_paths;
                if (!cached_paths_) { std::vector<Path>().swap(paths_); path_bytes_=0; }
            }
            // The measured <=8-electron molecular sectors fit the tiny keys.
            // Bypass trie metadata there when that memo exists. Unexpected
            // unsupported terms still have the complete scalar recurrence.
            hybrid_small_sector_=((mode_==Hybrid)) && k_<=64 && basis_->context().nelec<=8 &&
                reuse_ && packed_ && reuse_bytes_/threads_>=16;
            if ((mode_==Selected || ((mode_==Hybrid || mode_==Weighted) && !hybrid_small_sector_)) && cached_paths_ && order_.size()<=max_trie/sizeof(spin_tries_[0])) {
                spin_tries_.resize(order_.size());
                trie_bytes_=spin_tries_.capacity()*sizeof(spin_tries_[0]);
                for (size_t g=0;g<order_.size();++g) if (order_[g]->second.size()>1) {
                    auto trie=SelectedSpinTrie::build(order_[g]->first,order_[g]->second,paths_,max_trie-trie_bytes_);
                    if (!trie) continue;
                    trie_bytes_+=trie->bytes();trie_nodes_+=trie->nodes.size();++trie_groups_;
                    spin_tries_[g]=std::move(trie);
                }
            }
            for (size_t w=0;w<threads_;++w) {
                memo_.push_back(std::make_unique<GugaMemo>(reuse_bytes_/threads_,reuse_,packed_,mode_==WideSpin ? GugaMemo::Wide : mode_==SemiDirect ? GugaMemo::Generic : GugaMemo::Tiny,collect_spin_,
                    mode_==CompactCache,mode_==CompactCache));
                if (symbolic()) symbolic_.push_back(std::make_unique<SymbolicTiles>(reuse_ ? max_symbolic/threads_ : 0));
            }
            diagonal_.resize(basis_->size());
            std::vector<GugaPlan> diagonal_plans(threads_);
            std::vector<GugaBitPlan> diagonal_bits(threads_);
            std::vector<GugaLeanPlan> diagonal_small(threads_);
            parallel_for(order_.size(),threads_,[&](size_t g,size_t w) {
                const auto &a=*order_[g];
                auto c=connection(PairDescriptor(a.first,a.first,g,g),one_.data(),two_.data(),terms_,true);
                if ((mode_!=WideSpin && mode_!=SemiDirect) && k_<=64) prepare_guga_lean_connection(a.first,a.first,c,diagonal_small[w],(mode_!=WideSpin && mode_!=SemiDirect));
                else if (mode_!=SemiDirect && k_<=64) prepare_guga_bit_connection(a.first,a.first,c,diagonal_plans[w],diagonal_bits[w],(mode_!=SemiDirect));
                std::vector<Path> local;
                auto p=get_paths(a.second,local);
                for (size_t r=0;r<a.second.size();++r) {
                    check_bonds(*p[r],*p[r],bonds_);
                    diagonal_[a.second[r]]=(mode_!=WideSpin && mode_!=SemiDirect) && k_<=64 ?
                        memo_[w]->small_element(*p[r],*p[r],c,diagonal_small[w],path_down_[a.second[r]],path_down_[a.second[r]],true) :
                        (mode_!=SemiDirect) && k_<=64 ?
                        memo_[w]->bit_element(*p[r],*p[r],c,diagonal_plans[w],diagonal_bits[w],
                            path_down_[a.second[r]],path_down_[a.second[r]],true) : memo_[w]->element(*p[r],*p[r],c);
                }
            });
            // Graph budget is enforced before each retained list. Partial
            // graphs are discarded, never silently treated as complete.

             {
                build_compact(max_graph);
                // Release the construction index; exact wide-key lookups
                // remain available for active-source products.
                if (compact_ready_)
                    index_=std::make_unique<SelectedIndex>(groups_,0,0);
            }
            prepare_rank_masks(max_ordered_);
            prepare_compact_spins(max_ordered_-rank_masks_.capacity()*sizeof(ConfigMasks));
            build_value_cache();
        } catch (const std::length_error &e) { throw BudgetError(e.what()); }
    }
    BasisSnapshot basis() const { return basis_; }
    double evaluate_pair_entry(size_t i,size_t j) const {
        // Independent, unscreened diagnostics; never mutate the product memo.
        std::string a_scratch,b_scratch;
        const auto &bra=basis_->label(i,a_scratch),&ket=basis_->label(j,b_scratch);
        size_t a=basis_->configuration_ids()[i],b=basis_->configuration_ids()[j];
        py::gil_scoped_release release;
        size_t budget=tensors_;
        Path p(bra,k_,two_s_,budget,false,factor_mode_,spin_checkpoints_);
        budget=tensors_;
        Path q(ket,k_,two_s_,budget,false,factor_mode_,spin_checkpoints_);
        check_bonds(p,q,bonds_);
        auto c=connection(PairDescriptor(order_[a]->first,order_[b]->first,a,b),
                          one_.data(),two_.data(),terms_,true,false);
        auto backend=entry_backend([&](size_t,size_t,size_t,size_t) { return guga_element(p,q,c); });
        return backend.evaluate_pair_entry(0,0,i,j);
    }
    Array diagonal() const {
        Array result(diagonal_.size());
        std::copy(diagonal_.begin(),diagonal_.end(),result.mutable_data());
        return result;
    }
    Stats info() const {
        Stats result;
        { py::gil_scoped_release release; std::lock_guard<std::mutex> lock(mutex_); result=info_unlocked(); }
        return result;
    }
    template<bool GeneratorScreening,bool Exact=false>
    py::tuple apply_fast(const Array &x,double generator_tol,int generator_screen,bool unscreened) {
        const double tolerance=unscreened ? 0. : tolerance_,cutoff=unscreened ? 0. : cutoff_;
        size_t n=basis_->size(), ng=order_.size();
        if (x.ndim()!=2 || size_t(x.shape(0))!=n)
            throw std::invalid_argument("trial block must have shape (basis, columns)");
        size_t cols=x.shape(1);
        const double *input=x.data();
        for (py::ssize_t p=0;p<x.size();++p)
            if (!std::isfinite(input[p])) throw std::invalid_argument("nonfinite trial block");
        Array result({py::ssize_t(n),py::ssize_t(cols)}), error_inf(cols),error_l2(cols);
        double *output=result.mutable_data();
        Stats stats;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            try {
                std::vector<double> row_max(n,0.),group_max(ng,0.);
                std::vector<long double> sums(Exact ? 0 : ng*cols,0.);
                std::vector<size_t> active;
                for (size_t g=0;g<ng;++g) {
                    for (size_t i:order_[g]->second) for (size_t v=0;v<cols;++v) {
                        double value=std::abs(input[i*cols+v]);
                        row_max[i]=std::max(row_max[i],value);
                        group_max[g]=std::max(group_max[g],value);
                        if constexpr(!Exact) sums[g*cols+v]+=static_cast<long double>(value);
                    }
                    if (group_max[g]!=0.) active.push_back(g);
                }
                // Conservative opt-in sparse policy. Enumerate only active
                // sources; dense and screened inputs retain symmetric blocks.
                bool sparse_source=Exact && ((mode_==Adaptive)) &&
                    active.size()<=std::min(size_t(1024),ng/64);
                bool source=sparse_source;
                bool blocks=!source;
                double max_coefficient=ng ? *std::max_element(group_max.begin(),group_max.end()) : 0.;
                bool late= (mode_!=SemiDirect) ||
                          (((mode_==SemiDirect)) && !source && max_coefficient>64*tolerance);
                bool planned=true;
                bool bit_kernel=(mode_!=SemiDirect), accelerated=true;

                bool compact=!source && compact_ready_;
                std::vector<size_t> tasks;
                if (source) tasks=std::move(active);
                else { tasks.resize(ng); std::iota(tasks.begin(),tasks.end(),size_t(0)); }
                size_t workers=std::min(threads_,std::max(size_t(1),tasks.size()));
                if (sparse_source) workers=std::min(workers,std::max(size_t(1),(tasks.size()+3)/4));
                if (cols && n>std::numeric_limits<size_t>::max()/cols/workers)
                    throw BudgetError("direct product scratch dimension overflow");
                size_t count=n*cols;
                std::vector<double> local_output;
                double *scratch;
                if (retained_product()) {
                    // Uninitialized allocation permits owner first-touch. Keep
                    // one width only so a block probe does not pin its peak RSS.
                    if (output_workspace_size_!=workers*count) {
                        output_workspace_.reset(new double[workers*count]);
                        output_workspace_size_=workers*count;
                    }
                    scratch=output_workspace_.get();
                    product_for(workers,workers,[&](size_t w,size_t) {
                        std::fill(scratch+w*count,scratch+(w+1)*count,0.);
                    });
                } else { local_output.assign(workers*count,0.);scratch=local_output.data(); }
                std::vector<long double> omitted(Exact ? 0 : workers*count,0.),uniform(Exact ? 0 : workers*cols,0.);
                struct alignas(64) Counts {
                    size_t pairs=0,zero_pairs=0,prepared=0,coarse=0,norm=0,zero_csf=0,
                           coefficient=0,elements=0,kept=0,dropped=0,cache_blocks=0,cache_elements=0,
                           sampled_blocks=0,sampled_prepared=0,sampled_elements=0,
                           preparation_ns=0,contraction_ns=0,batch_blocks=0,batch_pairs=0,
                           batch_terms_rejected=0,generators_considered=0,generators_removed=0,
                           generator_blocks=0,generator_empty_blocks=0;
                    std::array<size_t,3> rank_pairs{},rank_terms{},rank_preparation{},rank_contraction{};
                    std::array<size_t,16> shapes{};
                };
                std::vector<Counts> counters(workers);
                std::vector<Stats> connectivity(workers);
                std::vector<std::array<size_t,9>> before(workers);
                std::vector<std::array<size_t,6>> symbolic_before(workers);
                std::vector<Connection> local_connections;
                std::vector<GugaPlan> local_plans;
                std::vector<GugaBitPlan> local_bits;
                std::vector<GugaLeanPlan> local_small;
                std::vector<GugaBatchScratch> local_batches;
                auto &rank_scratch=retained_product() ? retained_connections_ : local_connections;
                auto &plans=retained_product() ? retained_plans_ : local_plans;
                auto &bitplans=retained_product() ? retained_bits_ : local_bits;
                auto &smallplans=retained_product() ? retained_small_ : local_small;
                auto &batches=retained_product() ? retained_batches_ : local_batches;
                rank_scratch.resize(workers);plans.resize(workers);bitplans.resize(workers);
                smallplans.resize(workers);batches.resize(workers);
                std::vector<SpinTraversalStats> traversal(workers);
                std::vector<HybridSpinScratch> hybrid_scratch(workers);
                std::vector<WeightedSpinScratch> weighted_scratch(workers);
                for (size_t w=0;w<workers;++w) {
                    before[w]=memo_[w]->product_counters();
                    if (symbolic()) symbolic_before[w]=symbolic_[w]->counters();
                }
                product_for(tasks.size(),workers,[&](size_t task,size_t w) {
                    size_t g=tasks[task];
                    const auto &a=*order_[g];
                    auto &s=counters[w];
                    double *out=scratch+w*count;
                    long double *err=Exact ? nullptr : omitted.data()+w*count;
                    std::vector<const Group *> temporary;
                    if (!compact) {
                         if (blocks) temporary=graph_partners(g,connectivity[w]);
                        else temporary=fast_partners(a,source ? std::numeric_limits<size_t>::max() :
                                                                            a.second.back(),connectivity[w]);
                    }
                    std::vector<Path> local_a;
                    std::vector<const Path *> pa;
                    bool cached_row=!value_offsets_.empty() &&
                        value_offsets_[g]!=std::numeric_limits<size_t>::max();
                    size_t cache_at=cached_row ? value_offsets_[g] : 0;
                    if (!cached_paths_ && !cached_row) pa=get_paths(a.second,local_a);
                    size_t begin=compact ? compact_ptr_[g] : 0;
                    size_t end=compact ? compact_ptr_[g+1] : temporary.size();
                    for (size_t at=begin;at<end;++at) {
                        size_t bg=compact ? size_t(compact_cols_[at]) : ordinal_.at(temporary[at]);
                        const auto &b=*order_[bg];
                        const double *cached=cached_row ? values_.data()+cache_at : nullptr;
                        if (cached_row) cache_at+=a.second.size()*b.second.size();
                        double maximum=source ? group_max[g] : std::max(group_max[g],group_max[bg]);
                        if (maximum==0.) { ++s.zero_pairs; continue; }
                        ++s.pairs;
                        auto omit_block=[&](long double bound) {
                            if constexpr(!Exact) {
                                if (source) {
                                    for (size_t i:b.second) for (size_t v=0;v<cols;++v)
                                        err[i*cols+v]+=bound*sums[g*cols+v];
                                } else {
                                    for (size_t i:a.second) for (size_t v=0;v<cols;++v)
                                        err[i*cols+v]+=bound*sums[bg*cols+v];
                                    for (size_t j:b.second) for (size_t v=0;v<cols;++v)
                                        err[j*cols+v]+=bound*sums[g*cols+v];
                                }
                            }
                        };
                        if (!late && &a!=&b && tolerance>0.) {
                            double bound=prescreen_bound(a.first,b.first,one_.data(),two_.data(),k_);
                            if (screen_bound(bound,maximum,tolerance,k_)) {
                                ++s.coarse; omit_block(bound); continue;
                            }
                        }
                        // Retained blocks never construct a Connection or
                        // path-pointer lists during a product. This is a
                        // separate semi-direct path, with identical screening.
                        if (cached) {
                            bool sample=profile_couplings_ && (at%256)==0;
                            auto begin=sample ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                            ++s.cache_blocks;
                            if (sample) ++s.sampled_blocks;
                            auto backend=entry_backend([&](size_t r,size_t t,size_t,size_t) {
                                return cached[r*b.second.size()+t];
                            });
                            auto accept=[&](size_t,size_t,size_t i,size_t j) {
                                if (std::max(row_max[i],row_max[j])==0.) { ++s.zero_csf; return false; }
                                return true;
                            };
                            auto publish_entry=[&](size_t i,size_t j,double value) {
                                ++s.cache_elements;
                                if (sample) ++s.sampled_elements;
                                if constexpr(Exact) {
                                    if (value==0.) s.dropped+=2*cols;
                                    else {
                                        for (size_t v=0;v<cols;++v) {
                                            out[i*cols+v]+=value*input[j*cols+v];
                                            out[j*cols+v]+=value*input[i*cols+v];
                                        }
                                        s.kept+=2*cols;
                                    }
                                } else for (size_t v=0;v<cols;++v) {
                                    double left=value*input[j*cols+v],right=value*input[i*cols+v];
                                    if (std::abs(value)<=cutoff || std::abs(left)<tolerance) {
                                        err[i*cols+v]+=std::abs(static_cast<long double>(value)*input[j*cols+v]); ++s.dropped;
                                    } else { out[i*cols+v]+=left; ++s.kept; }
                                    if (std::abs(value)<=cutoff || std::abs(right)<tolerance) {
                                        err[j*cols+v]+=std::abs(static_cast<long double>(value)*input[i*cols+v]); ++s.dropped;
                                    } else { out[j*cols+v]+=right; ++s.kept; }
                                }
                            };
                            apply_selected_block(backend,g==bg,{a.second,b.second,false,g==bg,false},accept,publish_entry);
                            if (sample) s.contraction_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now()-begin).count();
                            continue;
                        }
                        bool sample=profile_couplings_ && (at%256)==0;
                        auto prep_start=sample ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                        auto descriptor=describe_pair(source ? bg : g,source ? g : bg);
                        Connection ordinary;Connection &c=compact_mode() ? rank_scratch[w] : ordinary;
                        if (compact_mode()) ranked_connection(descriptor,one_.data(),two_.data(),terms_,!late,c);
                        else c=connection(descriptor,one_.data(),two_.data(),terms_,true,!late);
                        if (profile_couplings_) {
                            ++s.rank_pairs[descriptor.transfer_rank];s.rank_terms[descriptor.transfer_rank]+=c.generators.size();
                        }
                        ++s.prepared;
                        double bound=c.term_bound;
                        if (!late && &a!=&b && tolerance>0. && screen_bound(bound,maximum,tolerance,k_)) {
                            ++s.norm; omit_block(bound); continue;
                        }
                        bool generator_empty=false;
                        if constexpr (GeneratorScreening) if (!c.generators.empty()) {
                            s.generators_considered+=c.generators.size();
                            auto removed=screen_generators(c,maximum,generator_tol,generator_screen);
                            s.generators_removed+=removed.removed;
                            if (removed.removed) {
                                ++s.generator_blocks;
                                // This bounds all output columns, including changes to
                                // decisions made by the final contribution filter.
                                if (&a!=&b) omit_block(removed.bound);
                                else if (a.second.size()>1) for (size_t i:a.second) for (size_t v=0;v<cols;++v)
                                    err[i*cols+v]+=removed.bound*sums[g*cols+v];
                                generator_empty=c.generators.empty();
                            }
                        }
                        if (!generator_empty && (mode_!=WideSpin && mode_!=SemiDirect) && bit_kernel) {
                            if (compact_mode() && !rank_masks_.empty()) {
                                const auto &am=rank_masks_[source ? bg : g],&bm=rank_masks_[source ? g : bg];
                                 prepare_guga_mask_connection(c,smallplans[w],true,am.singles,am.doubles,bm.singles,bm.doubles);
                            } else prepare_guga_lean_connection(source ? b.first : a.first,source ? a.first : b.first,c,smallplans[w],(mode_!=WideSpin && mode_!=SemiDirect));
                        }
                        else if (!generator_empty && bit_kernel) prepare_guga_bit_connection(a.first,b.first,c,plans[w],bitplans[w],(mode_!=SemiDirect));
                        else if (!generator_empty && planned) prepare_guga_plan(a.first,b.first,c,plans[w]);
                        if (sample) {
                            ++s.sampled_blocks; s.sampled_prepared+=!cached;
                            size_t duration=std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now()-prep_start).count();
                            s.preparation_ns+=duration;s.rank_preparation[descriptor.transfer_rank]+=duration;
                        }
                        // Scalars act only on identical CSFs; the diagonal was
                        // constructed without generator screening during setup.
                        if (generator_empty) { ++s.generator_empty_blocks;continue; }
                        std::vector<Path> local_b;
                        std::vector<const Path *> pb;
                        if (!cached_paths_) pb=get_paths(b.second,local_b);
                        auto contraction_start=sample ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                        bool batched=false;
                        if (mode_!=SemiDirect && profile_couplings_) {
                            auto bin=[](size_t n) { return n<2 ? 0 : n<4 ? 1 : n<16 ? 2 : 3; };
                            ++s.shapes[4*bin(a.second.size()*b.second.size())+bin(c.generators.size())];
                        }
                        bool batch_mode=(mode_!=SemiDirect);
                        bool batch_size=
                            a.second.size()>=min_batch_/b.second.size()+size_t(min_batch_%b.second.size()!=0);
                        bool batch_terms=((mode_==SemiDirect)) || c.generators.size()>=min_batch_terms_;
                        if (batch_mode && batch_size && !batch_terms) ++s.batch_terms_rejected;
                        if (!source && batch_mode && batch_size && batch_terms && cached_paths_) {
                            if (bonds_<2 && k_) throw BudgetError("GUGA segment scratch exceeds max_bond_elements");
                            bool exact=tolerance==0. && cutoff==0. && generator_tol==0.;
                            const auto *ta=spin_tries_.empty() ? nullptr : spin_tries_[g].get();
                            const auto *tb=spin_tries_.empty() ? nullptr : spin_tries_[bg].get();
                            int hybrid_result=0;
                            if (mode_==Weighted && exact && ta && tb && weighted_guga_apply(a.first,b.first,*ta,*tb,
                                paths_,a.second,b.second,row_max,g==bg,c,input,out,cols,max_batch_,weighted_scratch[w],traversal[w]))
                                hybrid_result=2;
                            if (mode_==Hybrid && tb)
                                hybrid_result=hybrid_guga_apply(a.first,b.first,*tb,paths_,a.second,b.second,path_down_,
                                    row_max,g==bg,c,smallplans[w],*memo_[w],accelerated,basis_->context().nelec,max_batch_,
                                    batches[w],hybrid_scratch[w],traversal[w]);
                            if (hybrid_result==2) {
                                if (sample) s.contraction_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now()-contraction_start).count();
                                continue;
                            }
                            batched=hybrid_result==1;
                            if (mode_==Selected && !spin_tries_.empty() && spin_tries_[bg])
                                batched=selected_guga_batch(a.first,b.first,*spin_tries_[bg],paths_,a.second,b.second,
                                    row_max,g==bg,c,max_batch_,batches[w],traversal[w]);

                            if (mode_==Tuga) batched=symbolic_[w]->batch(a.first,b.first,paths_,a.second,b.second,c,
                                smallplans[w],path_down_,*memo_[w],accelerated,max_batch_,g==bg,batches[w]);
                            if (!batched && !compact_spins_.empty()) batched=memo_[w]->compact_batch(paths_,compact_spins_,a.second,b.second,
                                path_down_,row_max,g==bg,c,smallplans[w],accelerated,max_batch_,batches[w]);
                            if (!batched && k_<=64) batched=(mode_!=WideSpin && mode_!=SemiDirect) ? memo_[w]->small_batch(paths_,a.second,b.second,path_down_,row_max,g==bg,
                                c,smallplans[w],accelerated,max_batch_,batches[w]) : memo_[w]->batch_elements(paths_,a.second,b.second,path_down_,row_max,g==bg,
                                c,plans[w],bitplans[w],accelerated,max_batch_,batches[w]);
                            if (batched) { ++s.batch_blocks; s.batch_pairs+=a.second.size()*b.second.size(); }
                        }
                        auto backend=entry_backend([&](size_t r,size_t t,size_t i,size_t j) {
                            const Path &p=cached_paths_ ? paths_[i] : (source ? *pb[t] : *pa[r]);
                            const Path &q=cached_paths_ ? paths_[j] : (source ? *pa[r] : *pb[t]);
                            if (!compact_spins_.empty()) {
                                if (bonds_<2 && k_) throw BudgetError("GUGA segment scratch exceeds max_bond_elements");
                                return batched ? batches[w].values[r*b.second.size()+t] :
                                    memo_[w]->compact_element(p,q,compact_spins_[i],compact_spins_[j],c,smallplans[w],path_down_[i],path_down_[j],accelerated);
                            }
                            check_bonds(p,q,bonds_);
                            return batched ? batches[w].values[r*b.second.size()+t] :
                                (mode_!=WideSpin && mode_!=SemiDirect) ? memo_[w]->small_element(p,q,c,smallplans[w],
                                    path_down_.empty() ? 0 : path_down_[i],path_down_.empty() ? 0 : path_down_[j],accelerated) :
                                bit_kernel ? memo_[w]->bit_element(p,q,c,plans[w],bitplans[w],
                                    path_down_.empty() ? 0 : path_down_[i],path_down_.empty() ? 0 : path_down_[j],accelerated) :
                                planned ? memo_[w]->planned_element(p,q,c,plans[w]) :
                                memo_[w]->element(p,q,c);
                        });
                        auto accept=[&](size_t,size_t,size_t i,size_t j) {
                            double weight=source ? row_max[j] : std::max(row_max[i],row_max[j]);
                            if (weight==0.) { ++s.zero_csf; return false; }
                            if constexpr(!Exact) if (!late && tolerance>0. && screen_bound(bound,weight,tolerance,k_)) {
                                ++s.coefficient;
                                for (size_t v=0;v<cols;++v) {
                                    err[i*cols+v]+=bound*std::abs(input[j*cols+v]);
                                    if (!source) err[j*cols+v]+=bound*std::abs(input[i*cols+v]);
                                }
                                return false;
                            }
                            return true;
                        };
                        auto publish_entry=[&](size_t i,size_t j,double value) {
                            ++s.elements;
                            if (sample) ++s.sampled_elements;
                            if constexpr(Exact) {
                                if (value==0.) s.dropped+=(source ? 1 : 2)*cols;
                                else {
                                    for (size_t v=0;v<cols;++v) {
                                        out[i*cols+v]+=value*input[j*cols+v];
                                        if (!source) out[j*cols+v]+=value*input[i*cols+v];
                                    }
                                    s.kept+=(source ? 1 : 2)*cols;
                                }
                            } else {
                                auto publish=[&](size_t to,size_t from) {
                                    for (size_t v=0;v<cols;++v) {
                                        double term=value*input[from*cols+v];
                                        if (std::abs(value)<=cutoff || std::abs(term)<tolerance) {
                                            err[to*cols+v]+=std::abs(static_cast<long double>(value)*input[from*cols+v]);
                                            ++s.dropped;
                                        } else { out[to*cols+v]+=term; ++s.kept; }
                                    }
                                };
                                publish(i,j); if (!source) publish(j,i);
                            }
                        };
                        apply_selected_block(backend,g==bg,
                            {a.second,b.second,source,!source && (!blocks || g==bg),false},accept,publish_entry);
                        if (sample) {
                            size_t duration=std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now()-contraction_start).count();
                            s.contraction_ns+=duration;s.rank_contraction[descriptor.transfer_rank]+=duration;
                        }
                    }
                },sparse_source);
                for (size_t w=0;w<workers;++w) {
                    const auto &s=counters[w];
                    stats["configuration_pairs"]+=s.pairs; stats["zero_pairs"]+=s.zero_pairs;
                    stats["prepared_pairs"]+=s.prepared; stats["coarse_pruned_blocks"]+=s.coarse;
                    stats["norm_pruned_blocks"]+=s.norm; stats["zero_csf_pairs"]+=s.zero_csf;
                    stats["coefficient_pruned_pairs"]+=s.coefficient; stats["matrix_elements"]+=s.elements;
                    stats["kept_contributions"]+=s.kept; stats["dropped_contributions"]+=s.dropped;
                    stats["cached_blocks"]+=s.cache_blocks; stats["cached_matrix_elements"]+=s.cache_elements;
                    stats["sampled_blocks"]+=s.sampled_blocks; stats["sampled_prepared_blocks"]+=s.sampled_prepared;
                    stats["sampled_matrix_elements"]+=s.sampled_elements;
                    stats["sampled_preparation_ns"]+=s.preparation_ns;
                    stats["sampled_contraction_accumulation_ns"]+=s.contraction_ns;
                    stats["batch_blocks"]+=s.batch_blocks; stats["batch_rectangle_pairs"]+=s.batch_pairs;
                    stats["batch_term_rejected_blocks"]+=s.batch_terms_rejected;
                    stats["generators_considered"]+=s.generators_considered;
                    stats["generators_removed"]+=s.generators_removed;
                    stats["generator_screened_blocks"]+=s.generator_blocks;
                    stats["generator_empty_blocks"]+=s.generator_empty_blocks;
                    if (mode_!=SemiDirect && profile_couplings_) for (size_t b=0;b<16;++b)
                        stats["block_shape_"+std::to_string(b/4)+"_"+std::to_string(b%4)]+=s.shapes[b];
                    stats["batch_scratch_bytes"]+=batches[w].bytes();
                    const auto &t=traversal[w];
                    stats["selected_guga_blocks"]+=t.blocks;stats["selected_guga_nodes"]+=t.nodes;
                    stats["selected_guga_pruned"]+=t.pruned;stats["selected_guga_leaves"]+=t.leaves;
                    stats["selected_guga_channels"]+=t.channels;stats["selected_spin_scratch_bytes"]+=t.scratch_peak;
                    stats["hybrid_short_blocks"]+=t.hybrid_short_blocks;stats["hybrid_split_blocks"]+=t.hybrid_split_blocks;
                    stats["hybrid_short_generators"]+=t.hybrid_short_terms;stats["hybrid_long_generators"]+=t.hybrid_long_terms;
                    stats["weighted_guga_blocks"]+=t.weighted_blocks;stats["weighted_guga_nodes"]+=t.weighted_nodes;
                    stats["weighted_suffix_hits"]+=t.weighted_hits;stats["weighted_guga_pruned"]+=t.weighted_pruned;
                    stats["weighted_guga_channels"]+=t.weighted_channels;
                    stats["weighted_scratch_bytes"]+=weighted_scratch[w].bytes();
                    stats["weighted_buffer_peak_elements"]=std::max(stats["weighted_buffer_peak_elements"],t.weighted_buffer_peak);
                    stats["hybrid_scratch_bytes"]+=hybrid_scratch[w].bytes();
                    stats["rank_preparation_scratch_bytes"]+=rank_scratch[w].generators.capacity()*sizeof(GugaTerm);
                    for (size_t rank=0;rank<3;++rank) {
                        auto prefix="rank_"+std::to_string(rank)+"_";
                        stats[prefix+"pairs"]+=s.rank_pairs[rank];stats[prefix+"generators"]+=s.rank_terms[rank];
                        stats[prefix+"sampled_preparation_ns"]+=s.rank_preparation[rank];
                        stats[prefix+"sampled_contraction_ns"]+=s.rank_contraction[rank];
                    }
                    stats["plan_scratch_bytes"]+=plans[w].capacity()*sizeof(GugaPreparedTerm)+
                                                  bitplans[w].capacity()*sizeof(GugaBitTerm)+smallplans[w].capacity()*sizeof(GugaLeanTerm);
                    for (const auto &item:connectivity[w]) stats[item.first]+=item.second;
                    auto after=memo_[w]->product_counters();
                    const std::array<const char *,9> keys={"reuse_hits","reuse_misses","reuse_orthogonal","reuse_replacements",
                        "reuse_primary_hits","reuse_secondary_hits","reuse_empty_misses","reuse_full_misses",
                        "reuse_bypass"};
                    for (size_t p=0;p<keys.size();++p) stats[keys[p]]+=after[p]-before[w][p];
                    if (symbolic()) {
                        auto values=symbolic_[w]->counters();
                        const std::array<const char *,6> names={"symbolic_hits","symbolic_misses","symbolic_bypass",
                            "symbolic_evictions","symbolic_generated_coefficients","symbolic_contracted_coefficients"};
                        for (size_t p=0;p<names.size();++p) stats[names[p]]+=values[p]-symbolic_before[w][p];
                    }
                }
                std::vector<long double> uniform_sum(Exact ? 0 : cols,0.);
                if constexpr(!Exact) {
                    for (size_t w=0;w<workers;++w) for (size_t v=0;v<cols;++v)
                        uniform_sum[v]+=uniform[w*cols+v];
                    std::vector<long double> largest(workers*cols,0.),squared(workers*cols,0.);
                    product_for((n+1023)/1024,workers,[&](size_t tile,size_t owner) {
                        for (size_t i=tile*1024;i<std::min(n,(tile+1)*1024);++i) for (size_t v=0;v<cols;++v) {
                            size_t at=i*cols+v;
                            double value=diagonal_[i]*input[at];
                            long double error=uniform_sum[v];
                            for (size_t w=0;w<workers;++w) {
                                value+=scratch[w*count+at]; error+=omitted[w*count+at];
                            }
                            output[at]=value;
                            largest[owner*cols+v]=std::max(largest[owner*cols+v],error);
                            squared[owner*cols+v]+=error*error;
                        }
                    });
                    for (size_t v=0;v<cols;++v) {
                        long double top=0.,square=0.;
                        for (size_t w=0;w<workers;++w) { top=std::max(top,largest[w*cols+v]); square+=squared[w*cols+v]; }
                        long double gamma=16*(static_cast<long double>(n)+workers+1)*std::numeric_limits<long double>::epsilon();
                        long double inflate=gamma<1 ? 1/(1-gamma) : INFINITY;
                        error_inf.mutable_data()[v]=std::nextafter(double(top*inflate),INFINITY);
                        error_l2.mutable_data()[v]=std::nextafter(double(std::sqrt(square)*inflate),INFINITY);
                    }
                    stats["screening_error_scratch_bytes"]=(omitted.capacity()+uniform.capacity()+
                        uniform_sum.capacity()+largest.capacity()+squared.capacity())*sizeof(long double);
                } else {
                    product_for((n+1023)/1024,workers,[&](size_t tile,size_t) {
                        for (size_t i=tile*1024;i<std::min(n,(tile+1)*1024);++i) for (size_t v=0;v<cols;++v) {
                            size_t at=i*cols+v;double value=diagonal_[i]*input[at];
                            for (size_t w=0;w<workers;++w) value+=scratch[w*count+at];
                            output[at]=value;
                        }
                    });
                    for (size_t v=0;v<cols;++v) error_inf.mutable_data()[v]=error_l2.mutable_data()[v]=std::nextafter(0.,INFINITY);
                    stats["screening_error_scratch_bytes"]=0;
                }
                stats["exact_product_kernel"]=Exact;
                stats["workers"]=workers; stats["optimized"]=1;
                stats["active_source_groups"]=source ? tasks.size() : active.size();
                stats["source_traversal"]=source; stats["compact_traversal"]=compact;
                stats["block_traversal"]=blocks;
                stats["late_screening"]=late;
                stats["prepared_guga"]=planned;

                stats["output_scratch_bytes"]=workers*count*sizeof(double);
                stats["retained_product_workspace_bytes"]=workspace_bytes();
                stats["sparse_source_policy"]=sparse_source;
                stats["coefficient_metadata_bytes"]=row_max.capacity()*sizeof(double)+
                    group_max.capacity()*sizeof(double)+sums.capacity()*sizeof(long double)+
                    tasks.capacity()*sizeof(size_t)+active.capacity()*sizeof(size_t);
                last_=stats;
            } catch (const std::length_error &e) { throw BudgetError(e.what()); }
        }
        return py::make_tuple(result,error_inf,error_l2,py::cast(stats));
    }
    py::tuple apply(const Array &x,double generator_tol=0.,int generator_screen=0,bool unscreened=false) {
        if (!std::isfinite(generator_tol) || generator_tol<0. || generator_screen<0 || generator_screen>1)
            throw std::invalid_argument("invalid generator screening tolerance/policy");
        if ((generator_tol>0. || unscreened) && (mode_==WideSpin || mode_==SemiDirect))
            throw std::invalid_argument("generator screening and unscreened overrides require a prepared lean/tiny strategy");
        if (unscreened) generator_tol=0.;
        if (compact_mode() && generator_tol==0. && (unscreened || (tolerance_==0. && cutoff_==0.)))
            return apply_fast<false,true>(x,0.,generator_screen,unscreened);
        return generator_tol>0. ? apply_fast<true>(x,generator_tol,generator_screen,unscreened) :
                                 apply_fast<false>(x,0.,generator_screen,unscreened);
    }
};

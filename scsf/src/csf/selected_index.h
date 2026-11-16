/* This file is part of SCSF, licensed under GNU GPL version 3 or later. */
#pragma once
#include "selected_basis.h"
#include <algorithm>
#include <numeric>
#include <random>

namespace scsf {
// Experimental indexes contain only selected occupation configurations. Spin
// paths remain in the original groups with their stable append indices.
template<class GroupMap>
class ConfigurationIndex {
    using Group = typename GroupMap::value_type;
    struct Node {
        std::array<int,3> child{-1,-1,-1};
        const Group *leaf = nullptr;
        size_t first_column = std::numeric_limits<size_t>::max();
    };
    struct Residue { std::string key; const Group *group; };
    struct HashedResidue { std::uint64_t hash; size_t group; };
    const GroupMap &groups_;
    std::vector<Node> nodes;
    std::vector<Residue> residues;
    std::vector<HashedResidue> hashed_;
    std::vector<size_t> buckets_;
    std::vector<std::array<std::uint64_t,3>> zobrist_;
    std::vector<const Group *> hash_groups_;
    std::uint64_t hash_mask_ = ~std::uint64_t(0);
    int mode_ = 0;
    size_t bytes_ = 0;

    static std::string packed(const Config &config) {
        std::string key((config.size()+3)/4,'\0');
        for (size_t p = 0; p < config.size(); ++p)
            key[p/4] = static_cast<unsigned char>(key[p/4]) | (config[p] << (2*(p%4)));
        return key;
    }
    template<class Callback>
    static void remove_two(const Config &source, Callback callback) {
        int n = std::accumulate(source.begin(),source.end(),0);
        Config residue = source;
        if (n == 0) { callback(packed(residue)); return; }
        for (size_t q = 0; q < source.size(); ++q) {
            if (!source[q]) continue;
            --residue[q];
            if (n == 1) callback(packed(residue));
            else for (size_t s = q; s < source.size(); ++s) {
                if (!residue[s]) continue;
                --residue[s];
                callback(packed(residue));
                ++residue[s];
            }
            ++residue[q];
        }
    }
    template<class Callback>
    void hashed_remove_two(const Config &source, Callback callback) const {
        std::uint64_t hash=0;int electrons=0;
        for (size_t p=0;p<source.size();++p) {
            hash^=zobrist_[p][source[p]];electrons+=source[p];
        }
        if (!electrons) { callback(hash&hash_mask_);return; }
        for (size_t q=0;q<source.size();++q) if (source[q]) {
            auto first=hash^zobrist_[q][source[q]]^zobrist_[q][source[q]-1];
            if (electrons==1) callback(first&hash_mask_);
            else for (size_t s=q;s<source.size();++s) {
                int n=source[s]-(q==s);
                if (n) callback((first^zobrist_[s][n]^zobrist_[s][n-1])&hash_mask_);
            }
        }
    }
public:
    ConfigurationIndex(const GroupMap &right, int mode, size_t maximum, unsigned hash_bits=64) : groups_(right) {
        if (mode<0 || mode>3 || hash_bits>64) throw std::invalid_argument("unknown configuration index mode/hash width");
        if (hash_bits<64) hash_mask_=(std::uint64_t(1)<<hash_bits)-1;
        if (!mode || right.empty()) return;
        if (mode == 1) {
            size_t max_nodes = std::min(maximum/sizeof(Node),static_cast<size_t>(std::numeric_limits<int>::max()));
            if (!max_nodes) return;
            auto append = [&] {
                if (nodes.size() == max_nodes) return false;
                if (nodes.size() == nodes.capacity())
                    nodes.reserve(std::min(max_nodes,std::max(size_t(16),nodes.capacity()*2)));
                nodes.emplace_back();
                return true;
            };
            append();
            for (const auto &group : right) {
                size_t at = 0;
                nodes[at].first_column = std::min(nodes[at].first_column,group.second.front());
                for (int n : group.first) {
                    int next = nodes[at].child[n];
                    if (next < 0) {
                        next = static_cast<int>(nodes.size());
                        if (!append()) { std::vector<Node>().swap(nodes); return; }
                        nodes[at].child[n] = next;
                    }
                    at = next;
                    nodes[at].first_column = std::min(nodes[at].first_column,group.second.front());
                }
                nodes[at].leaf = &group;
            }
            bytes_ = nodes.capacity()*sizeof(Node);
        } else {
            size_t count = 0;
            // Include dynamic key storage even for keys fitting small-string
            // optimization. This intentionally conservative cap precedes allocation.
            size_t item_bytes = mode==3 ? sizeof(HashedResidue) : sizeof(Residue)+
                std::max(std::string().capacity(),(right.begin()->first.size()+3)/4)+1;
            size_t max_items = maximum/item_bytes;
            for (const auto &group : right) {
                size_t singles = 0, doubles = 0;
                for (int n : group.first) { singles += n != 0; doubles += n == 2; }
                size_t n = std::accumulate(group.first.begin(),group.first.end(),size_t(0));
                size_t add = n < 2 ? 1 : singles*(singles-1)/2+doubles;
                if (add > max_items-count) return;
                count += add;
            }
            if (mode==3) {
                // Flat bucket lists retain fingerprints and selected endpoints,
                // never numerical couplings. The index cap covers all three
                // retained vectors. Construction additionally uses one cursor
                // per bucket, released before publishing the index.
                size_t bucket_count=1;
                while (bucket_count<count/4) bucket_count*=2;
                size_t table_count=right.begin()->first.size();
                size_t records=count*sizeof(HashedResidue);
                if (bucket_count+1>(maximum-records)/sizeof(size_t)) return;
                size_t directory=(bucket_count+1)*sizeof(size_t);
                if (right.size()>(maximum-records-directory)/sizeof(hash_groups_[0])) return;
                directory+=right.size()*sizeof(hash_groups_[0]);
                if (table_count>(maximum-records-directory)/sizeof(zobrist_[0])) return;
                hashed_.resize(count);buckets_.resize(bucket_count+1,0);zobrist_.resize(table_count);
                hash_groups_.reserve(right.size());
                for (const auto &group:right) hash_groups_.push_back(&group);
                bytes_=hashed_.capacity()*sizeof(HashedResidue)+buckets_.capacity()*sizeof(size_t)+
                    zobrist_.capacity()*sizeof(zobrist_[0])+hash_groups_.capacity()*sizeof(hash_groups_[0]);
                if (bytes_>maximum) {
                    decltype(hashed_)().swap(hashed_);decltype(buckets_)().swap(buckets_);
                    decltype(zobrist_)().swap(zobrist_);decltype(hash_groups_)().swap(hash_groups_);bytes_=0;return;
                }
                std::mt19937_64 rng(0x23a23cf5033c3c81ULL);
                for (auto &site:zobrist_) site={0,rng(),rng()};
                for (const auto &group:right) hashed_remove_two(group.first,[&](auto h) {
                    ++buckets_[(h&(bucket_count-1))+1];
                });
                std::partial_sum(buckets_.begin(),buckets_.end(),buckets_.begin());
                auto cursor=buckets_;
                for (size_t group=0;group<hash_groups_.size();++group) hashed_remove_two(hash_groups_[group]->first,[&](auto h) {
                    hashed_[cursor[h&(bucket_count-1)]++]={h,group};
                });
                mode_=mode;return;
            }
            residues.reserve(count);
            for (const auto &group : right)
                remove_two(group.first,[&](std::string key) { residues.push_back({std::move(key),&group}); });
            std::sort(residues.begin(),residues.end(),[](const auto &a, const auto &b) {
                return a.key < b.key;
            });
            bytes_ = residues.capacity()*sizeof(Residue);
            for (const auto &r : residues) bytes_ += r.key.capacity()+1;
            if (bytes_ > maximum) { std::vector<Residue>().swap(residues); bytes_ = 0; return; }
        }
        mode_ = mode;
    }
    int mode() const { return mode_; }
    size_t bytes() const { return bytes_; }
    size_t entries() const { return mode_==1 ? nodes.size() : mode_==3 ? hashed_.size() : residues.size(); }
    std::vector<const Group *> query(const Config &source, size_t last_column, Stats &stats) const {
        std::vector<const Group *> partners;
        if (mode_ == 0) {
            // An absent/overflowed index still denotes the complete selected
            // space. Streaming clients must never interpret it as no edges.
            for (const auto &group:groups_) {
                if (group.second.front()>last_column) continue;
                int distance=0;
                for (size_t p=0;p<source.size();++p) distance+=std::abs(source[p]-group.first[p]);
                ++stats["index_visits"];
                if (distance<=4) partners.push_back(&group);
            }
        } else if (mode_ == 1) {
            auto visit = [&](auto &&self, int at, size_t p, int distance) -> void {
                ++stats["index_visits"];
                const Node &node = nodes[at];
                if (node.first_column > last_column) return;
                if (p == source.size()) { partners.push_back(node.leaf); return; }
                for (int n = 0; n < 3; ++n) {
                    int next = node.child[n], cost = distance+std::abs(source[p]-n);
                    if (next >= 0 && cost <= 4) self(self,next,p+1,cost);
                }
            };
            visit(visit,0,0,0);
        } else if (mode_ == 2) {
            remove_two(source,[&](const std::string &key) {
                ++stats["index_visits"];
                auto first = std::lower_bound(residues.begin(),residues.end(),key,
                                             [](const auto &r, const auto &s) { return r.key < s; });
                for (auto it = first; it != residues.end() && it->key == key; ++it) {
                    ++stats["residue_records_visited"];
                    if (it->group->second.front() <= last_column) partners.push_back(it->group);
                }
            });
            size_t before = partners.size();
            std::sort(partners.begin(),partners.end(),[](const auto *a, const auto *b) { return a->first < b->first; });
            partners.erase(std::unique(partners.begin(),partners.end()),partners.end());
            stats["duplicate_partners"] += before-partners.size();
        } else if (mode_==3) {
            size_t visits=0;
            std::vector<size_t> candidates;
            hashed_remove_two(source,[&](auto h) {
                ++stats["index_visits"];
                size_t bucket=h&(buckets_.size()-2);
                for (size_t i=buckets_[bucket];i<buckets_[bucket+1];++i) {
                    ++visits;
                    const auto &r=hashed_[i];
                    if (r.hash==h && hash_groups_[r.group]->second.front()<=last_column)
                        candidates.push_back(r.group);
                }
            });
            stats["hash_records_visited"]+=visits;stats["residue_records_visited"]+=candidates.size();
            size_t before=candidates.size();
            // IDs follow the map's exact lexicographic occupation order.
            // Integer sorting preserves the reference accumulation order
            // without repeatedly comparing full configuration vectors.
            std::sort(candidates.begin(),candidates.end());
            candidates.erase(std::unique(candidates.begin(),candidates.end()),candidates.end());
            stats["duplicate_partners"]+=before-candidates.size();
            // Hash equality is only a candidate filter. Check full endpoint
            // occupations after deduplication: equal-electron configurations
            // share a two-removal residue iff their L1 distance is <=4.
            // This also makes deliberately truncated/constant hashes exact.
            partners.reserve(candidates.size());
            for (size_t id:candidates) {
                const auto *g=hash_groups_[id];
                int distance=0;
                for (size_t p=0;p<source.size();++p) {
                    distance+=std::abs(source[p]-g->first[p]);
                    if (distance>4) break;
                }
                if (distance<=4) partners.push_back(g);
                else ++stats["hash_collision_rejections"];
            }
        }
        return partners;
    }
};

using SelectedIndex = ConfigurationIndex<Groups>;
} // namespace scsf

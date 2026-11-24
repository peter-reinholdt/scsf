/* This file is part of SCSF, licensed under GNU GPL version 3 or later. */
#pragma once
#include "csf.h"
#include <algorithm>
#include <limits>
#include <numeric>

namespace scsf {
// A streamed structural description borrowing occupations from a basis. It
// contains no integrals, coefficients, bounds, hashes or numerical H entries.
// The common representation uses the identity physical-orbital map. Reduction,
// phase conversion and full symbolic cache keys belong to the coupling backend.
struct PairDescriptor {
    const Config &bra, &ket;
    size_t I, J;
    int transfer_rank = 0;
    std::array<int,2> holes{}, particles{};
    static constexpr size_t no_id = std::numeric_limits<size_t>::max();

    PairDescriptor(const Config &a, const Config &b, size_t i=no_id, size_t j=no_id)
        : bra(a), ket(b), I(i), J(j) {
        if (a.size()!=b.size()) throw std::invalid_argument("configuration dimensions differ");
        int removed=0, inserted=0;
        for (size_t p=0;p<a.size();++p) {
            if (a[p]<0 || a[p]>2 || b[p]<0 || b[p]>2)
                throw std::invalid_argument("occupations must be 0,1,2");
            for (int n=b[p];n>a[p];--n) {
                if (removed<2) holes[removed]=static_cast<int>(p);
                ++removed;
            }
            for (int n=a[p];n>b[p];--n) {
                if (inserted<2) particles[inserted]=static_cast<int>(p);
                ++inserted;
            }
        }
        if (removed!=inserted) throw std::invalid_argument("configuration electron numbers differ");
        transfer_rank=removed;
    }
    // Transfers produced by the canonical occupation enumerator already
    // carry the minimal, ordered hole and particle multisets.
    PairDescriptor(const Config &a,const Config &b,size_t i,size_t j,
                   std::array<int,2> q,std::array<int,2> p,int rank)
        : bra(a),ket(b),I(i),J(j),transfer_rank(rank),holes(q),particles(p) {}
    // Trusted immutable-basis masks: occupations, orbital count and electron
    // sector have already been validated by the occupation basis. No integral data.
    static PairDescriptor from_masks(const Config &a,const Config &b,size_t i,size_t j,
        std::uint64_t a1,std::uint64_t a2,std::uint64_t b1,std::uint64_t b2) {
        PairDescriptor pair(a,b,i,j,Trusted{});
        auto removed=(b1&~(a1|a2))|(b2&~a2),double_removed=b2&~(a1|a2);
        auto inserted=(a1&~(b1|b2))|(a2&~b2),double_inserted=a2&~(b1|b2);
        pair.transfer_rank=__builtin_popcountll(removed)+__builtin_popcountll(double_removed);
        if (pair.transfer_rank<=2) {
            auto fill=[](std::uint64_t first,std::uint64_t second,std::array<int,2> &sites) {
                size_t at=0;
                while (first) {
                    int p=__builtin_ctzll(first);sites[at++]=p;
                    if (second&(std::uint64_t(1)<<p)) sites[at++]=p;
                    first&=first-1;
                }
            };
            fill(removed,double_removed,pair.holes);fill(inserted,double_inserted,pair.particles);
        }
        return pair;
    }
private:
    struct Trusted {};
    PairDescriptor(const Config &a,const Config &b,size_t i,size_t j,Trusted) : bra(a),ket(b),I(i),J(j) {}
public:
    bool connected() const { return transfer_rank<=2; }

    template<class Callback> void one_body_patterns(Callback callback) const {
        if (transfer_rank==0) {
            for (size_t p=0;p<ket.size();++p) if (ket[p]) callback(int(p),int(p));
        } else if (transfer_rank==1) callback(particles[0],holes[0]);
    }
    // Structural hole/particle pair order, including repeated indices and
    // occupied spectator terms. Integral permutation expansion follows later.
    template<class Callback> void two_body_patterns(Callback callback) const {
        if (!connected()) return;
        if (transfer_rank==2) callback(holes[0],holes[1],particles[0],particles[1]);
        else if (transfer_rank==1) {
            for (int t=0;t<static_cast<int>(ket.size());++t)
                if (ket[t]>(t==holes[0]))
                    callback(std::min(holes[0],t),std::max(holes[0],t),
                             std::min(particles[0],t),std::max(particles[0],t));
        } else for (int q=0;q<static_cast<int>(ket.size());++q) {
            if (!ket[q]) continue;
            for (int s=q;s<static_cast<int>(ket.size());++s)
                if (ket[s]>=(q==s ? 2 : 1)) callback(q,s,q,s);
        }
    }
};

// Rank-specialized preparation reuses a worker-owned generator buffer.
namespace structured { struct SpatialERI; }
void ranked_connection(const PairDescriptor &,const double *,const structured::SpatialERI &,bool,Connection &);
// Same-occupation terms only require (pp|rr) and (pr|rp), stored as two k-by-k arrays.
Connection guga_diagonal_connection(const Config &,const double *,const double *);
void guga_diagonal_connection(const Config &,const double *,const double *,Connection &);

// Contraction is explicitly separate from the descriptor's lifetime/identity.
Connection connection(const PairDescriptor &, const double *h, const double *v,
                      size_t max_terms, bool guga, bool compute_bounds=true);
} // namespace scsf

/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */

#include "csf.h"
#include "pair_descriptor.h"
#include "structured_eri.h"
#include <type_traits>
#include <algorithm>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace scsf {

Config configuration(const std::string &label, int two_s) {
    Config result;
    int spin = 0;
    for (char step : label) {
        if (step == 'u') { ++spin; result.push_back(1); }
        else if (step == 'd') { --spin; result.push_back(1); }
        else if (step == '0' || step == '2') result.push_back(step == '0' ? 0 : 2);
        else throw std::invalid_argument("CSF symbols must be 0,u,d,2");
        if (spin < 0) throw std::invalid_argument("negative intermediate CSF spin");
    }
    if (two_s < 0 || spin != two_s)
        throw std::invalid_argument("CSF final spin differs from two_s");
    return result;
}

namespace {
// Visit spatial orderings once for both the cheap norm bound and the exact
// term builder. This avoids a second set of spin/integral multiplicity rules.
template<bool DiagonalERI=false,class ERI,class Callback>
void spatial_coefficients(int q, int s, int a, int b, ERI v,
                          int k, Callback callback) {
    for (int hp = 0; hp < (q == s ? 1 : 2); ++hp) {
        int qq = hp == 0 ? q : s, ss = hp == 0 ? s : q;
        for (int pp = 0; pp < (a == b ? 1 : 2); ++pp) {
            int p = pp == 0 ? a : b, r = pp == 0 ? b : a;
            double integral;
            if constexpr (std::is_same_v<ERI,structured::SpatialERI>) {
                integral=0.5*v(p,qq,r,ss);
            } else if constexpr (DiagonalERI) {
                integral=0.5*v[(p!=qq ? size_t(k)*k : 0)+static_cast<size_t>(p)*k+r];
            } else {
                integral=0.5*v[((static_cast<size_t>(p)*k+r)*k+qq)*k+ss];
            }
            if (integral != 0.0) callback(qq, ss, p, r, integral);
        }
    }
}

struct NormBound {
    long double value = 0.0L;
    size_t count = 0;
    void add(double coefficient, size_t multiplicity) {
        if (coefficient == 0.0) return;
        value += std::abs(static_cast<long double>(coefficient))*multiplicity;
        count += multiplicity;
    }
    void add_scaled(double coefficient, double norm) {
        if (coefficient == 0.0 || norm == 0.0) return;
        value += std::abs(static_cast<long double>(coefficient))*norm;
        count += static_cast<size_t>(std::ceil(norm))+1;
    }
    double finish(int k) const {
        if (!count) return 0.0;
        // Positive extended-precision summation has relative error bounded
        // by n*u. Inflate with a conservative denominator, then round the
        // final double outward. Avoid an expensive nextafterl per coefficient.
        long double error = count*std::numeric_limits<long double>::epsilon();
        if (error >= 0.25L) return std::numeric_limits<double>::infinity();
        long double upper = value/(1.0L-2*error);
        long double steps = 512.0L*(k+1)+8.0L*(count+1);
        long double bound = upper+steps*(std::numeric_limits<double>::epsilon()*upper+
                            count*static_cast<long double>(std::numeric_limits<double>::denorm_min()));
        return std::nextafter(static_cast<double>(bound), std::numeric_limits<double>::infinity());
    }
};
} // namespace

double transfer_bound(const Config &ket, const std::array<int, 2> &holes,
                      const std::array<int, 2> &particles, int order,
                      const double *h, const double *v, int k) {
    // A coarse external-state norm bound before creating a target key, term
    // list, projected action, or cache entry. Particle and hole sets are
    // disjoint; order is the minimal spatial occupation transfer rank.
    NormBound bound;
    auto add = [&](int q, int s, int p, int r, double value) {
        bound.add(value, p == r || q == s ? 2 : 4);
    };
    if (order == 2) {
        spatial_coefficients(holes[0], holes[1], particles[0], particles[1], v, k, add);
    } else if (order == 1) {
        bound.add(h[particles[0]*k+holes[0]], 2);
        for (int t = 0; t < k; ++t)
            if (ket[t] > (t == holes[0]))
                spatial_coefficients(std::min(holes[0], t), std::max(holes[0], t),
                                     std::min(particles[0], t), std::max(particles[0], t),
                                     v, k, add);
    } else {
        // The one-body occupation scalar vanishes between distinct labels.
        PairDescriptor(ket,ket).two_body_patterns([&](int q, int s, int p, int r) {
            spatial_coefficients(q,s,p,r,v,k,add);
        });
    }
    return bound.finish(k);
}

namespace {
template<bool Ranked,bool DiagonalERI=false,class ERI>
void fill_guga_connection(const PairDescriptor &pair,const double *h,ERI v,
                         bool compute_bounds,Connection &result) {
    const auto &bra=pair.bra,&ket=pair.ket;int k=int(bra.size());
    result.scalar=0.;result.term_bound=0.;
    result.generators.clear();
    if constexpr (Ranked) {
        size_t reserve=pair.transfer_rank==2 ? 2 : pair.transfer_rank==1 ? 2*size_t(k)+1 : size_t(k)*(k-1)/2;
        reserve=std::min(reserve,size_t(128));
        if (result.generators.capacity()<reserve) result.generators.reserve(reserve);
    }
    NormBound bound, original;
    size_t single_index=std::numeric_limits<size_t>::max(),last_two=single_index;
    auto append = [&](double coefficient, int p, int q, int r, int s, bool two_body) {
        if (coefficient == 0.0) return;
        if (compute_bounds) original.add(coefficient, two_body ? 4+2*(q == r) : 2);
        // Project number operators onto the fixed occupations. These
        // identities apply to every spin path of the occupation pair.
        if (Ranked && pair.transfer_rank==2) {
            // Disjoint holes and particles exclude number operators and
            // same-occupation exchange. Retain canonical pair aggregation.
        } else if (two_body && p == q) {
            coefficient *= bra[p]-(q == r);
            p = r; q = s; two_body = false;
        } else if (two_body && r == s) {
            coefficient *= ket[r]-(q == r);
            two_body = false;
        } else if (two_body && p == s && q == r &&
                   (ket[p] != 1 || ket[q] != 1)) {
            int factor = !ket[p] || !ket[q] ? 0 : (ket[p] == 2 && ket[q] == 2 ? -2 : -1);
            result.scalar += coefficient*factor;
            return;
        }
        if (!two_body && p == q) {
            result.scalar += coefficient*ket[p];
            return;
        }
        if (coefficient == 0.0) return;
        if (!two_body) r = s = 0;
        // e[p,q,r,s] == e[r,s,p,q], including coincident indices.
        else if (std::pair<int,int>(p,q) > std::pair<int,int>(r,s)) {
            std::swap(p,r); std::swap(q,s);
        }
        if constexpr (Ranked) {
            // Rank one has one common E[p,q] spectator sum. At ranks zero
            // and one, every spectator yields one unique canonical e term;
            // its second orbital permutation can only repeat the last e term.
            // Keep the raw addition order without linear search.
            if (!two_body && pair.transfer_rank==1) {
                if (single_index==std::numeric_limits<size_t>::max()) {
                    single_index=result.generators.size();result.generators.push_back({coefficient,p,q,0,0,false});
                } else result.generators[single_index].coefficient+=coefficient;
                return;
            }
            if (two_body && pair.transfer_rank<=1) {
                if (last_two!=std::numeric_limits<size_t>::max()) {
                    auto &term=result.generators[last_two];
                    if (term.p==p && term.q==q && term.r==r && term.s==s) { term.coefficient+=coefficient;return; }
                }
                last_two=result.generators.size();result.generators.push_back({coefficient,p,q,r,s,true});return;
            }
        }
        for (auto &term : result.generators)
            if (term.p == p && term.q == q && term.r == r && term.s == s && term.two_body == two_body) {
                term.coefficient += coefficient;
                return;
            }
        result.generators.push_back({coefficient,p,q,r,s,two_body});
    };
    pair.one_body_patterns([&](int p,int q) {
        if (p==q) {
            double value=h[p*k+p]*ket[p];
            result.scalar+=value;
            if (compute_bounds) original.add(value,1);
        } else append(h[p*k+q],p,q,0,0,false);
    });
    pair.two_body_patterns([&](int q, int s, int a, int b) {
        spatial_coefficients<DiagonalERI>(q,s,a,b,v,k,[&](int qq,int ss,int p,int r,double coefficient) {
            append(coefficient,p,qq,r,ss,true);
        });
    });
    result.generators.erase(std::remove_if(result.generators.begin(),result.generators.end(),
        [](const auto &term) { return term.coefficient == 0.0; }),result.generators.end());
    // Only the HCI selector needs a per-connection norm bound.
    if (!compute_bounds) {
        result.term_bound=std::numeric_limits<double>::infinity();
        return;
    }
    for (const auto &term : result.generators)
        bound.add_scaled(term.coefficient,guga_local_bound(ket,term.p,term.q,term.r,term.s,term.two_body));
    // Also cover coefficient/scalar aggregation relative to the original
    // integral terms. Never omit this when cancellations tighten the bound.
    long double steps = 512.0L*(k+1)+16.0L*(original.count+1);
    long double n_eps = steps*std::numeric_limits<double>::epsilon();
    double error = n_eps < 0.25L ? static_cast<double>(n_eps*original.finish(k)) :
                   std::numeric_limits<double>::infinity();
    error += static_cast<double>(steps*(original.count+1)*std::numeric_limits<double>::denorm_min());
    result.term_bound = std::nextafter(bound.finish(k)+error,std::numeric_limits<double>::infinity());
}
} // namespace

Connection guga_diagonal_connection(const Config &cfg,const double *h,const double *coulomb_exchange) {
    Connection result;
    fill_guga_connection<false,true>(PairDescriptor(cfg,cfg),h,coulomb_exchange,false,result);
    return result;
}

static void ranked_dense_connection(const PairDescriptor &pair,const double *h,const double *v,
                                    bool compute_bounds,Connection &result) {
    if (!pair.connected()) { result=Connection{};return; }
    if (pair.transfer_rank!=2 || compute_bounds) {
        fill_guga_connection<true>(pair,h,v,compute_bounds,result);return;
    }
    // Pure double transfer: disjoint hole/particle sets permit at most two
    // canonical two-electron generators. Sum every integral permutation in
    // its original order.
    result.scalar=0.;result.generators.clear();
    result.term_bound=std::numeric_limits<double>::infinity();
    if (result.generators.capacity()<2) result.generators.reserve(2);
    int k=int(pair.bra.size());
    spatial_coefficients(pair.holes[0],pair.holes[1],pair.particles[0],pair.particles[1],v,k,
        [&](int q,int s,int p,int r,double coefficient) {
            if (std::pair<int,int>(p,q)>std::pair<int,int>(r,s)) { std::swap(p,r);std::swap(q,s); }
            for (auto &term:result.generators) if (term.p==p && term.q==q && term.r==r && term.s==s) {
                term.coefficient+=coefficient;return;
            }
            result.generators.push_back({coefficient,p,q,r,s,true});
        });
    result.generators.erase(std::remove_if(result.generators.begin(),result.generators.end(),
        [](const auto &term) { return term.coefficient==0.; }),result.generators.end());
}

void ranked_connection(const PairDescriptor &pair,const double *h,const structured::SpatialERI &v,
                       bool compute_bounds,Connection &result) {
    if(v.format==structured::ERIFormat::dense) {
        ranked_dense_connection(pair,h,v.data,compute_bounds,result);
        return;
    }
    if(!pair.connected()) {result=Connection{};return;}
    fill_guga_connection<true>(pair,h,v,compute_bounds,result);
}

} // namespace scsf

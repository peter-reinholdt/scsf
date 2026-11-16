/* This file is part of SCSF, licensed under the GNU GPL version 3 or later.
 * See LICENSE for details. */

// CSFs are chains of local Clebsch--Gordan tensors. Contract local fermionic
// operator strings with two such chains; no determinant supports are generated.
#include "csf.h"
#include "pair_descriptor.h"
#include <pybind11/pybind11.h>
#include <map>
#include <algorithm>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace scsf {

// Operators are passed in execution order (rightmost operator first).
double contract(const Path &bra, const Path &ket, const std::array<Operator, 4> &ops,
                int count, int two_s) {
    std::vector<double> current(1, 1.0), next;
    for (size_t p = 0; p < bra.sites.size(); ++p) {
        const Site &a = bra.sites[p], &b = ket.sites[p];
        next.assign(static_cast<size_t>(a.after+1)*(b.after+1), 0.0);
        bool nonzero = false;
        for (int t = 0; t < b.nstate; ++t) {
            int local = b.state[t], sign = 1;
            for (int o = 0; o < count && sign; ++o) {
                const Operator &op = ops[o];
                if (static_cast<int>(p) < op.orbital) {
                    if (local == 1 || local == 2) sign = -sign;
                } else if (static_cast<int>(p) == op.orbital) {
                    bool occupied = (local & (1 << op.spin)) != 0;
                    if (occupied == op.create) {
                        sign = 0;
                    } else {
                        // In the local double, alpha precedes beta.
                        if (op.spin == 1 && (local & 1)) sign = -sign;
                        local ^= 1 << op.spin;
                    }
                }
            }
            if (!sign) continue;
            int u = 0;
            while (u < a.nstate && a.state[u] != local) ++u;
            if (u == a.nstate) continue;
            for (int i = 0; i <= a.before; ++i) {
                double ca = sign*a.coefficient[u][i];
                if (ca == 0.0) continue;
                int ni = a.destination[u][i];
                for (int j = 0; j <= b.before; ++j) {
                    double value = ca*b.coefficient[t][j]*current[i*(b.before+1)+j];
                    if (value != 0.0) {
                        int nj = b.destination[t][j];
                        next[ni*(b.after+1)+nj] += value;
                        nonzero = true;
                    }
                }
            }
        }
        if (!nonzero) return 0.0;
        current.swap(next);
    }
    return current[static_cast<size_t>(two_s)*(two_s+1)+two_s];
}

// Compile-time local projection has already handled parity, Pauli zeros, and
// local states. Only orbitals that are singly occupied on either side carry
// spin; closed/empty sites are identity CG transfers and can be omitted.
// Reuse two bounded buffers over all actions of a CSF pair.
double contract_compiled(const Path &bra, const Path &ket, int two_s,
                         const Connection &connection, size_t action,
                         std::vector<double> &current, std::vector<double> &next) {
    current.assign(1, 1.0);
    size_t offset = action*2*connection.open_sites.size();
    for (int p : connection.open_sites) {
        const Site &a = bra.sites[p], &b = ket.sites[p];
        next.assign(static_cast<size_t>(a.after+1)*(b.after+1), 0.0);
        bool nonzero = false;
        for (int t = 0; t < b.nstate; ++t) {
            int code = connection.local_maps[offset+t];
            if (!code) continue;
            int u = std::abs(code)-1, sign = code > 0 ? 1 : -1;
            for (int i = 0; i <= a.before; ++i) {
                double ca = sign*a.coefficient[u][i];
                if (ca == 0.0) continue;
                int ni = a.destination[u][i];
                for (int j = 0; j <= b.before; ++j) {
                    double value = ca*b.coefficient[t][j]*current[i*(b.before+1)+j];
                    if (value != 0.0) {
                        int nj = b.destination[t][j];
                        next[ni*(b.after+1)+nj] += value;
                        nonzero = true;
                    }
                }
            }
        }
        if (!nonzero) return 0.0;
        offset += 2;
        current.swap(next);
    }
    return connection.action_signs[action]*current[static_cast<size_t>(two_s)*(two_s+1)+two_s];
}

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

void check_bonds(const Path &bra, const Path &ket, size_t maximum) {
    if (!bra.spins.empty()) {
        if (bra.spins.size() > 1 && maximum < 2)
            throw std::length_error("GUGA segment scratch exceeds max_bond_elements");
        return;
    }
    for (size_t p = 0; p < bra.sites.size(); ++p) {
        size_t size = static_cast<size_t>(bra.sites[p].after+1)*(ket.sites[p].after+1);
        if (size > maximum)
            throw std::length_error("spin contraction exceeds max_bond_elements");
    }
}

namespace {
// Visit spatial orderings once for both the cheap norm bound and the exact
// term builder. This avoids a second set of spin/integral multiplicity rules.
template<class Callback>
void spatial_coefficients(int q, int s, int a, int b, const double *v,
                          int k, Callback callback) {
    for (int hp = 0; hp < (q == s ? 1 : 2); ++hp) {
        int qq = hp == 0 ? q : s, ss = hp == 0 ? s : q;
        for (int pp = 0; pp < (a == b ? 1 : 2); ++pp) {
            int p = pp == 0 ? a : b, r = pp == 0 ? b : a;
            size_t index = ((static_cast<size_t>(p)*k+r)*k+qq)*k+ss;
            double integral = 0.5*v[index];
            if (integral != 0.0) callback(qq, ss, p, r, integral);
        }
    }
}

template<class Callback>
void two_body_coefficients(const Config &bra, const Config &ket, const double *v,
                           int k, Callback callback) {
    PairDescriptor(bra,ket).two_body_patterns([&](int q, int s, int a, int b) {
        spatial_coefficients(q, s, a, b, v, k, callback);
    });
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
        two_body_coefficients(ket, ket, v, k, add);
    }
    return bound.finish(k);
}

double prescreen_bound(const Config &bra, const Config &ket, const double *h,
                       const double *v, int k) {
    std::array<int, 2> holes{}, particles{};
    int nh = 0, np = 0;
    for (int p = 0; p < k; ++p) {
        int diff = bra[p]-ket[p];
        while (diff < 0) {
            if (nh == 2) return 0.0;
            holes[nh++] = p;
            ++diff;
        }
        while (diff > 0) {
            if (np == 2) return 0.0;
            particles[np++] = p;
            --diff;
        }
    }
    if (nh != np) return 0.0;
    return transfer_bound(ket, holes, particles, nh, h, v, k);
}

namespace {
template<bool Ranked>
void fill_guga_connection(const PairDescriptor &pair,const double *h,const double *v,
                         size_t max_terms,bool compute_bounds,Connection &result) {
    const auto &bra=pair.bra,&ket=pair.ket;int k=int(bra.size());
    result.guga=true;result.scalar=0.;result.bound=result.term_bound=0.;
    result.guga_input_terms=0;result.generators.clear();
    if constexpr (Ranked) {
        size_t reserve=pair.transfer_rank==2 ? 2 : pair.transfer_rank==1 ? 2*size_t(k)+1 : size_t(k)*(k-1)/2;
        reserve=std::min({reserve,max_terms,size_t(128)});
        if (result.generators.capacity()<reserve) result.generators.reserve(reserve);
    }
    NormBound bound, original;
    size_t single_index=std::numeric_limits<size_t>::max(),last_two=single_index;
    auto append = [&](double coefficient, int p, int q, int r, int s, bool two_body) {
        if (coefficient == 0.0) return;
        if (result.guga_input_terms >= max_terms)
            throw std::length_error("operator terms exceed max_connection_terms");
        ++result.guga_input_terms;
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
            // Keep the raw addition order and guard count, without linear search.
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
        spatial_coefficients(q,s,a,b,v,k,[&](int qq,int ss,int p,int r,double coefficient) {
            append(coefficient,p,qq,r,ss,true);
        });
    });
    result.generators.erase(std::remove_if(result.generators.begin(),result.generators.end(),
        [](const auto &term) { return term.coefficient == 0.0; }),result.generators.end());
    // Direct late-screened products still assemble exactly the same
    // Hamiltonian element, but need no per-connection norm certificate.
    // Infinity disables every accidental bound-based rejection upstream.
    if (!compute_bounds) {
        result.bound=result.term_bound=std::numeric_limits<double>::infinity();
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
    result.bound = std::nextafter(result.term_bound+std::abs(result.scalar)+error,
                                  std::numeric_limits<double>::infinity());
    return;

}
} // namespace

void ranked_connection(const PairDescriptor &pair,const double *h,const double *v,
                       size_t max_terms,bool compute_bounds,Connection &result) {
    if (!pair.connected()) { result=Connection{};result.guga=true;return; }
    if (pair.transfer_rank!=2 || compute_bounds) {
        fill_guga_connection<true>(pair,h,v,max_terms,compute_bounds,result);return;
    }
    // Pure double transfer: disjoint hole/particle sets permit at most two
    // canonical two-electron generators. Preserve every integral permutation,
    // addition order, exact zero and raw-term guard; assume no integral symmetry.
    result.guga=true;result.scalar=0.;result.guga_input_terms=0;result.generators.clear();
    result.bound=result.term_bound=std::numeric_limits<double>::infinity();
    if (result.generators.capacity()<2) result.generators.reserve(2);
    int k=int(pair.bra.size());
    spatial_coefficients(pair.holes[0],pair.holes[1],pair.particles[0],pair.particles[1],v,k,
        [&](int q,int s,int p,int r,double coefficient) {
            if (result.guga_input_terms>=max_terms) throw std::length_error("operator terms exceed max_connection_terms");
            ++result.guga_input_terms;
            if (std::pair<int,int>(p,q)>std::pair<int,int>(r,s)) { std::swap(p,r);std::swap(q,s); }
            for (auto &term:result.generators) if (term.p==p && term.q==q && term.r==r && term.s==s) {
                term.coefficient+=coefficient;return;
            }
            result.generators.push_back({coefficient,p,q,r,s,true});
        });
    result.generators.erase(std::remove_if(result.generators.begin(),result.generators.end(),
        [](const auto &term) { return term.coefficient==0.; }),result.generators.end());
}

Connection connection(const Config &bra, const Config &ket, const double *h,
                      const double *v, int k, size_t max_terms, bool guga, bool compute_bounds) {
    if (bra.size()!=static_cast<size_t>(k))
        throw std::invalid_argument("configuration dimensions differ from orbital count");
    return connection(PairDescriptor(bra,ket),h,v,max_terms,guga,compute_bounds);
}

Connection connection(const PairDescriptor &pair, const double *h, const double *v,
                      size_t max_terms, bool guga, bool compute_bounds) {
    const auto &bra=pair.bra,&ket=pair.ket;
    int k=static_cast<int>(bra.size());
    Connection result;
    result.guga = guga;
    if (!pair.connected()) return result;
    if (guga) {
        fill_guga_connection<false>(pair,h,v,max_terms,compute_bounds,result);return result;
    }
    long double norm_bound = 0.0L, scalar_bound = 0.0L;
    const long double infinity = std::numeric_limits<long double>::infinity();
    auto add_bound = [&](double value) {
        if (value != 0.0)
            norm_bound = std::nextafter(norm_bound+std::abs(static_cast<long double>(value)),
                                        infinity);
    };
    auto append = [&](double coefficient, const std::array<Operator, 4> &operators, int n) {
        if (result.terms.size() >= max_terms)
            throw std::length_error("operator terms exceed max_connection_terms");
        result.terms.push_back({coefficient, operators, n});
        add_bound(coefficient);
    };
    std::array<Operator, 4> ops{};
    pair.one_body_patterns([&](int particle,int hole) {
        if (particle==hole) {
            double diagonal=h[particle*k+particle]*ket[particle];
            result.scalar+=diagonal;
            if (diagonal!=0.0)
                scalar_bound=std::nextafter(scalar_bound+std::abs(static_cast<long double>(diagonal)),infinity);
        } else if (h[particle*k+hole]!=0.0) {
            for (int spin=0;spin<2;++spin) {
                ops[0]={hole,spin,false};
                ops[1]={particle,spin,true};
                append(h[particle*k+hole],ops,2);
            }
        }
    });
    pair.two_body_patterns([&](int q0, int s0, int a, int b) {
        spatial_coefficients(q0,s0,a,b,v,k,[&](int q, int s, int p, int r, double integral) {
            for (int sigma = 0; sigma < 2; ++sigma) {
                for (int tau = 0; tau < 2; ++tau) {
                    ops = {{{q, sigma, false}, {s, tau, false},
                            {r, tau, true}, {p, sigma, true}}};
                    if ((p == r && sigma == tau) || (q == s && sigma == tau))
                        continue;
                    append(integral, ops, 4);
                }
            }
        });
    });
    auto outward = [](long double bound) {
        double value = static_cast<double>(bound);
        if (bound > 0.0L && std::isfinite(value))
            value = std::nextafter(value, std::numeric_limits<double>::infinity());
        return value;
    };
    // Project each monomial onto the fixed local occupations. Its local
    // factors are partial signed permutations, with norm at most one. Equal
    // products (up to a global sign) share a contraction and can cancel in a
    // certified bound. This uses O(k*terms) work, never determinant supports.
    struct Group {
        size_t representative;
        int sign;
        long double low = 0.0L, high = 0.0L;
    };
    std::map<std::vector<signed char>, Group> actions;
    for (int p = 0; p < k; ++p)
        if (bra[p] == 1 || ket[p] == 1) result.open_sites.push_back(p);
    std::vector<int> prefix(k+1, 0);
    std::vector<signed char> identity;
    identity.reserve(2*k);
    for (int p = 0; p < k; ++p) {
        prefix[p+1] = prefix[p]+ket[p];
        identity.push_back(ket[p] == 0 ? 1 : ket[p] == 2 ? 4 : 2);
        identity.push_back(ket[p] == 1 ? 3 : 0);
    }
    for (size_t index = 0; index < result.terms.size(); ++index) {
        Term &term = result.terms[index];
        std::vector<signed char> key = identity;
        // Jordan--Wigner parity outside the operated orbitals is a scalar
        // fixed by spatial occupations. Earlier operators toggle that prefix
        // parity, independently of creation/annihilation or alpha/beta spin.
        int parity = 0;
        std::array<int, 4> active{};
        int nactive = 0;
        for (int o = 0; o < term.count; ++o) {
            int p = term.operators[o].orbital;
            parity ^= prefix[p] & 1;
            for (int t = 0; t < o; ++t) parity ^= term.operators[t].orbital < p;
            if (std::find(active.begin(), active.begin()+nactive, p) == active.begin()+nactive)
                active[nactive++] = p;
        }
        int overall = parity ? -1 : 1;
        bool zero = false;
        // Only at most four operated sites need local projection; all other
        // entries are copied identities. Complete fixed-length keys are kept.
        for (int site = 0; site < nactive; ++site) {
            int p = active[site], first_sign = 0;
            int count = ket[p] == 1 ? 2 : 1;
            for (int t = 0; t < count; ++t) {
                int local = ket[p] == 0 ? 0 : ket[p] == 2 ? 3 : 1 << t;
                int sign = 1;
                for (int o = 0; o < term.count && sign; ++o) {
                    const auto &op = term.operators[o];
                    if (p != op.orbital) continue;
                    if (bool(local & (1 << op.spin)) == op.create) sign = 0;
                    else {
                        if (op.spin == 1 && (local & 1)) sign = -sign;
                        local ^= 1 << op.spin;
                    }
                }
                int occupation = (local & 1)+((local >> 1) & 1);
                if (occupation != bra[p]) sign = 0;
                if (sign && !first_sign) first_sign = sign;
                key[2*p+t] = sign*(local+1);
            }
            if (!first_sign) { zero = true; break; }
            for (int t = 0; t < count; ++t) key[2*p+t] *= first_sign;
            overall *= first_sign;
        }
        if (zero) { term.sign = 0; continue; }
        auto found = actions.find(key);
        if (found == actions.end()) {
            size_t r = result.representatives.size();
            result.representatives.push_back(index);
            result.action_signs.push_back(overall);
            for (int p : result.open_sites) {
                for (int t = 0; t < 2; ++t) {
                    int value = key[2*p+t];
                    int local = std::abs(value)-1;
                    int u = bra[p] == 1 ? (local == 1 ? 0 : 1) : 0;
                    result.local_maps.push_back(value == 0 ? 0 : (value > 0 ? 1 : -1)*(u+1));
                }
            }
            found = actions.emplace(std::move(key), Group{r, overall}).first;
        }
        auto &group = found->second;
        term.contraction = group.representative;
        term.sign = overall*group.sign;
        long double value = static_cast<long double>(term.coefficient)*overall;
        group.low = std::nextafter(group.low+value, -infinity);
        group.high = std::nextafter(group.high+value, infinity);
    }
    long double full = scalar_bound, external = 0.0L;
    std::map<std::vector<int>, std::pair<long double, long double>> diagonal;
    for (const auto &item : actions) {
        const auto &g = item.second;
        const auto &key = item.first;
        bool is_diagonal = bra == ket;
        for (int p = 0; p < k && is_diagonal; ++p)
            for (int t = 0; t < (ket[p] == 1 ? 2 : 1); ++t) {
                int local = ket[p] == 0 ? 0 : ket[p] == 2 ? 3 : 1 << t;
                if (key[2*p+t] && std::abs(key[2*p+t]) != local+1) is_diagonal = false;
            }
        if (is_diagonal) {
            // n_alpha=(I+Z)/2, n_beta=(I-Z)/2 on a singly occupied
            // orbital. Collect diagonal products in a common Z basis so that
            // spin-summed Coulomb density terms become one occupation scalar.
            // A quartic monomial contains at most two such local projectors.
            std::vector<std::pair<std::vector<int>, long double>> polynomial{{{}, 1.0L}};
            for (int p = 0; p < k; ++p) {
                if (ket[p] != 1) continue;
                auto sign = [](int value) { return (value > 0)-(value < 0); };
                long double a = sign(key[2*p]), b = sign(key[2*p+1]);
                long double identity = (a+b)/2, z = (a-b)/2;
                std::vector<std::pair<std::vector<int>, long double>> next;
                for (auto &term : polynomial) {
                    if (identity) next.push_back({term.first, term.second*identity});
                    if (z) {
                        term.first.push_back(p);
                        next.push_back({std::move(term.first), term.second*z});
                    }
                }
                polynomial = std::move(next);
            }
            for (const auto &term : polynomial) {
                auto &interval = diagonal[term.first];
                long double a = term.second*g.low, b = term.second*g.high;
                interval.first = std::nextafter(interval.first+std::min(a,b), -infinity);
                interval.second = std::nextafter(interval.second+std::max(a,b), infinity);
            }
            continue;
        }
        long double magnitude = std::max(std::abs(g.low), std::abs(g.high));
        full = std::nextafter(full+magnitude, infinity);
        external = std::nextafter(external+magnitude, infinity);
    }
    for (const auto &term : diagonal) {
        long double magnitude = std::max(std::abs(term.second.first), std::abs(term.second.second));
        full = std::nextafter(full+magnitude, infinity);
        if (!term.first.empty()) external = std::nextafter(external+magnitude, infinity);
    }
    // Include an error envelope for the unchanged floating-point contraction
    // and original-order term sum, including identity terms excluded above.
    long double steps = 512.0L*(k+1)+8.0L*(result.terms.size()+1);
    long double tiny = std::numeric_limits<double>::denorm_min();
    long double error = steps*(std::numeric_limits<double>::epsilon()*norm_bound+
                               result.terms.size()*tiny);
    result.term_bound = outward(external+error);
    result.bound = outward(full+error+steps*std::numeric_limits<double>::epsilon()*scalar_bound);
    return result;
}

double element(const Path &bra, const Path &ket, int two_s, const Connection &connection,
               bool compiled) {
    if (connection.guga) return guga_element(bra, ket, connection);
    double value = bra.label == ket.label ? connection.scalar : 0.0;
    std::vector<double> contractions;
    contractions.reserve(connection.representatives.size());
    std::vector<double> current, next;
    if (compiled && !connection.representatives.empty()) {
        size_t maximum = 1;
        for (int p : connection.open_sites)
            maximum = std::max(maximum, static_cast<size_t>(bra.sites[p].after+1)*(ket.sites[p].after+1));
        current.reserve(maximum);
        next.reserve(maximum);
    }
    for (size_t r = 0; r < connection.representatives.size(); ++r) {
        const auto &term = connection.terms[connection.representatives[r]];
        contractions.push_back(compiled ? contract_compiled(bra, ket, two_s, connection, r, current, next) :
                               contract(bra, ket, term.operators, term.count, two_s));
    }
    for (const Term &term : connection.terms)
        if (term.sign)
            value += term.coefficient*(term.sign*contractions[term.contraction]);
    return value;
}

Array block(const Array &one, const Array &two, const std::vector<std::string> &rows,
            const std::vector<std::string> &cols, int two_s, size_t max_bond_elements,
            size_t max_tensor_entries, size_t max_connection_terms,
            std::shared_ptr<ConnectionCache> cache, bool compiled, bool guga, int guga_mode) {
    if (one.ndim() != 2 || one.shape(0) != one.shape(1) ||
        one.shape(0) > std::numeric_limits<int>::max() || two_s < 0)
        throw std::invalid_argument("invalid orbital dimensions or total spin");
    int k = static_cast<int>(one.shape(0));
    if (guga_mode < 0 || guga_mode > 2) throw std::invalid_argument("invalid guga_mode");
    if (cache) cache->validate(one, two);
    if (two.ndim() != 4 || two.shape(0) != k || two.shape(1) != k ||
        two.shape(2) != k || two.shape(3) != k)
        throw std::invalid_argument("two-electron array must have four spatial-orbital axes");
    std::vector<Path> left, right;
    for (const auto &label : rows) left.emplace_back(label, k, two_s, max_tensor_entries, !guga, guga_mode);
    for (const auto &label : cols) right.emplace_back(label, k, two_s, max_tensor_entries, !guga, guga_mode);
    for (const auto &a : left) {
        for (const auto &b : right) {
            if (a.electrons != b.electrons)
                throw std::invalid_argument("CSF block must use one electron-number sector");
            check_bonds(a, b, max_bond_elements);
        }
    }
    Array result({static_cast<py::ssize_t>(rows.size()), static_cast<py::ssize_t>(cols.size())});
    double *out = result.mutable_data();
    const double *h = one.data(), *v = two.data();
    {
        py::gil_scoped_release release;
        std::map<Config, std::vector<size_t>> left_groups, right_groups;
        for (size_t i = 0; i < rows.size(); ++i)
            left_groups[configuration(rows[i], two_s)].push_back(i);
        for (size_t j = 0; j < cols.size(); ++j)
            right_groups[configuration(cols[j], two_s)].push_back(j);
        for (const auto &a : left_groups) {
            for (const auto &b : right_groups) {
                auto terms = cache ? cache->get(a.first, b.first, h, v, k, max_connection_terms, guga) :
                    std::make_shared<const Connection>(connection(a.first, b.first, h, v, k, max_connection_terms, guga));
                for (size_t i : a.second)
                    for (size_t j : b.second)
                        out[i*right.size()+j] = element(left[i], right[j], two_s, *terms, compiled);
            }
        }
    }
    return result;
}
} // namespace scsf

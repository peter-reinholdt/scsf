/* SCSF; GPL-3.0-or-later. Integral-independent spin matrices for structured CI. */
#pragma once
#include "csf.h"
#include <algorithm>
#include <limits>
#include <numeric>

namespace scsf { namespace structured {

struct SpinPaths {
    int m=0,s=0;
    std::vector<std::string> paths;
    std::vector<std::vector<size_t>> ways;
    SpinPaths(int count,int spin,size_t maximum) : m(count),s(spin) {
        if (m<0 || s<0 || s>m || (m-s)%2) return;
        ways.assign(m+1,std::vector<size_t>(m+2,0));ways[0][s]=1;
        for (int n=1;n<=m;++n) for (int j=0;j<=m;++j) {
            size_t a=ways[n-1][j+1],b=j ? ways[n-1][j-1] : 0;
            if (a>maximum || b>maximum-a) throw BudgetError("spin class exceeds coefficient budget");
            ways[n][j]=a+b;
        }
        std::string path;
        auto visit=[&](auto &&self,int j,int left)->void {
            if (!left) { if (j==s) paths.push_back(path);return; }
            if (ways[left-1][j+1]) { path+='u';self(self,j+1,left-1);path.pop_back(); }
            if (j && ways[left-1][j-1]) { path+='d';self(self,j-1,left-1);path.pop_back(); }
        };
        visit(visit,0,m);
    }
    size_t rank(const std::string &path) const {
        size_t r=0;int j=0,left=m;
        for (char c:path) if (c=='u' || c=='d') {
            --left;
            if (left<0) throw std::invalid_argument("too many spin steps");
            if (c=='d') { r+=ways[left][j+1];--j; } else ++j;
            if (j<0) throw std::invalid_argument("invalid spin path");
        }
        if (left || j!=s || r>=paths.size()) throw std::invalid_argument("spin path sector mismatch");
        return r;
    }
};

struct Matrix {
    size_t rows=0,cols=0;
    std::vector<double> values,squared_columns;
    // Sparse rows avoid applying exact structural zeros in matrix factors.
    std::vector<size_t> ptr;
    std::vector<unsigned> indices;
    std::vector<double> data;
    Matrix()=default;
    Matrix(size_t a,size_t b) : rows(a),cols(b),values(a*b,0.) {}
    double &at(size_t a,size_t b) { return values[a*cols+b]; }
    double at(size_t a,size_t b) const { return values[a*cols+b]; }
    void sparse() {
        ptr.assign(1,0);indices.clear();data.clear();
        for (size_t a=0;a<rows;++a) {
            for (size_t b=0;b<cols;++b) if (at(a,b)!=0.) { indices.push_back(b);data.push_back(at(a,b)); }
            ptr.push_back(data.size());
        }
    }
    void make_squared_columns() {
        squared_columns.assign(cols,0.);
        for(size_t row=0;row<rows;++row)
            for(size_t j=ptr[row];j<ptr[row+1];++j)
                squared_columns[indices[j]]+=data[j]*data[j];
    }
    size_t bytes() const { return (values.capacity()+squared_columns.capacity())*8+ptr.capacity()*sizeof(size_t)+indices.capacity()*sizeof(unsigned)+data.capacity()*8; }
};

inline Matrix identity(size_t n) {
    Matrix a(n,n);for(size_t i=0;i<n;++i)a.at(i,i)=1.;return a;
}
inline Matrix transpose(const Matrix &a) {
    Matrix b(a.cols,a.rows);
    for(size_t i=0;i<a.rows;++i)for(size_t j=0;j<a.cols;++j)b.at(j,i)=a.at(i,j);
    return b;
}
inline Matrix multiply(const Matrix &a,const Matrix &b) {
    if(a.cols!=b.rows) throw std::logic_error("spin matrix dimension mismatch");
    Matrix c(a.rows,b.cols);
    for(size_t i=0;i<a.rows;++i)for(size_t k=0;k<a.cols;++k)if(a.at(i,k)!=0.)
        for(size_t j=0;j<b.cols;++j)c.at(i,j)+=a.at(i,k)*b.at(k,j);
    return c;
}

// Yamanouchi adjacent-transposition representation. UD/DU paths form 2x2
// blocks; equal steps are symmetric. Row operations apply a sparse factor.
inline Matrix spin_permutation(const SpinPaths &basis,const std::vector<int> &order) {
    Matrix u=identity(basis.paths.size());std::vector<int> current(order.size());
    std::iota(current.begin(),current.end(),0);
    for(size_t i=0;i<order.size();++i) {
        size_t j=i;while(j<current.size() && current[j]!=order[i])++j;
        if(j==current.size())throw std::logic_error("invalid spin permutation");
        for(;j>i;--j) {
            size_t at=j-1;Matrix v(u.rows,u.cols);
            for(size_t row=0;row<u.rows;++row) {
                const auto &p=basis.paths[row];
                if(p[at]==p[at+1]) { for(size_t col=0;col<u.cols;++col)v.at(row,col)=u.at(row,col);continue; }
                int before=0;for(size_t d=0;d<at;++d)before+=(p[d]=='u') ? 1 : -1;
                double a=1./(before+1),diagonal=p[at]=='u' ? -a : a;
                std::string other=p;std::swap(other[at],other[at+1]);
                bool valid=before>0;size_t partner=valid ? basis.rank(other) : row;
                double off=valid ? std::sqrt(std::max(0.,1.-a*a)) : 0.;
                for(size_t col=0;col<u.cols;++col)v.at(row,col)=diagonal*u.at(row,col)+off*u.at(partner,col);
            }
            u=std::move(v);std::swap(current[j-1],current[j]);
        }
    }
    return u;
}

// Fermionic action in orbital-major local order. Closed-site parity is even,
// allowing common closed/empty spectators to be removed from canonical classes.
inline int act(std::vector<int> &state,int orbital,int spin,bool create) {
    int bit=1<<spin;
    if(bool(state[orbital]&bit)==create)return 0;
    int parity=0;for(int p=0;p<orbital;++p)parity+=__builtin_popcount(unsigned(state[p]));
    if(spin && (state[orbital]&1))++parity;
    state[orbital]^=bit;return parity%2 ? -1 : 1;
}
inline std::vector<int> alpha_state(const Config &cfg) {
    std::vector<int> x;for(int n:cfg)x.push_back(n==2 ? 3 : n);return x;
}
inline int move_sign(const Config &ket,int p,int q) {
    auto target=ket;++target[p];--target[q];auto wanted=alpha_state(target);
    for(int spin=0;spin<2;++spin) {
        auto d=alpha_state(ket);int sign=act(d,q,spin,false);if(sign)sign*=act(d,p,spin,true);
        if(sign && d==wanted)return sign;
    }
    throw std::logic_error("missing one-electron reference sign");
}

inline Matrix one_body(const Config &ket,int p,int q,int s,size_t maximum) {
    int m=std::count(ket.begin(),ket.end(),1);SpinPaths source(m,s,maximum);
    if(source.paths.size() && source.paths.size()>maximum/source.paths.size())throw BudgetError("permutation matrix exceeds coefficient budget");
    if(p==q) { auto a=identity(source.paths.size());for(auto &v:a.values)v*=ket[p];a.sparse();return a; }
    if(ket[p]>=2 || !ket[q])return Matrix();
    Config bra=ket;++bra[p];--bra[q];
    SpinPaths target(std::count(bra.begin(),bra.end(),1),s,maximum);
    if(source.paths.size() && target.paths.size()>maximum/source.paths.size())throw BudgetError("coupling matrix exceeds coefficient budget");
    if(ket[p]==0 && ket[q]==2) { auto reverse=one_body(bra,q,p,s,maximum);auto a=transpose(reverse);a.sparse();return a; }
    std::vector<int> sites;for(size_t i=0;i<ket.size();++i)if(ket[i]==1)sites.push_back(i);
    Matrix result;
    if(ket[p]!=ket[q]) {
        std::vector<int> order;
        int from=ket[q]==1 ? q : p,to=ket[q]==1 ? p : q;
        for(size_t i=0;i<bra.size();++i)if(bra[i]==1) {
            int old=int(i)==to ? from : int(i);
            order.push_back(std::find(sites.begin(),sites.end(),old)-sites.begin());
        }
        result=spin_permutation(source,order);int sign=move_sign(ket,p,q);
        for(auto &v:result.values)v*=sign;
    } else {
        int first=std::min(p,q),last=std::max(p,q);std::vector<int> order;
        for(size_t i=0;i<sites.size();++i)if(sites[i]!=first && sites[i]!=last)order.push_back(i);
        order.push_back(std::find(sites.begin(),sites.end(),first)-sites.begin());
        order.push_back(std::find(sites.begin(),sites.end(),last)-sites.begin());
        auto u=spin_permutation(source,order);Matrix projection(target.paths.size(),source.paths.size());
        // E acting on an ordered singlet pair supplies sqrt(2). Determine its
        // occupation-only sign from one spin assignment, not a determinant sum.
        auto d=alpha_state(ket);d[first]=1;d[last]=2;auto wanted=alpha_state(bra);
        int sign=0;
        for(int spin=0;spin<2;++spin) { auto tmp=d;int v=act(tmp,q,spin,false);if(v)v*=act(tmp,p,spin,true);if(v && tmp==wanted)sign=v; }
        if(!sign)throw std::logic_error("missing singlet projection sign");
        for(size_t col=0;col<source.paths.size();++col) {
            const auto &path=source.paths[col];auto suffix=path.substr(path.size()-2);
            if(suffix!="ud" && suffix!="du")continue;
            size_t row=target.rank(path.substr(0,path.size()-2));
            projection.at(row,col)=sign*(suffix=="ud" ? std::sqrt(double(s+2)/(s+1)) : -std::sqrt(double(s)/(s+1)));
        }
        result=multiply(projection,u);
    }
    result.sparse();return result;
}

class OneBank {
    int n_,side_,spin_;
    size_t maximum_,bytes_=0;
    std::vector<std::unique_ptr<Matrix>> tables_;
    size_t address(int m,int type,int p,int q) const { return ((size_t(m)*4+type)*side_+p)*side_+q; }
    static size_t directory_count(int n,size_t maximum) {
        size_t count=1,limit=maximum/sizeof(std::unique_ptr<Matrix>);
        for(size_t factor:{size_t(n)+1,size_t(4),size_t(n)+3,size_t(n)+3}) {
            if(factor>limit/count)throw BudgetError("one-electron matrix directory exceeds coefficient budget");
            count*=factor;
        }
        return count;
    }
public:
    static constexpr int mask_electron_limit=60;
    OneBank(int n,int spin,size_t maximum,bool norms=false):n_(n),side_(n+3),spin_(spin),maximum_(maximum),
        tables_(directory_count(n,maximum)) {
        bytes_=tables_.capacity()*sizeof(tables_[0]);
        if(bytes_>maximum_)throw BudgetError("one-electron matrix directory exceeds coefficient budget");
        for(int m=spin;m<=n;m+=2)for(int type=0;type<4;++type) {
            if(type!=0 && m+2>n && type!=2)continue;
            int length=m+(type==2 ? 0 : type==3 ? 2 : 1);
            for(int p=0;p<length;++p)for(int q=0;q<length;++q)if(p!=q) {
                Config ket(length,1);ket[p]=(type==0 || type==3) ? 0 : 1;ket[q]=(type==1 || type==3) ? 2 : 1;
                if(std::count(ket.begin(),ket.end(),1)!=m)continue;
                auto matrix=std::make_unique<Matrix>(one_body(ket,p,q,spin_,maximum_/8));
                if(norms)matrix->make_squared_columns();
                if(matrix->bytes()>maximum_-bytes_)throw BudgetError("one-electron matrices exceed coefficient budget");
                bytes_+=matrix->bytes();tables_[address(m,type,p,q)]=std::move(matrix);
            }
        }
    }
    const Matrix &get(const Config &ket,int p,int q) const {
        if(p==q || ket[p]>=2 || !ket[q])throw std::logic_error("non-offdiagonal one-electron matrix request");
        int m=0,pi=0,qi=0;
        for(int i=0;i<int(ket.size());++i) {
            if(ket[i]==1)++m;
            if(ket[i]==1 || i==p || i==q) { pi+=i<p;qi+=i<q; }
        }
        int type=ket[p]==0 ? (ket[q]==1 ? 0 : 3) : (ket[q]==1 ? 2 : 1);
        auto &table=tables_.at(address(m,type,pi,qi));
        if(!table)throw std::logic_error("missing canonical one-electron matrix");
        return *table;
    }
    unsigned index(std::uint64_t occupied,std::uint64_t doubled,int p,int q) const {
        const auto open=occupied^doubled;
        const auto pb=std::uint64_t(1)<<p,qb=std::uint64_t(1)<<q;
        const auto active=open|pb|qb;
        const int type=occupied&pb ? (doubled&qb ? 1 : 2) : (doubled&qb ? 3 : 0);
        return address(__builtin_popcountll(open),type,
            __builtin_popcountll(active&(pb-1)),__builtin_popcountll(active&(qb-1)));
    }
    const Matrix &at(unsigned id) const {return *tables_[id];}
    size_t bytes() const { return bytes_; }
};

struct TensorSite {
    int before, after, occupation, nstate;
    std::array<int, 2> state;
    std::array<std::vector<int>, 2> destination;
    std::array<std::vector<double>, 2> coefficient;
};

struct TensorPath {
    std::vector<TensorSite> sites;
    TensorPath(const std::string &steps, int two_s) {
        int j = 0;
        for (char step : steps) {
            TensorSite site;
            site.before = j;
            if (step == 'u' || step == 'd') {
                j += step == 'u' ? 1 : -1;
                site.occupation = 1;
                site.nstate = 2;
                site.state = {1, 2}; // local |alpha>, |beta>
            } else if (step == '0' || step == '2') {
                site.occupation = step == '0' ? 0 : 2;
                site.nstate = 1;
                site.state = {step == '0' ? 0 : 3, 0};
            } else {
                throw std::invalid_argument("CSF symbols must be 0,u,d,2");
            }
            if (j < 0)
                throw std::invalid_argument("negative intermediate CSF spin");
            site.after = j;
            for (int t = 0; t < site.nstate; ++t) {
                for (int i = 0; i <= site.before; ++i) {
                    int m = -site.before + 2*i;
                    int next = i;
                    double coefficient = 1.0;
                    if (site.occupation == 1) {
                        int sigma = t == 0 ? 1 : -1;
                        int next_m = m + sigma;
                        next = (next_m+j)/2;
                        if (std::abs(next_m) > j) {
                            coefficient = 0.0;
                            next = 0;
                        } else if (step == 'u') {
                            coefficient = std::sqrt((site.before+sigma*m+2.0) /
                                                    (2.0*(site.before+1)));
                        } else {
                            coefficient = (sigma == 1 ? -1.0 : 1.0) *
                                std::sqrt((site.before-sigma*m) / (2.0*(site.before+1)));
                        }
                    }
                    site.destination[t].push_back(next);
                    site.coefficient[t].push_back(coefficient);
                }
            }
            sites.push_back(std::move(site));
        }
        if (j != two_s)
            throw std::invalid_argument("CSF final spin differs from two_s");
    }
};

// CG tensor transfer for number-changing operator classes. Magnetic components
// are explicit here; no Slater determinant expansion or support is generated.
inline double cg_operator(const std::string &bra,int sb,int mb,const std::string &ket,int sk,int mk,
                          const std::vector<Operator> &ops) {
    const TensorPath a(bra,sb),b(ket,sk);
    std::vector<double> current(1,1.),next;
    for(size_t p=0;p<a.sites.size();++p) {
        const auto &x=a.sites[p],&y=b.sites[p];next.assign((x.after+1)*(y.after+1),0.);
        for(int t=0;t<y.nstate;++t) {
            int local=y.state[t],sign=1;
            for(auto &op:ops) {
                if(int(p)<op.orbital) { if(__builtin_popcount(unsigned(local))%2)sign=-sign; }
                else if(int(p)==op.orbital) {
                    int bit=1<<op.spin;if(bool(local&bit)==op.create) { sign=0;break; }
                    if(op.spin && (local&1))sign=-sign;
                    local^=bit;
                }
            }
            if(!sign)continue;
            int u=0;while(u<x.nstate && x.state[u]!=local)++u;
            if(u==x.nstate)continue;
            for(int i=0;i<=x.before;++i)for(int j=0;j<=y.before;++j)
                next[x.destination[u][i]*(y.after+1)+y.destination[t][j]]+=sign*x.coefficient[u][i]*y.coefficient[t][j]*current[i*(y.before+1)+j];
        }
        current.swap(next);
    }
    if(std::abs(mb)>sb || std::abs(mk)>sk || (mb+sb)%2 || (mk+sk)%2)return 0.;
    return current[((sb+mb)/2)*(sk+1)+(sk+mk)/2];
}

} } // namespace scsf::structured

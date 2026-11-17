#include "parallel.h"
#include <array>
#include <cassert>
#include <stdexcept>
int main() {
    using scsf::ThreadTeam;
    std::vector<int> cpus;
#ifdef __linux__
    cpu_set_t original;assert(!sched_getaffinity(0,sizeof(original),&original));
    for (int i=0;i<CPU_SETSIZE && cpus.size()<3;++i) if (CPU_ISSET(i,&original)) cpus.push_back(i);
#endif
    size_t count=cpus.empty() ? 3 : cpus.size();
    ThreadTeam team(count,cpus);
    std::array<std::thread::id,3> ids;
    team.apply_workers(count,[&](size_t worker) { ids[worker]=std::this_thread::get_id(); });
    for (size_t active:{size_t(1),count,size_t(1),count}) {
        team.apply_workers(active,[&](size_t worker) {
            assert(worker<active && ids[worker]==std::this_thread::get_id());
#ifdef __linux__
            assert(sched_getcpu()==cpus[worker]);
#endif
        });
    }
    bool caught=false;
    try { team.apply_workers(count,[](size_t worker) { if (worker==0) throw std::runtime_error("worker failure"); }); }
    catch (const std::runtime_error &) { caught=true; }
    assert(caught);
    std::array<size_t,3> calls{};
    team.apply_workers(count,[&](size_t worker) {
        ++calls[worker];assert(ids[worker]==std::this_thread::get_id());
    });
    for (size_t i=0;i<count;++i) assert(calls[i]==1);
    // Legacy dynamic tasks, used by the CSR products, still run exactly once.
    std::array<size_t,31> tasks{};
    team.apply(tasks.size(),[&](size_t task) { ++tasks[task]; });
    for (auto n:tasks) assert(n==1);
#ifdef __linux__
    cpu_set_t after;assert(!sched_getaffinity(0,sizeof(after),&after));
    assert(CPU_EQUAL(&original,&after));
#endif
}

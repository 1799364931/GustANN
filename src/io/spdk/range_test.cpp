#include <cstdlib>
#include <thread>
#include <vector>
#include <string>
#include <sys/time.h>

#include "spdk/env.h"
#ifdef PAGE_SIZE
#undef PAGE_SIZE
#endif
#include "../interface.hpp"

#include "zipf.hpp"

constexpr long PG_SIZE = 4096;

static void bind_core(int core_num) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core_num, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set) != 0) {
    perror("pthread_setaffinity_np");
    exit(-1);
  }
}

static double elapsed() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}


int main() {
  std::vector<std::string> ssds = {
    "0000:8b:00.0",    
    //"0000:8c:00.0",
    //"0000:8d:00.0",
    //"0000:8e:00.0",
    //"0000:22:00.0",
    //"0000:23:00.0"
//    "0000:e1:00.0",
//    "0000:37:00.0",
  };
  const int thread_cnt = 1;
  const int ctx_per_thread = 32;
  const int range = 100 * 1024 * 1024 / 4;
  const int base = 0;
  const int BATCH_SIZE = 1024;
  auto spdk = gustann::create_spdk_loader(ssds, BATCH_SIZE * ctx_per_thread, thread_cnt, ctx_per_thread * thread_cnt);
  
  
  auto worker = [&](int threadid) {
    bind_core(threadid * 2 + 21);
    printf("TID WORKER %d %d: \n", gettid(), sched_getcpu());
    uint8_t *buffer_raw = 
      (uint8_t *) spdk_dma_zmalloc_socket(BATCH_SIZE * PG_SIZE * ctx_per_thread, PG_SIZE, NULL, 1);    

    std::vector<uint8_t*> buffer(ctx_per_thread);
    for (int i = 0; i < ctx_per_thread; i++) {
      buffer[i] = buffer_raw + BATCH_SIZE * PG_SIZE * i;
    }
    unsigned int seed = threadid;
    int c = 0;
    double tot = 0;
    //for (const double coff: {0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,5e-4,2e-4,1e-4,1e-5,1e-6}) {
    for (int range: {1,2,5,10,20,50,100,200,500}) {
    //for (int range: {500}) {
      int tot_req = 0;
      const int target_req = 1e7 / BATCH_SIZE;
      //zipf_distribution<int64_t> gen_int(range, zipf);
      std::mt19937 mt(seed);
      //printf("%lf started\n", zipf);
      double start = elapsed();
      long cur = 0;
      while(tot_req < target_req) {        
        for (int i = 0; i < ctx_per_thread && tot_req < target_req; i++) {
          int cid = threadid * ctx_per_thread + i;
          if(spdk->poll_task(cid)) {
            tot_req++;
            double start = elapsed();
            c++;
            //if (c % 1000 == 0) printf("!!!%d\n", threadid);
            std::vector<std::pair<int, void*>> req;;
            for (int j = 0; j < BATCH_SIZE; j++) {
              //long r = gen_int(mt) - 1;
              long r = (cur++) % range;
              if (!(0 <= r && r < range)) throw;
              req.emplace_back(r, buffer[i] + j * PG_SIZE);
            }
            spdk->submit_task(req, threadid, cid);
            double end = elapsed();
            tot += end - start;
          }
        }
        if (elapsed() - start > 10) break;
      }
      double end = elapsed();
      //printf("[REPORT] zipf %lf time %lf bw %lf\n", zipf, end - start, (double) tot_req * BATCH_SIZE * 4 / (end - start) / 1024 / 1024);
      printf("[REPORT] range %d time %lf bw %lf\n", range, end - start, (double) tot_req * BATCH_SIZE * 4 / (end - start) / 1024 / 1024);
      for (int i = 0; i < ctx_per_thread; i++) {
        int cid = threadid * ctx_per_thread + i;
        while(!spdk->poll_task(cid));
      }
      spdk->log_latency({0.5, 0.9, 0.99});
      spdk->clear_stats();
    }
  };
  std::vector<std::thread> th;
  for (int i = 0; i < thread_cnt; i++) {
    th.emplace_back(worker, i);
  }
  
  for (int i = 0; i < thread_cnt; i++) {
    th[i].join();
  }
  fflush(stdout);
  
}

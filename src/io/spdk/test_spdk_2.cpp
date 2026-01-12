
#include <vector>
#include <string>
#include <cstdlib>
#include <thread>
#include <sys/time.h>

#include "spdk/env.h"
#ifdef PAGE_SIZE
#undef PAGE_SIZE
#endif
#include "../interface.hpp"

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
    "0000:8c:00.0",
    "0000:8d:00.0",
    "0000:8e:00.0",
    //"0000:22:00.0",
    //"0000:23:00.0"
//    "0000:e1:00.0",
//    "0000:37:00.0",
  };
  const int thread_cnt = 2;
  const int ctx_per_thread = 32;
  const int range = 10 * 1024 * 1024;//20 * 1024 * 1024;
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
    double last = elapsed();
    double tot = 0;
    printf("%d!!!\n", threadid);
    int tot_req = 0;
    while(true) {

      for (int i = 0; i < ctx_per_thread; i++) {
        int cid = threadid * ctx_per_thread + i;
        if(spdk->poll_task(cid)) {
          tot_req++;
          double start = elapsed();
          c++;
          //if (c % 1000 == 0) printf("!!!%d\n", threadid);
          std::vector<std::pair<int, void*>> req;;
          for (int j = 0; j < BATCH_SIZE; j++) {
            long r = rand_r(&seed) % range;
            req.emplace_back(r, buffer[i] + j * PG_SIZE);
          }
          spdk->submit_task(req, threadid, cid);
          double end = elapsed();
          tot += end - start;
        }
      }
      double t = elapsed();
      if (t - last > 1) {
        last = t;
        printf("%d %lf\n", threadid, tot);
        tot = 0;
      }

/*
      std::vector<std::pair<int, void*>> req;;
      for (int j = 0; j < BATCH_SIZE; j++) {
        long r = rand_r(&seed) % range + base;
        req.emplace_back(r, buffer_raw + j * PG_SIZE);
      }
      spdk->push_queue(req, threadid, threadid);
      while(!spdk->check_ready(threadid));
*/
    }
  };
  std::vector<std::thread> th;
  for (int i = 0; i < thread_cnt; i++) {
    th.emplace_back(worker, i);
  }
  
  for (int i = 0; i < thread_cnt; i++) {
    th[i].join();
  }
  
}

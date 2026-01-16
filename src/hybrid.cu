#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <string>
#include <map>

#include <sys/time.h>

//#include "ssd_io.hpp"
#include "hybrid.hpp"

#include "common.hpp"
#include "common_cuda.cuh"

#include "nav_graph.hpp"
//#include "ssd_search_kernel.hpp"
#include "hyd_task_runner.cuh"

#include "io/interface.hpp"

#define REPORT(fmt, ...) printf("[REPORT] " fmt "\n", __VA_ARGS__)

namespace gustann {

  HybridExecutor::HybridExecutor(const Layout &layout,
                                 const DataType &data_type,
                                 const std::string &fpath,
                                 const HybridExecutorConfig &config)
    : layout_(layout), data_type_(data_type) {
    mini_batch_ = config.mini_batch;
    thread_cnt_ = config.thread_cnt;
    ctx_per_thread_ = config.ctx_per_thread;
    
    FILE *input = fopen(fpath.c_str(), "rb");
    if (!input) {
      ERROR("Failed to open index file: {}", fpath);
      exit(-1);
    }
    

    starter_ = new uint8_t[PAGE_SIZE];
    fseek(input, (long) PAGE_SIZE * (layout_.enter_point / layout_.nodes_per_page + 1), SEEK_SET);
    fread((char *)starter_, sizeof(char), PAGE_SIZE, input);
    

    fclose(input);


    if (config.use_backend == HybridExecutorConfig::SPDK) {
#ifdef USE_SPDK
      const auto& ssds = config.ssd_lists;
      if (ssds.empty()) {
        ERROR("NO SSD IN USE!");
        exit(-1);
      }
      loader_ = create_spdk_loader(ssds, mini_batch_ * ctx_per_thread_,
                                   thread_cnt_, thread_cnt_ * ctx_per_thread_);
#else
      ERROR("SPDK Not Supported! Please recompile with `cmake "
            "-DGUSTANN_USE_SPDK=ON`");
      exit(-1);
#endif
    } else if (config.use_backend == HybridExecutorConfig::MEMORY) {
      loader_ = create_mem_loader_sync(fpath.c_str(), layout_.num_pages);
    } else {
      ERROR("Wrong IO Backend setting!");
      exit(-1);
    }

  }

  void HybridExecutor::search(const float *qdata, int num_queries, int topk,
                             int ef_search, int *nns, float *distances, int *found_cnt,
                              PQSearch *pq_, NavGraph *nav_
                             ) {
    int batch_cnt = mini_batch_ * thread_cnt_ * ctx_per_thread_;
    //num_queries = 1;
    CHECK_CUDA(cudaHostRegister((void*)qdata, sizeof(float) * num_queries * layout_.num_dims, cudaHostRegisterDefault));

    if (pq_) {
      pq_->init_device(layout_.num_dims, layout_.num_data, batch_cnt,
                       ef_search);
    } else {
      ERROR("PQ is not inited!");
      throw;
    }
    
    std::atomic<int> tot_reads(0);

    int* start_pts = new int [num_queries];
    memset(start_pts, -1, sizeof(int) * num_queries);

    std::atomic<int> cur(0);
    //cur = 56521;
    auto worker = [&](int threadid) {
      bind_core(threadid * 2 + 21);

      bool finished = false;
      int tot_task = 0;
      std::vector<TaskRunner> tasks;
      for (int i = 0; i < ctx_per_thread_; i++) {
        tasks.emplace_back(threadid, threadid * ctx_per_thread_ + i,
                           mini_batch_, (int)layout_.num_dims, topk,
                           layout_.num_data, (int)layout_.max_m0, ef_search,
                           (int)layout_.enter_point, starter_, pq_,
                           (int)layout_.nodes_per_page, (int)layout_.node_size,
                           (int)layout_.data_size, data_type_, loader_, nav_);       
      }
      double t0 = elapsed();
      INFO("Thread {} started", threadid);

      while(!finished) {
        finished = true;
        for (auto& task: tasks) {
          if (task.update_state()) {
            if (cur.load() < num_queries) {
              int qstart = cur.fetch_add(mini_batch_);             
              int qend = std::min(qstart + mini_batch_, num_queries);
              int qcnt = qend - qstart;
              //printf("!!! %d %d\n", qstart, qcnt);
              if (qstart < num_queries) {
#ifdef CACHE_START
                for (int i = qstart; i < qend; i++) {
                  if (start_pts[i % 10000] != -1) {
                    start_pts[i] = start_pts[i % 10000];
                  }
                }
#endif
                task.init_query(qdata + (int64_t) qstart * layout_.num_dims, qcnt,
                                nns + qstart * topk,
                                distances + qstart * topk,
                                found_cnt + qstart,
                                start_pts + qstart);
                finished = false;
              }
              tot_task += qcnt;
            }
          } else {
            finished = false;
          }
        }
      }

      double tot_gpu = 0;
      double tot_ssd = 0;

      double tot_init_issue = 0;
      double tot_gpu_issue = 0;
      double tot_ssd_issue = 0;
      double tot_fin_issue = 0;

      double tot_lat = 0;
      int cnt_batch = 0;
#ifdef LOAD_PROBE
      std::vector<std::vector<double>> round_probe;
      std::map<std::pair<int, int>, int> tot_freq;
#endif
      for (auto& task: tasks) {
        tot_reads.fetch_add(task.num_reads);
        tot_gpu += task.time_gpu;
        tot_ssd += task.time_ssd;

        tot_init_issue += task.time_init_issue;
        tot_gpu_issue += task.time_gpu_issue;
        tot_ssd_issue += task.time_ssd_issue;
        tot_fin_issue += task.time_fin_issue;
        tot_lat += task.latency;
        cnt_batch += task.cnt_query;
#ifdef LOAD_PROBE
        for (int i = 0; i < (int) task.ssd_overall.size(); i++) {
          if (i == (int) round_probe.size()) {
            round_probe.push_back(std::vector<double>(task.sample_ssd));
          }
          for (int j = 0; j < task.sample_ssd; j++) {
            round_probe[i][j] += task.ssd_overall[i][j];
            //printf("%lf\n", task.ssd_overall[i][j]);
          }
        }
        for (auto x: task.freq) {
          tot_freq[x.first] += x.second;
        }
#endif
      }
      double t1 = elapsed();
      INFO("Thread {}: {} / {} s, {} queries in {} s, {} qps",
           threadid, tot_gpu, tot_ssd, tot_task, t1 - t0, tot_task / (t1 - t0));
      INFO("Thread {}: Init {}, GPU {}, SSD {}, FIN {}",
           threadid, tot_init_issue, tot_gpu_issue, tot_ssd_issue, tot_fin_issue);
      INFO("Thread {}: Latency {} ms",
           threadid, tot_lat / cnt_batch * 1000);
      REPORT("LAT%d %lf", threadid, tot_lat / cnt_batch * 1000);
#ifdef LOAD_PROBE
#if 1
      int tot_r = round_probe.size();
      for (int i = 0; i < tot_r; i++) {
        printf("%lf ", round_probe[i].front() / cnt_batch);
      }
      printf("\n");
      for (int i = 0; i < tot_r; i++) {
        printf("%lf ", round_probe[i].back() / cnt_batch);
      }
      printf("\n");
#endif
#if 1
      std::map<int, std::vector<int>> g;
      for (auto x: tot_freq) {
        g[x.first.first].push_back(x.second);
      }
      int y = 0;
      for (auto &x: g) {
        auto &v = x.second;
        std::sort(v.begin(), v.end());
        std::reverse(v.begin(), v.end());
        printf("%d\t%d\t%lu\n", v[0], v[(v.size() - 1) / 2], v.size());
        if (y++ < 3) {
          const int G = 50;
          for (int i = 0; i < G; i++) {
            printf("%d ", v[i * v.size() / G]);
          }
          
          printf("%d \n", v.back());
          const int X = 500;
          const int Y = 50;
          for (int i = 0; i <= Y; i++) {
            printf("%d ", v[i * v.size() / X]);
          }
          printf("%d\n", v.back());
        }        

      }
#endif
#endif
    };

    std::vector<std::thread> th;
    CHECK_CUDA(cudaDeviceSynchronize());
    double start = elapsed();
    for (int i = 0; i < thread_cnt_; i++) {
      th.emplace_back(worker, i);
    }

    for (int i = 0; i < thread_cnt_; i++) {
      th[i].join();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double end = elapsed();
    DEBUG("End Search");
    INFO("Use time: {}", end - start);
    INFO("Total reads: {}", tot_reads.load());
    REPORT("Time %lf", end - start);
    REPORT("IO %d", tot_reads.load());
    CHECK_CUDA(cudaDeviceSynchronize());

  }
}

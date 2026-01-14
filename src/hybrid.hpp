#pragma once
#include <string>

#include "layout.hpp"
#include "common.hpp"
#include "pq_search.hpp"
#include "nav_graph.hpp"
#include "io/interface.hpp"

namespace gustann {
  struct HybridExecutorConfig {
    int mini_batch;
    int thread_cnt;
    int ctx_per_thread;
    enum {
      SPDK,
      MEMORY,
    } use_backend;    
    std::vector<std::string> ssd_lists;
  };
  class HybridExecutor {
  public:
    HybridExecutor(const Layout &layout, const DataType &data_type, const std::string &fpath, const HybridExecutorConfig& config);
    void search(const float *qdata, const int num_queries, const int topk,
                const int ef_search, int *nns, float *distances, int *found_cnt,
                PQSearch *pq = nullptr, NavGraph *nav = nullptr);
  private:
    Layout layout_;
    DataType data_type_;

    uint8_t* starter_;

    std::shared_ptr<IndexLoader> loader_;

    int mini_batch_;
    int thread_cnt_;
    int ctx_per_thread_;

  };
}

/* bam.hpp
 * Legacy I/O backend and scheduler
 */
#pragma once

#include <string>

#include "bam_config.hpp"
#include "page_cache.h"
#include "page_manager.hpp"

#include "layout.hpp"
#include "pq_search.hpp"
#include "common.hpp"
#include "nav_graph.hpp"

namespace gustann {
  struct GustANNStats;

  class BaMExecutor {
  public:
    BaMExecutor(const std::string &fpath, const Layout& layout, const DataType& data_type, const BaMConfig& config, bool copy_data);
    void search(const float *qdata, const int num_queries_, const int topk,
                const int ef_search, int *nns, float *distances, int *found_cnt,
                PQSearch* pq, NavGraph* nav, GustANNStats* stats = nullptr);
  private:
    struct BaMContext {
      std::vector<Controller *> ctrls;
      page_cache_t *h_pc = nullptr;
      range_t<uint8_t> *h_range = nullptr;
      std::vector<range_t<uint8_t> *> vr;
      array_t<uint8_t> *a = nullptr;
      range_d_t<uint8_t> *d_range = nullptr;
    } bam_data_;
    
    bool use_simple_cache = false;
    
    int block_cnt_, block_dim_;
    int visited_table_size_, visited_list_size_;

    int64_t page_size_;
    Layout layout_;
    DataType data_type_;

    uint8_t *mem_data_;

    void copy_to_bam(const std::string& fpath);
    void read_to_mem(const std::string &fpath);
    void init(const BaMConfig &config);
  };

  
}

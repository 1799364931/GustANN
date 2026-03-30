/* pure_mem.hpp
 * Optimized pure in-memory implementation
 */
#pragma once

#include <string>

#include "layout.hpp"
#include "pq_search.hpp"
#include "common.hpp"
#include "nav_graph.hpp"

namespace gustann {
  
  class PureMemExecutor {
  public:
    PureMemExecutor(const std::string &fpath, const Layout& layout, const DataType& data_type, bool use_gpu_mem);
    void search(const float *qdata, const int num_queries_, const int topk,
                const int ef_search, int *nns, float *distances, int *found_cnt,
                PQSearch *pq, NavGraph *nav);
    ~PureMemExecutor();
    
  private:
    static constexpr int64_t PAGE_SIZE = 4096;
    int block_cnt_, block_dim_;    

    Layout layout_;
    DataType data_type_;

    uint8_t *mem_data_; // = (use_gpu_mem ? mem_data_host_ : mem_data_dev_);
    uint8_t *mem_data_host_;
    uint8_t *mem_data_dev_ = nullptr;

    bool use_gpu_mem_; // true: GPU Mem; false: DRAM

    void read_to_mem(const std::string &fpath);

  };

  
}

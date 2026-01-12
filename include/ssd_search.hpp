#pragma once
#ifdef _USE_BAM
#include "page_cache.h"
#include "page_manager.hpp"
#endif
#include "pq_search.hpp"
#include "common.hpp"

namespace gustann {

class GustANN {
  struct BaMContext {
#ifdef _USE_BAM
    std::vector<Controller *> ctrls;
    page_cache_t *h_pc = nullptr;
    range_t<uint8_t> *h_range = nullptr;
    std::vector<range_t<uint8_t> *> vr;
    array_t<uint8_t> *a = nullptr;
    range_d_t<uint8_t> *d_range = nullptr;
#endif
  } bam_data_;

  bool use_simple_cache = false;

  int block_cnt_, block_dim_;
  int visited_table_size_, visited_list_size_;

  int64_t num_data_;
  int num_dims_, batch_size_;
  int max_m0_;
  int64_t enter_point_;
  int64_t num_pages_;

  uint64_t nodes_per_page_;
  uint64_t node_size_;
  uint64_t data_size_;
  uint64_t page_size_;
  uint8_t* mem_data_;

  DataType data_type_;

  enum SearchType {
    BAM,
    HYBRID
  } search_type = BAM;

  size_t get_data_size() const {
    switch (data_type_) {
    case FLOAT: return sizeof(float);
    case UINT8: return sizeof(uint8_t);      
    }
  }  

  void init_bam_(const BaMConfig &config, int64_t ssd_pages);
  void parse_diskann_metadata(const std::string& fpath);
  void copy_to_bam(const std::string& fpath);

  void read_to_mem(const std::string& fpath);

public:
  GustANN(DataType = FLOAT);

  void init(const BaMConfig &config, const std::string& fpath, bool copy);
  void init_hybrid(const std::string& fpath);

  struct Config {
    std::string nav_data = "";
    int warmup_batch = 0;
    std::vector<std::string> ssd_list;
  };
  void search(const float *qdata, const int num_queries, const int topk,
              const int ef_search, int *nns, float *distances, int *found_cnt, const Config& config, PQSearch* pq = nullptr);
  
  void search_hybrid(const float *qdata, const int num_queries, const int topk,
                     const int ef_search, int *nns, float *distances, int *found_cnt, int, int, int, const Config& config, PQSearch* pq = nullptr);
};
} // namespace gustann

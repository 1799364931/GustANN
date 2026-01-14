#pragma once
#ifdef _USE_BAM
#include "bam.hpp"
#endif
#include "pq_search.hpp"
#include "common.hpp"
#include "layout.hpp"
#include "hybrid.hpp"

namespace gustann {

  class GustANNConfig {
    std::string index_prefix;
    std::string pq_file_prefix;
    bool use_nav_graph;
    std::string nav_graph_prefix;
  };
  
  class GustANN {
    Layout layout_;
    
    uint64_t page_size_;
    uint8_t* mem_data_;
    
    DataType data_type_;

    union {
#ifdef _USE_BAM
      BaMExecutor* bam;
#endif
      HybridExecutor* hybrid;
    } executor_;

    enum SearchType {
      UNINITED,
      BAM,
      HYBRID
    } search_type = UNINITED;

    size_t get_data_size() const {
      switch (data_type_) {
      case FLOAT: return sizeof(float);
      case UINT8: return sizeof(uint8_t);      
      }
    }
    void parse_diskann_metadata(const std::string &fpath);

    void init_gustann_internal();
  public:
    GustANN(DataType = FLOAT);
#ifdef _USE_BAM
    void init_bam(const BaMConfig &config, const std::string &fpath, bool copy);
#endif
    void init_hybrid(const std::string& fpath, const HybridExecutorConfig& config);

    void search(const float *qdata, const int num_queries, const int topk,
                const int ef_search, int *nns, float *distances, int *found_cnt, const Config& config, PQSearch* pq = nullptr);  
  };
} // namespace gustann

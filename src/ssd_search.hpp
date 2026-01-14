#pragma once
#ifdef USE_BAM
#include "bam.hpp"
#endif
#include "pq_search.hpp"
#include "common.hpp"
#include "layout.hpp"
#include "hybrid.hpp"

namespace gustann {

  class GustANNConfig {
  public:
    std::string index_file;
    std::string pq_file_prefix;
    std::string nav_graph_prefix;
  };
  
  class GustANN {
    Layout layout_;
    
    uint64_t page_size_;
    uint8_t* mem_data_;
    
    DataType data_type_;

    union {
#ifdef USE_BAM
      BaMExecutor* bam;
#endif
      HybridExecutor* hybrid;
    } executor_;

    PQSearch *pq_ = nullptr;
    NavGraph *nav_ = nullptr;

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

    void init_gustann_internal(const GustANNConfig& config);
  public:
    GustANN(DataType = FLOAT);
#ifdef USE_BAM
    void init_bam(const GustANNConfig& gustann_config, const BaMConfig& bam_config, bool copy);
#endif
    void init_hybrid(const GustANNConfig& gustann_config, const HybridExecutorConfig& hybrid_config);

    void search(const float *qdata, const int num_queries, const int topk,
                const int ef_search, int *nns, float *distances, int *found_cnt);  
  };
} // namespace gustann

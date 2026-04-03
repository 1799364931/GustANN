#pragma once
#ifdef USE_BAM
#include "bam.hpp"
#endif
#include "pq_search.hpp"
#include "common.hpp"
#include "layout.hpp"
#include "hybrid.hpp"
#include "pure_mem.hpp"

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
    
    DataType data_type_;

    union {
#ifdef USE_BAM
      BaMExecutor* bam;
#endif
      HybridExecutor *hybrid;
      PureMemExecutor* pure_mem;      
    } executor_;
    
    PQSearch *pq_ = nullptr;
    NavGraph *nav_ = nullptr;

    enum SearchType {
      UNINITED,
      BAM,
      HYBRID,
      PURE_MEM,
    } search_type = UNINITED;

    size_t get_data_size() const {
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
      switch (data_type_) {
      case FLOAT: return sizeof(float);
      case UINT8: return sizeof(uint8_t);
      }
#pragma GCC diagnostics pop
    }

    void init_gustann_internal(const GustANNConfig& config);
  public:
    GustANN(DataType = FLOAT);
#ifdef USE_BAM
    void init_bam(const GustANNConfig& gustann_config, const BaMConfig& bam_config, bool copy);
#endif
    void init_hybrid(const GustANNConfig &gustann_config,
                     const HybridExecutorConfig &hybrid_config);
    void init_pure_mem(const GustANNConfig &gustann_config, bool use_gpu_mem);

    void search(const float *qdata, const int num_queries, const int topk,
                const int ef_search, int *nns, float *distances,
                int *found_cnt);

    ~GustANN() {
      if (pq_)
        delete pq_;
      if (nav_)
        delete nav_;
#ifdef USE_BAM
      if (search_type == BAM) {
        if (executor_.bam) delete executor_.bam;
      }
#endif
      if (search_type == HYBRID) {
        if (executor_.hybrid) delete executor_.hybrid;
      }
    }
  };
} // namespace gustann

#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

#include "bam_config.hpp"
#include "hybrid_config.hpp"
#include "common.hpp"
#include "layout.hpp"

namespace gustann {
  class BaMExecutor;
  class HybridExecutor;
  class PureMemExecutor;
  class PQSearch;
  struct NavGraph;

  class GustANNConfig {
  public:
    std::string index_file;
    std::string pq_file_prefix;
    std::string nav_graph_prefix;
  };

  struct GustANNStats {
    double run_time;
  };
  
  class GustANN {
    Layout layout_;  
    
    DataType data_type_;

    union {
#ifdef USE_BAM
      BaMExecutor* bam;
#endif
      HybridExecutor *hybrid;
      PureMemExecutor* pure_mem;      
    } search_executor_;
    
    enum SearchType {
      SEARCH_UNINITED,
      BAM,
      HYBRID,
      PURE_MEM,
    } search_type = SEARCH_UNINITED;

    enum BuildType {
      BUILD_UNINITED,
    } build_type = BUILD_UNINITED;

    PQSearch *pq_ = nullptr;
    NavGraph *nav_ = nullptr;

    size_t get_data_size() const {
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
      switch (data_type_) {
      case FLOAT: return sizeof(float);
      case UINT8: return sizeof(uint8_t);
      }
#pragma GCC diagnostic pop
      ASSERT(false);
      return 0;
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
                int *found_cnt, GustANNStats* stats = nullptr);

    ~GustANN();
  };
} // namespace gustann

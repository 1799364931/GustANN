#pragma once
#ifdef _USE_BAM
#include "bam.hpp"
#endif
#include "pq_search.hpp"
#include "common.hpp"
#include "layout.hpp"

namespace gustann {

class GustANN {
  Layout layout_;
  
  uint64_t page_size_;
  uint8_t* mem_data_;

  DataType data_type_;

#ifdef _USE_BAM
  BaMExecutor* bam_ = nullptr;
#endif

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
  void parse_diskann_metadata(const std::string& fpath);
public:
  GustANN(DataType = FLOAT);
#ifdef _USE_BAM
  void init(const BaMConfig &config, const std::string &fpath, bool copy);
#endif
  void init_hybrid(const std::string& fpath);

  void search(const float *qdata, const int num_queries, const int topk,
              const int ef_search, int *nns, float *distances, int *found_cnt, const Config& config, PQSearch* pq = nullptr);
  
  void search_hybrid(const float *qdata, const int num_queries, const int topk,
                     const int ef_search, int *nns, float *distances, int *found_cnt, int, int, int, const Config& config, PQSearch* pq = nullptr);
};
} // namespace gustann

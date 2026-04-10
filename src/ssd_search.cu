#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "common.hpp"
#include "common_cuda.cuh"
#ifdef USE_BAM
#include "bam.hpp"
#endif
#include "hybrid.hpp"
#include "pure_mem.hpp"
#include "pq_search.hpp"
#include "ssd_search.hpp"

#include "nav_graph.hpp"

void init_opt();

namespace gustann {

  GustANN::GustANN(DataType data_type) {
    data_type_ = data_type;
  }

  GustANN::~GustANN() {
    if (pq_) {
      delete pq_;
    }
    if (nav_) {
      delete nav_;
    }
#ifdef USE_BAM
    if (search_type == BAM) {
      if (search_executor_.bam) {
        delete search_executor_.bam;
      }
    }
#endif
    if (search_type == HYBRID) {
      if (search_executor_.hybrid) {
        delete search_executor_.hybrid;
      }
    }

    if (search_type == PURE_MEM) {
      if (search_executor_.pure_mem) {
        delete search_executor_.pure_mem;
      }
    }
  }

  void GustANN::init_gustann_internal(const GustANNConfig &config) {
    layout_.parse_diskann_metadata(config.index_file, get_data_size());
    // Initialize PQ search if pq_file_prefix is provided
    INFO("PQ {}, NAV {}", config.pq_file_prefix, config.nav_graph_prefix);
    if (!config.pq_file_prefix.empty()) {
      pq_ = new PQSearch();
      std::string pq_pivots_file = config.pq_file_prefix + "_pivots.bin";
      std::string pq_compressed_file = config.pq_file_prefix + "_compressed.bin";
      pq_->read_data(pq_pivots_file, pq_compressed_file);
    } else {
      ERROR("PQ file not set!");
      throw;
    }
    
    // Initialize navigation graph if use_nav_graph is true
    if (!config.nav_graph_prefix.empty()) {
      nav_ = new NavGraph();
      std::string nav_index_file = config.nav_graph_prefix + "/" + "nav_index";
      std::string nav_data_file = config.nav_graph_prefix + "/" + "nav_index.data";
      std::string nav_map_file = config.nav_graph_prefix + "/" + "map.txt";
      nav_->init(nav_index_file, nav_data_file, nav_map_file, get_data_size());
    } else {
      INFO("Not using Nav Graph!");
    } 
  }
  

#ifdef USE_BAM  
  void GustANN::init_bam(const GustANNConfig& gustann_config, const BaMConfig& bam_config, bool copy) {

    init_gustann_internal(gustann_config);
    DEBUG("{} {}", layout_.node_size * layout_.nodes_per_page, bam_config.page_size);
    ASSERT(layout_.node_size * layout_.nodes_per_page <= bam_config.page_size);

    search_type = BAM;
    search_executor_.bam = new BaMExecutor(gustann_config.index_file, layout_, data_type_, bam_config, copy);
    
    DEBUG("Initialization finished");
  }
#endif

  void GustANN::init_hybrid(const GustANNConfig &gustann_config,
                            const HybridExecutorConfig &hybrid_config) {
    init_gustann_internal(gustann_config);
    search_type = HYBRID;

    search_executor_.hybrid = new HybridExecutor(layout_, data_type_, gustann_config.index_file, hybrid_config);
    

    
    DEBUG("Initialization finished");
  }

  void GustANN::init_pure_mem(const GustANNConfig &gustann_config, bool use_gpu_mem) {
    init_gustann_internal(gustann_config);
    search_type = PURE_MEM;
    search_executor_.pure_mem = new PureMemExecutor(gustann_config.index_file, layout_, data_type_, use_gpu_mem);
  }

  void GustANN::search(const float *qdata, const int num_queries_,
                       const int topk, const int ef_search, int *nns,
                       float *distances, int *found_cnt, GustANNStats *stats) {
      
#ifdef USE_BAM
    if (search_type == BAM) {
      search_executor_.bam->search(qdata, num_queries_, topk, ef_search, nns, distances,
                   found_cnt, pq_, nav_, stats);
      return;
    }
#endif

    if (search_type == HYBRID) {
      search_executor_.hybrid->search(qdata, num_queries_, topk, ef_search, nns,
                               distances, found_cnt, pq_, nav_, stats);
      return;
    }

    if (search_type == PURE_MEM) {
      search_executor_.pure_mem->search(qdata, num_queries_, topk, ef_search, nns,
                                 distances, found_cnt, pq_, nav_, stats);
      return;
    }

    ERROR("UnInited!");
    throw;
  }


}

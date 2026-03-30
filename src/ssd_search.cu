#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "common.hpp"
#include "common_cuda.cuh"
#include "hybrid.hpp"
#include "pure_mem.hpp"
#include "ssd_search.hpp"

#include "nav_graph.hpp"

void init_opt();

namespace gustann {

  GustANN::GustANN(DataType data_type) {
    data_type_ = data_type;
  }

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))
  void GustANN::parse_diskann_metadata(const std::string& fpath) {
    std::ifstream input(fpath, std::ios::binary);
    if (!input.is_open()) {
      ERROR("Failed to open file {}", fpath);
      exit(-1);
    }
    
    INFO("load DiskANN index from {}", fpath);
    
    // reqd meta values
    DEBUG("read meta values");
    
    // from: https://github.com/microsoft/DiskANN/blob/main/src/pq_flash_index.cpp#L1043
    
    uint32_t nr, nc; // metadata itself is stored as bin format (nr is number of
    // metadata, nc should be 1)
    READ_U32(input, nr);
    READ_U32(input, nc);
    
    uint64_t disk_nnodes;
    uint64_t disk_ndims; // can be disk PQ dim if disk_PQ is set to true
    READ_U64(input, disk_nnodes);
    READ_U64(input, disk_ndims);
    
    layout_.num_data = disk_nnodes;
    layout_.num_dims = disk_ndims;
    uint64_t _disk_bytes_per_point = layout_.num_dims * get_data_size();
    
    uint64_t medoid_id_on_file;
    uint64_t _max_node_len, _nnodes_per_sector;
    READ_U64(input, medoid_id_on_file);
    layout_.enter_point = medoid_id_on_file;
    
    READ_U64(input, _max_node_len);
    READ_U64(input, _nnodes_per_sector);
    layout_.max_m0 = ((_max_node_len - _disk_bytes_per_point) / sizeof(uint32_t)) - 1;

    // NOTE: These three parameters are unused. Align with PipeANN!
    size_t _num_frozen_points, _reorder_data_exists;
    READ_U64(input, _num_frozen_points);
    uint64_t file_frozen_id;
    READ_U64(input, file_frozen_id);
    
    READ_U64(input, _reorder_data_exists);
    
    INFO("meta values loaded, num_data: {}, num_dims: {}, max_m0: {}, enter_point: {}",
         layout_.num_data, layout_.num_dims, layout_.max_m0, layout_.enter_point);

    layout_.nodes_per_page = _nnodes_per_sector;
    layout_.num_pages = (layout_.num_data + layout_.nodes_per_page - 1) / layout_.nodes_per_page;
    layout_.node_size = _max_node_len;
    layout_.data_size = _disk_bytes_per_point;

    INFO("node size: {}, data size: {}, nodes_per_page: {}, tot_pages: {}",
         layout_.node_size,
         layout_.data_size,
         layout_.nodes_per_page,
         layout_.num_pages);
  }

  void GustANN::init_gustann_internal(const GustANNConfig &config) {
    parse_diskann_metadata(config.index_file);
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
    executor_.bam = new BaMExecutor(gustann_config.index_file, layout_, data_type_, bam_config, copy);

    

    //page_size_ = config.page_size;
    DEBUG("Initialization finished");
  }
#endif

  void GustANN::init_hybrid(const GustANNConfig &gustann_config,
                            const HybridExecutorConfig &hybrid_config) {
    init_gustann_internal(gustann_config);
    search_type = HYBRID;

    executor_.hybrid = new HybridExecutor(layout_, data_type_, gustann_config.index_file, hybrid_config);
    

    
    DEBUG("Initialization finished");
  }

  void GustANN::init_pure_mem(const GustANNConfig &gustann_config, bool use_gpu_mem) {
    init_gustann_internal(gustann_config);
    search_type = PURE_MEM;
    executor_.pure_mem = new PureMemExecutor(gustann_config.index_file, layout_, data_type_, use_gpu_mem);
  }

  void GustANN::search(const float *qdata, const int num_queries_,
                       const int topk, const int ef_search, int *nns,
                       float *distances, int *found_cnt) {
      
#ifdef USE_BAM
    if (search_type == BAM) {
      executor_.bam->search(qdata, num_queries_, topk, ef_search, nns, distances,
                   found_cnt, pq_, nav_);
      return;
    }
#endif

    if (search_type == HYBRID) {
      executor_.hybrid->search(qdata, num_queries_, topk, ef_search, nns,
                               distances, found_cnt, pq_, nav_);
      return;
    }

    if (search_type == PURE_MEM) {
      executor_.pure_mem->search(qdata, num_queries_, topk, ef_search, nns,
                                 distances, found_cnt, pq_, nav_);
      return;
    }

    ERROR("UnInited!");
    throw;
  }


}

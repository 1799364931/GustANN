#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "common.hpp"
#include "common_cuda.cuh"
#include "hybrid.hpp"
#include "ssd_search.hpp"
#include "ssd_search_kernel.hpp"

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
    
    // setting up concept of frozen points in disk index for streaming-DiskANN
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
#ifdef _USE_BAM  
  void GustANN::init_bam(const BaMConfig& config, const std::string& fpath, bool copy) {
    parse_diskann_metadata(fpath);
    DEBUG("{} {}", layout_.node_size * layout_.nodes_per_page, config.page_size);
    ASSERT(layout_.node_size * layout_.nodes_per_page <= config.page_size);

    search_type = BAM;
    executor_.bam = new BaMExecutor(fpath, layout_, data_type_, config, copy);

    //page_size_ = config.page_size;
    DEBUG("Initialization finished");
  }
#endif

// TODO: walkaround!!!
#ifndef _USE_BAM
  void GustANN::init_hybrid(const std::string& fpath, const HybridExecutorConfig& config) {

    parse_diskann_metadata(fpath);
    
    search_type = HYBRID;
    executor_.hybrid = new HybridExecutor(layout_, data_type_, fpath, config);
    DEBUG("Initialization finished");
  }
#endif
  void GustANN::search(const float *qdata, const int num_queries_,
                       const int topk, const int ef_search, int *nns,
                       float *distances, int *found_cnt, const Config &config,
                       PQSearch *pq) {
      
#ifdef _USE_BAM
    if (search_type == BAM) {
      executor_.bam->search(qdata, num_queries_, topk, ef_search, nns, distances,
                   found_cnt, config, pq);
      return;
    }
#endif
// TODO: walkaround!
#ifndef _USE_BAM
    if (search_type == HYBRID) {
      executor_.hybrid->search(qdata, num_queries_, topk, ef_search, nns,
                               distances, found_cnt, config, pq);
      return;
    }
#endif
    ERROR("UnInited!");
    throw;
  }


}

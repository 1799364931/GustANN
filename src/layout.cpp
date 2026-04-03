#include <fstream>

#include "common.hpp"
#include "layout.hpp"

namespace gustann {

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))

void Layout::parse_diskann_metadata(const std::string& fpath, uint64_t data_size_per_dim) {
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

  num_data = disk_nnodes;
  num_dims = disk_ndims;
  uint64_t disk_bytes_per_point = num_dims * data_size_per_dim;

  uint64_t medoid_id_on_file;
  uint64_t max_node_len, nnodes_per_sector;
  READ_U64(input, medoid_id_on_file);
  enter_point = medoid_id_on_file;

  READ_U64(input, max_node_len);
  READ_U64(input, nnodes_per_sector);
  max_m0 = ((max_node_len - disk_bytes_per_point) / sizeof(uint32_t)) - 1;

  // NOTE: These three parameters are unused. Align with PipeANN!
  size_t num_frozen_points, reorder_data_exists;
  READ_U64(input, num_frozen_points);
  uint64_t file_frozen_id;
  READ_U64(input, file_frozen_id);

  READ_U64(input, reorder_data_exists);

  INFO("meta values loaded, num_data: {}, num_dims: {}, max_m0: {}, enter_point: {}",
       num_data, num_dims, max_m0, enter_point);

  nodes_per_page = nnodes_per_sector;
  num_pages = (num_data + nodes_per_page - 1) / nodes_per_page;
  node_size = max_node_len;
  data_size = disk_bytes_per_point;

  INFO("node size: {}, data size: {}, nodes_per_page: {}, tot_pages: {}",
       node_size,
       data_size,
       nodes_per_page,
       num_pages);
}

} // namespace gustann

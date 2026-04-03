#pragma once

#include <cstdint>
#include <string>

namespace gustann {

// TODO: enhance it by adding custom node_id -> (blk_id, offset) functions.
struct Layout {
  int64_t num_dims; // # of vector dimension
  int64_t max_m0; // # of neighbors
  int64_t enter_point; // default graph entry node (when not using Nav. graph)

  int64_t data_size; // total size of the vector, = num_dims * sizeof(data_type)
  int64_t node_size; // total size of a node, = data_size + (max_m0 + 1) * 4

  int64_t num_pages; // # of Pages, should be ceil(num_nodes / node_size)
  int64_t num_data;  // # of vectors/nodes

  int64_t nodes_per_page;

  enum {
    PAGE_ALIGNED,
  } align_stype = PAGE_ALIGNED; // TODO.

  void parse_diskann_metadata(const std::string& fpath, uint64_t data_size_per_dim);
};

} // namespace gustann

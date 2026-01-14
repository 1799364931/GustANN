#pragma once

#include "pq.cuh"

namespace gustann {
  struct __align__(128) Data {
    int visited_cnt;
    int size;
  };

#ifdef FLOAT_DATA
  using data_type = float;
#else
  using data_type = uint8_t;
#endif

  
  __global__ void init_search(float* qdata, PQSearchData* pq_data, int stream_offset, int dim) {
    pq_data->init_query(qdata + blockIdx.x * dim, stream_offset);
  }
}

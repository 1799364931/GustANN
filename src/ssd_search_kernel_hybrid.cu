//#include "cuda_utils_kernels.cuh"
#include "cuda_base_kernels.cuh"
#include "ssd_search_kernel.hpp"

#include "impl/pq.cuh"
//#include "impl/calc.cuh"

#include "impl/util.cuh"

#include "impl/opt_hyd.cuh"
#include "impl/opt_hyd_v2.cuh"
#include "impl/nav.cuh"

namespace gustann {

  __global__ void init_search(float* qdata, PQSearchData* pq_data, int stream_offset, int dim) {
    pq_data->init_query(qdata + blockIdx.x * dim, stream_offset);
  }


  __global__ void get_pq_dist_kernel(uint8_t *buffer, int32_t *request,
                                     float *tmp_dist, int *tmp_id,
                                     PQSearchData* pq_data,
                                     int nodes_per_page, int node_len, int data_len,
                                     int pq_offset, int max_m0) {
    int buffer_id = blockIdx.x;
    int nodeid_u = request[buffer_id];
    if (nodeid_u == -1) return;
    //if (threadIdx.x == 0) printf("!!!!%d\n", nodeid_u);
    int offset = nodeid_u % nodes_per_page * node_len + data_len;
    int* buffer_u = (int*)(buffer + buffer_id * 4096 + offset);
    
    //printf("??? %d %d\n", threadIdx.x, buffer_u[0]);
    if (threadIdx.x >= buffer_u[0]) {
      if (threadIdx.x < max_m0) {
        tmp_id[blockIdx.x * max_m0 + threadIdx.x] = -1;
        tmp_dist[blockIdx.x * max_m0 + threadIdx.x] = INFINITY;
      }
      return;
    }
    int nodeid_v = tmp_id[blockIdx.x * max_m0 + threadIdx.x] = buffer_u[threadIdx.x + 1];    
    float* dist_vec = pq_data->pq_dists + (pq_offset + buffer_id) * pq_data->num_pivots * pq_data->num_chunks;
    float dist = 0;
    uint8_t* data = pq_data->compressed_data + (long) nodeid_v * pq_data->num_chunks;
    //printf("%d\n", nodeid_v);
    for (int j = 0; j < pq_data->num_chunks; j++) {
      dist += dist_vec[j * pq_data->num_pivots + data[j]];
    }
    tmp_dist[blockIdx.x * max_m0 + threadIdx.x] = dist;
    //printf("! %d %d %lf\n", blockIdx.x, nodeid_v, dist);
  }

  } // namespace gustann



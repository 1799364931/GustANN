#pragma once

#include <cstdint>

#include "pq.cuh"
#include "util.cuh"


namespace gustann {
  template <class T>
  struct GraphData {
    int deg;
    int* edge;
    T* data;
    __inline__ __device__
    int operator [] (int x) const { return edge[x]; }
  };

  struct PageData {
    int nodes_per_page;
    int node_len;
    int data_len;
    uint8_t* disk;
    static const int page_size = 4096;
    __inline__ __device__ void init() {}
    __inline__ __device__ int get_block(int nodeid) {
      return nodeid / nodes_per_page;
    }
    __inline__ __device__ int get_offset(int nodeid) {
      return nodeid % nodes_per_page * node_len;
    }
    __inline__ __device__ uint8_t* read_node(int nodeid) {
      int block = get_block(nodeid);
      int64_t start_idx = (int64_t) block * page_size;
      return (uint8_t*) (disk + start_idx);
    }
    template<class T>
    __inline__ __device__ GraphData<T> get_graph(int nodeid) {
      int offset = get_offset(nodeid);
      //if (threadIdx.x == 0) printf("O %d\n", offset);
      uint8_t *tmp = (read_node(nodeid) + offset);
      //printf("%lx %lx %lx!!!\n", disk, get_block(nodeid), tmp);
      T *data = (T *)tmp;
      int* result = (int*)(tmp + data_len);
      return (GraphData<T>) { result[0], result + 1, data };
    }
  };

  static const int WARP_SIZE = 32;
  template <class T>
  __global__ void __launch_bounds__(128, 14)    
    search_disk_graph_kernel
    (uint8_t* data, float* qdata, const int num_dims,
     PQSearchData* pq_data,
     int nodes_per_page, int node_len, int data_len,
     int* entries,
     const int max_m, const int ef_search, const int topk,
     //uint32_t* neighbor_id, float* neighbor_dist,
     int* nns, float* distances, int* found_cnt,
     int qcnt) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    static __shared__ int ctx_size;
    static __shared__ int ctx_u;
    PageData pager;
    pager.nodes_per_page = nodes_per_page;
    pager.node_len = node_len;
    pager.disk = data;
    pager.data_len = data_len;
    pager.init();

    extern __shared__ uint8_t shm_pool[];
    int* mv_pos = (int*) shm_pool;
    uint32_t* mv_id = (uint32_t*)(mv_pos + ef_search + max_m);
    float* mv_dist = (float*)(mv_id + ef_search + max_m);
    uint32_t* tmp_id = (uint32_t*)(mv_dist + ef_search + max_m);
    float* tmp_dist = (float*)(tmp_id + ef_search + max_m);
    
    GraphData<T> graph;
    for (int qid = blockIdx.x; qid < qcnt; qid += gridDim.x) {
      __syncthreads();
      if (tid == 0) {
        ctx_size = 0;
      }
      __syncthreads();
      float* src_vec = qdata + qid * num_dims;
      pq_data->init_query(src_vec);
      int u = entries[qid];
      
      while(u != -1) {   
        //printf("%d!!\n", u);
        graph = pager.get_graph<T>(u);
        //printf("%d %d!!\n", graph.deg, graph[0]);
        int sz = ctx_size;
        const int& deg = graph.deg;
        if (tid >= graph.deg) {
          if (tid < max_m) {
            tmp_id[sz + tid] = 0xffffffff - tid;
            tmp_dist[sz + tid] = INFINITY;           
          }
        } else {
          int nodeid_v = tmp_id[sz + tid] = graph[threadIdx.x];
          float* dist_vec = pq_data->pq_dists + bid * PQSearchData::num_pivots * pq_data->num_chunks;
          float dist = 0;
          uint8_t* data = pq_data->compressed_data + (long) nodeid_v * pq_data->num_chunks;
#pragma unroll 32
          for (int j = 0; j < pq_data->num_chunks; j++) {
            //int x = tmp_id[j + tid] & 0xff;
            //printf("%d %d\n", j, data[j]);
            dist += dist_vec[j * PQSearchData::num_pivots + data[j]];
          }
          tmp_dist[sz + tid] = dist;
        }

        __syncthreads();
        
        for (int len = 2; len < 2 * deg; len *= 2) {
          int array_id = tid / len;
          int start = array_id * len;
          int mid = min(start + len / 2, deg);
          int end = min(start + len, deg);
          // TODO: OPT
          if (tid >= start && tid < mid) {
            int id = lower_bound(tmp_dist + sz + mid,
                                 tmp_id + sz + mid,
                                 0, end - mid,
                                 tmp_dist[sz + tid],
                                 tmp_id[sz + tid]);
            mv_pos[tid] = id + tid;
          }
          if (tid >= mid && tid < end) {
            int id = lower_bound(tmp_dist + sz,
                                 tmp_id + sz,
                                 start, mid,
                                 tmp_dist[sz + tid],
                                 tmp_id[sz + tid]);
            mv_pos[tid] = id + tid - mid;
          }
          __syncthreads();
          __threadfence_block();

          if (tid < deg) {
            //assert(0 <= mv_pos[tid] && mv_pos[tid] < max_m);
            mv_id[mv_pos[tid]] = tmp_id[sz + tid];
            mv_dist[mv_pos[tid]] = tmp_dist[sz + tid];
          }
          __syncthreads();
          if (tid < deg) {
            tmp_id[sz + tid] = mv_id[tid];
            tmp_dist[sz + tid] = mv_dist[tid];
          }
          __syncthreads();
        }

        // Stage 2: external merge
        if (tid < deg) {
          int id = lower_bound(tmp_dist,
                               tmp_id,
                               0, sz,
                               tmp_dist[sz + tid],
                               tmp_id[sz + tid]                           
                               );
          if (id != sz &&
              ((tmp_id[id] ^ tmp_id[sz + tid])
               & 0x7fffffff) == 0) id++;
          mv_pos[sz + tid] = id + tid;
        }
        for (int i = tid; i < sz; i += blockDim.x) {
          int id = lower_bound(tmp_dist + sz,
                               tmp_id + sz,
                               0, deg,
                               tmp_dist[i],
                               tmp_id[i]);
          mv_pos[i] = id + i;      
        }
        __syncthreads();
        for (int i = tid; i < sz + deg; i += blockDim.x) {
          //printf("%d %d\n", i, mv_pos[i]);
          assert(0 <= mv_pos[i] && mv_pos[i] < sz + max_m);
          mv_id[mv_pos[i]] = tmp_id[i];
          mv_dist[mv_pos[i]] = tmp_dist[i];
        }
        if (tid == 0) ctx_size = min(sz + deg, 2 * ef_search);
        __syncthreads();

        if (tid < 32) {
          int sz = ctx_size;
          int id = sz + 1;
          int target = (sz + 31) / 32 * 32;
          int count = 0;
          for (int i = tid; i < target; i += 32) {
            bool flag = (i != 0) && (i < sz) &&
              (((mv_id[i] ^ mv_id[i - 1]) & 0x7fffffff) == 0);
            unsigned mask = __ballot_sync(0xffffffff, flag);
            int mv = i - count - __popc(mask & ((1u << tid) - 1));
            uint32_t x_id = mv_id[i];
            float x_dist = mv_dist[i];
            __syncwarp();
            
            if (i < sz && !flag) {
              //assert(0 <= mv && mv < sz);
              tmp_id[mv] = x_id;
              tmp_dist[mv] = x_dist;
            }
            __syncwarp();
            if (!flag && id > mv && ((x_id & 0x80000000u) == 0)) {
              id = mv;        
            }
            //printf("%d %d %d %d %u\n", i, id, mv, flag, (tmp_id & 0x80000000u) == 0);
            count += __popc(mask);
            __syncwarp();
          }

#pragma unroll
          for (int offset = 32 / 2; offset > 0; offset /= 2) {
            int _id = __shfl_down_sync(0xffffffff, id, offset);
            id = min(_id, id);
          }
          //printf("??????? %d\n", tid);
          if (tid == 0)  {
            if (id >= ef_search) {
              ctx_u = u = -1;
            } else {
              ctx_u = u = tmp_id[id];
              ctx_size = min(sz - count, ef_search);
              tmp_id[id] |= 0x80000000u;
            }
            //printf("!!? %d %d\n", id, ctx_size);
          }
        } else if (tid < 64) { 
          float dist = square_sum_32(src_vec, graph.data, num_dims);    
          retset_push_32(distances + qid * topk, nns + qid * topk, found_cnt[qid],
                         topk, dist, u);
        }
        __syncthreads();
        u = ctx_u;
        __syncthreads();
      }
      
      
    }
  }
}

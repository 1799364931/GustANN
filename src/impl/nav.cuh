#pragma once

namespace gustann {


#if 1
  const int L = 4;
  const int R = 28;
  inline __global__ void get_entry_kernel
  (float* qdata_global, uint8_t* data_g, int* graph, int qcnt,
   const int num_nodes, const int num_dims, const int max_m,
   const int ef_search, const int entry, int* result,
   uint32_t* neighbor_id, float* neighbor_dist
   ) {
    int buffer_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int tid = threadIdx.x % 32;
    if (buffer_id >= qcnt) return;
    const int tsz = 32;
    
    data_type* data = (data_type*) data_g;

    const int max_dim = 96 * 4;
    __shared__ float qdata_shm[max_dim * 2];
    float* qdata;
    if (num_dims <= max_dim) {
      qdata = qdata_shm + (threadIdx.x / 32) * max_dim;
      for (int i = tid; i < num_dims; i += tsz) {
        qdata[i] = qdata_global[num_dims * buffer_id + i];
      }
    } else {
      qdata = qdata_global + num_dims * buffer_id;
    }
    __shared__ uint32_t ef_search_pq_id_shm[(L + R) * 2];
    __shared__ float ef_search_pq_dist_shm[(L + R) * 2];
    uint32_t* ef_search_pq_id = ef_search_pq_id_shm + (threadIdx.x / 32) * (L + R);
    float* ef_search_pq_dist = ef_search_pq_dist_shm + (threadIdx.x / 32) * (L + R);
    
    __shared__ int size_shm[2];
    int& size = size_shm[threadIdx.x / 32];
    
    if (tid == 0) size = 0;
    //if (tid == 0) for (int i = 0; i < num_dims; i++) printf("%lf! ", qdata[i]);    
    __syncwarp();

    float dist = square_sum_32(qdata, data + 1l * num_dims * entry, num_dims);
    if (tid == 0) {
      ef_search_pq_id[0] = entry;
      ef_search_pq_dist[0] = dist;
    } else {
      ef_search_pq_id[tid] = 0xffffffffu;
      ef_search_pq_dist[tid] = INFINITY;
    }
    //if (tid == 0) for (int i = 0; i < num_dims; i++) printf("%lf!3 ", qdata[i]);    
    __syncwarp();
    int idx = 0;
    while(idx != -1) {
      int u = ef_search_pq_id[idx];
      //if (tid == 0) printf("!!! %d %d\n", buffer_id, u);
      __syncwarp();
      if (tid == 0) {
        ef_search_pq_id[idx] |= 0x80000000u;
      }
      __syncwarp();
      int* edge = graph + 1l * max_m * u;
      for (int i = 0; i < R; i++) {
        int v = edge[i];
        if (v == -1) break;
        //if (tid == 0) printf("%d %d!!!\n", v, entry);
        float dist = square_sum_32(qdata, data + 1l * num_dims * v, num_dims);
        //if (tid == 0) { for (int i = 0; i < num_dims; i++) printf("%d %d! ", (int) qdata[i], data[num_dims * v + i]); printf("\n");}
        if (tid == 0) {
          ef_search_pq_id[i + L] = v;
          ef_search_pq_dist[i + L] = dist;
          //printf("%lf\n", dist);
        }
      }
      __syncwarp();
      //if (tid == 0) for (int i = 0; i < 32; i++) printf("!? %d %d %d %d %f\n", buffer_id, i, ef_search_pq_id[i] & 0x7fffffff, ef_search_pq_id[i] >> 31, ef_search_pq_dist[i]);
#pragma unroll
      for (int i = 0; i < L; i++) {
        float dist = ef_search_pq_dist[tid];
        int pos = tid;
        if (tid < i) dist = INFINITY;
        else if (i != 0 && ((ef_search_pq_id[tid] ^ ef_search_pq_id[i - 1]) & 0x7fffffffu) == 0) {
          ef_search_pq_id[i - 1] |= ef_search_pq_id[tid];
          dist = ef_search_pq_dist[tid] = INFINITY;
        }
        __syncwarp();
#pragma unroll
        for (int offset = 32 / 2; offset > 0; offset /= 2) {
          int _id = __shfl_down_sync(0xffffffff, pos, offset);
          float _value = __shfl_down_sync(0xffffffff, dist, offset);
          if (_value < dist) {
            dist = _value;
            pos = _id;
          }
        }
        __syncwarp();
        
        if (tid == 0 && i != pos) {
          float t = ef_search_pq_dist[i];
          float y = ef_search_pq_dist[pos];
          if (t != y) {
            ef_search_pq_dist[i] = y;            
            ef_search_pq_dist[pos] = t;
            int m = ef_search_pq_id[i];
            ef_search_pq_id[i] = ef_search_pq_id[pos];
            ef_search_pq_id[pos] = m;
            //printf("/\\ %d\n", pos);
          }
        }
        __syncwarp();
        //if (tid == 0) for (int i = 0; i < 32; i++) printf("!? %d %d %d %d %f\n", buffer_id, i, ef_search_pq_id[i] & 0x7fffffff, ef_search_pq_id[i] >> 31, ef_search_pq_dist[i]);
      }
      int val = __ballot_sync(0xffffffff, !(ef_search_pq_id[tid] & 0x80000000));
      if (tid == 0) {
        int x = __clz(__brev(val));
        if (x < L) idx = x;
        else idx = -1;
        //printf(":: %d\n", idx);
        //if (tid == 0) for (int i = 0; i < 32; i++) printf("!? %d %d %d %d %f\n", buffer_id, i, ef_search_pq_id[i] & 0x7fffffff, ef_search_pq_id[i] >> 31, ef_search_pq_dist[i]);

      }
      idx = __shfl_sync(0xffffffff, idx, 0);      
    }
    __syncwarp();
    if (tid == 0)
      result[buffer_id] = ef_search_pq_id[0] & 0x7fffffff;
    //if (tid == 0) printf("!!!! %d %d\n", buffer_id, result[buffer_id]);
  }

  
  
#else
  // Based on naiive CuHNSW style search
  __global__ void get_entry_kernel
  (float* qdata_global, uint8_t* data_g, int* graph,
   const int num_nodes, const int num_dims, const int max_m,
   const int ef_search, const int entry, int* result,
   uint32_t* neighbor_id, float* neighbor_dist
   ) {
    int buffer_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int tid = threadIdx.x % 32;
    float* qdata = qdata_global + num_dims * buffer_id;
    data_type* data = (data_type*) data_g;

    int aligned_ef = (ef_search + 31) / 32 * 32;
    uint32_t* ef_search_pq_id = neighbor_id + aligned_ef * buffer_id;
    float* ef_search_pq_dist = neighbor_dist + aligned_ef * buffer_id;
  
    __shared__ int size;
    if (threadIdx.x == 0) size = 0;
    __syncwarp();
    float dist = square_sum_32(qdata, data + 1l * num_dims * entry, num_dims);
    update_pq(ef_search_pq_id, ef_search_pq_dist, size,
              ef_search, dist, entry);
    __syncwarp();
    int idx = get_cand_id(ef_search_pq_id, ef_search_pq_dist, size);  
    while(idx != -1) {

      int u = ef_search_pq_id[idx];
      //if (tid == 0) printf("!!! %d %d\n", buffer_id, u);
      __syncwarp();
      if (tid == 0) {
        ef_search_pq_id[idx] |= 0x80000000u;
      }
      __syncwarp();
      int* edge = graph + 1l * max_m * u;
      for (int i = 0; i < max_m; i++) {
        int v = edge[i];
        if (v == -1) break;     
        float dist = square_sum_32(qdata, data + 1l * num_dims * v, num_dims);
        update_pq_insertion(ef_search_pq_id, ef_search_pq_dist, size,
                            ef_search, dist, v);
      }
      __syncwarp();
      //if (tid == 0) printf("!!! %d %d\n", buffer_id, size);
      idx = get_cand_id(ef_search_pq_id, ef_search_pq_dist, size);
      __syncwarp();
    }
  
    int cand = -1;
    dist = INFINITY;  
    int tsz = 32;
    for (int i = tid; i < size; i += tsz) {
      if (ef_search_pq_dist[i] < dist) {
        cand = i;
        dist = ef_search_pq_dist[i];
      }
    }
  
#pragma unroll
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
      int _cand = __shfl_down_sync(0xffffffff, cand, offset);
      float _dist = __shfl_down_sync(0xffffffff, dist, offset);
      if (_dist < dist) {
        dist = _dist;
        cand = _cand;
      }
    }
    cand = __shfl_sync(0xffffffff, cand, 0);

    if (tid == 0)
      result[buffer_id] = ef_search_pq_id[cand] & 0x7fffffff;
  }
  
  
#endif  

}


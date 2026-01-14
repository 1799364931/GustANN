#pragma once

namespace gustann {

  __global__ void __launch_bounds__(128, 14)
    merge_data_kernel
    (uint8_t *buffer, int32_t *request,
     //PQSearchData* pq_data,
     int num_chunks, float* pq_dists, uint8_t *compressed_data, 
     int nodes_per_page, int node_len, int data_len,
     int pq_offset, 
     const int max_m, const int ef_search, 
     uint32_t* neighbor_id, float* neighbor_dist, Data* data) {
    // Stage 0: Calc PQ
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nodeid_u = request[bid];
    //if (tid == 0 && bid == 0) printf("M %d\n", bid);
    if (nodeid_u == -1) return;
    //__shared__ long __tmp__[4];
    //if (tid == 0) __tmp__[0] = clock64();

    extern __shared__ uint8_t shm_pool[];
    //__shared__ uint8_t shm_pool[10000];
    int* mv_pos = (int*) shm_pool;
    uint32_t* mv_id = (uint32_t*)(mv_pos + ef_search + max_m);
    float* mv_dist = (float*)(mv_id + ef_search + max_m);
    uint32_t* tmp_id = (uint32_t*)(mv_dist + ef_search + max_m);
    float* tmp_dist = (float*)(tmp_id + ef_search + max_m);

    int offset = (max_m + ef_search + 31) / 32 * 32 * bid;
    Data& ctx = data[bid];
    int sz = ctx.size;
    int edge_offset = nodeid_u % nodes_per_page * node_len + data_len;    
    int* buffer_u = (int*)(buffer + bid * 4096 + edge_offset);

    for (int i = tid; i < sz; i += blockDim.x) {
      tmp_id[i] = neighbor_id[offset + i];
      tmp_dist[i] = neighbor_dist[offset + i];
    }
    __syncthreads();
    
    int deg = buffer_u[0];

#if 1
    //if (tid == 0) printf("??? %d %d %d %d\n", tid, nodeid_u, buffer_u[0], edge_offset);
    if (tid >= deg) {
      if (tid < max_m) {
        tmp_id[sz + tid] = 0xffffffff - tid;
        tmp_dist[sz + tid] = INFINITY;
      }    
    } else {
      int nodeid_v = tmp_id[sz + tid] = buffer_u[threadIdx.x + 1];
      //printf("%d!!!\n", nodeid_v);
      float* dist_vec = pq_dists + (pq_offset + bid) * PQSearchData::num_pivots * num_chunks;
      float dist = 0;
      uint8_t* data = compressed_data + (long) nodeid_v * num_chunks;
      /*
        if (pq_data->num_chunks % 4 == 0 && (size_t)(data) % 4 == 0) {
        int j = 0;
        int np = pq_data->num_pivots;
        
        while((size_t) data % 4 != 0) {
        dist += dist_vec[j * pq_data->num_pivots + data[j]];
        j++;
        }
        uint32_t* data_aligned = (uint32_t*)(data + j);
        float dist1 = 0, dist2 = 0, dist3 = 0, dist4 = 0;
        
        #pragma unroll 8
        while(j + 3 < pq_data->num_chunks) {
        uint32_t num = *data_aligned;
        uint32_t x = num & 0xff;
        uint32_t y = (num >> 8) & 0xff;
        uint32_t z = (num >> 16) & 0xff;
        uint32_t w = (num >> 24) & 0xff;
        dist1 += dist_vec[(j + 0) * pq_data->num_pivots + x];
        dist2 += dist_vec[(j + 1) * pq_data->num_pivots + y];
        dist3 += dist_vec[(j + 2) * pq_data->num_pivots + z];
        dist4 += dist_vec[(j + 3) * pq_data->num_pivots + w];
        j += 4;
        data_aligned++;
        }
        dist += dist1 + dist2 + dist3 + dist4;

        } else {
      */
      //printf("%d\n", nodeid_v);

#pragma unroll 32
      for (int j = 0; j < num_chunks; j++) {
        //int x = tmp_id[j + tid] & 0xff;
        dist += dist_vec[j * PQSearchData::num_pivots + data[j]];
      }

      //}

      tmp_dist[sz + tid] = dist;
      //printf("! %d %d %d %lf\n", blockIdx.x, tid, tmp_id[sz + tid], dist);
    }
#elif 0
    if (tid >= deg) {
      if (tid < max_m) {
        tmp_id[sz + tid] = 0xffffffff - tid;
        tmp_dist[sz + tid] = INFINITY;
      }    
    } else {
      int nodeid_v = tmp_id[sz + tid] = buffer_u[threadIdx.x + 1];    
    }
    __syncthreads();

    uint8_t* data_shm = (uint8_t*)  (tmp_dist + ef_search + max_m);
    int subtask = blockDim.x / num_chunks;
    int subid = tid / num_chunks;
    int xid = tid % num_chunks;
    if (subid < subtask) {
      for (int i = subid; i < deg; i += subtask) {
        uint8_t* data = compressed_data +
          (long) tmp_id[sz + i] * num_chunks;
        data_shm[i * (num_chunks + 1) + xid]
          = data[xid];
        //printf("%d %d %d\n", i, xid, data_shm[i * num_chunks + xid]);
      }

    }
    __syncthreads();

    float dist = 0;
    if (tid < deg) {
      uint8_t* data = data_shm + tid * (num_chunks + 1);
      float* dist_vec = pq_dists + (pq_offset + bid) * PQSearchData::num_pivots * num_chunks;
      for (int j = 0; j < num_chunks; j++) {
        dist += dist_vec[j * PQSearchData::num_pivots + data[j]];
        //printf("??? %d %d %lf\n", tid, j, dist);
      }
      tmp_dist[sz + tid] = dist;
      //printf("! %d %d %lf\n", blockIdx.x, tmp_id[sz + tid], dist);
    }
#else
    if (tid >= deg) {
      if (tid < max_m) {
        tmp_id[sz + tid] = 0xffffffff - tid;
        tmp_dist[sz + tid] = INFINITY;
      }    
    } else {
      int nodeid_v = tmp_id[sz + tid] = buffer_u[threadIdx.x + 1];    
    }
    __syncthreads();
    assert(num_chunks == 32 && (size_t) (compressed_data) % 4 == 0);
    const int np = 8;
    int subtask = blockDim.x / np;
    int subid = tid / np;
    int xid = tid % np;
    float* dist_vec = pq_dists + (pq_offset + bid) * PQSearchData::num_pivots * num_chunks;
    for (int i = subid; i < (deg + np - 1) / np * np; i += subtask) {
      float dist;
      if (i < deg) {
        uint32_t val = *((uint32_t*)(compressed_data + (long)(tmp_id[sz + i]) * num_chunks) + xid);
      
        int x = val & 0xff;
        int y = (val >> 8) & 0xff;
        int z = (val >> 16) & 0xff;
        int w = (val >> 24) & 0xff;
        float dist1 = dist_vec[(xid * 4 + 0) * PQSearchData::num_pivots + x];
        float dist2 = dist_vec[(xid * 4 + 1) * PQSearchData::num_pivots + y];
        float dist3 = dist_vec[(xid * 4 + 2) * PQSearchData::num_pivots + z];
        float dist4 = dist_vec[(xid * 4 + 3) * PQSearchData::num_pivots + w];
        dist = (dist1 + dist2) + (dist3 + dist4);
      } else {
        dist = 0;
      }
#pragma unroll
      for (int offset = np / 2; offset > 0; offset /= 2) {
        dist += __shfl_down_sync(0xffffffff, dist, offset);
      }
      if (xid == 0 && i < deg) {
        tmp_dist[sz + i] = dist;
        //printf("! %d %d %lf\n", blockIdx.x, tmp_id[sz + i], dist);
      }
    }
#endif
    //if (tid == 0) printf("!!! A %d\n", bid);
    // Stage 1: internal sort of tmp

    offset += sz;
    __syncthreads();
    //if (tid == 0) __tmp__[1] = clock64();
    //__syncthreads();

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
    //if (tid == 0) __tmp__[2] = clock64();
    //__syncthreads();

    /*
    {
      if (tid == 0) {
        for (int i = 0; i < max_m + sz; i++) {
          printf("!$# %d %d %d %d %f\n", bid, i,
                 tmp_id[i] & 0x7fffffff,
                 tmp_id[i] >> 31,
                 tmp_dist[i]);
        }
        
      }
      __syncthreads();
    }
    */
    /*
      {
      if (tid == 0) {
      for (int i = 1; i < max_m; i++) {
      if (neighbor_dist[offset + i] < neighbor_dist[offset + i - 1]) {
      for (int i = 0; i < max_m + sz; i++) {
      printf("!$# %d %d %d %d %f\n", bid, i,
      neighbor_id[offset - sz + i] & 0x7fffffff,
      neighbor_id[offset - sz + i] >> 31,
      neighbor_dist[offset - sz + i]);
      }
      for (int i = 0; i < max_m; i++) {
      int nodeid_v = buffer_u[i + 1];    
      float* dist_vec = pq_data->pq_dists + (pq_offset + bid) * pq_data->num_pivots * pq_data->num_chunks;
      float dist = 0;
      uint8_t* data = pq_data->compressed_data + (long) nodeid_v * pq_data->num_chunks;
      for (int j = 0; j < pq_data->num_chunks; j++) {
      dist += dist_vec[j * pq_data->num_pivots + data[j]];
      }
      printf("!#3 %d %lf\n", nodeid_v, dist);
      }
      assert(false);
      }
      }
      }
      __syncwarp();
      }
    */
    // Stage 2: external merge
    offset -= sz;
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
    /*
    {
      if (tid == 0) {
      for (int i = 0; i < sz + max_m; i++) {
      printf("!?$ %d %d\n", i, mv_pos[i]);
      }
      }
      __syncthreads();
    }
    */

    for (int i = tid; i < sz + deg; i += blockDim.x) {
      //assert(0 <= mv_pos[i] && mv_pos[i] < sz + max_m);
      mv_id[mv_pos[i]] = tmp_id[i];
      mv_dist[mv_pos[i]] = tmp_dist[i];
    }
    __syncthreads();

    int target = min(sz + deg, 2 * ef_search);    
    for (int i = tid; i < target; i += blockDim.x) {
      neighbor_id[offset + i] = mv_id[i];
      neighbor_dist[offset + i] = mv_dist[i];
    }
    if (threadIdx.x == 0) ctx.size = target;
    //if (threadIdx.x == 0) printf("!!!!!!\n");
    //if (tid == 0) __tmp__[3] = clock64();
    //__syncthreads();
    //if (tid == 0) printf("%ld %ld %ld %ld\n", __tmp__[0], __tmp__[1], __tmp__[2], __tmp__[3]);
    /*
    {
      if (tid == 0) {
        for (int i = 0; i < max_m + sz; i++) {
          printf("!$$ %d %d %d %d %f\n", bid, i,
                 neighbor_id[offset + i] & 0x7fffffff,
                 neighbor_id[offset + i] >> 31,
                 neighbor_dist[offset + i]);
        }
        
      }
      __syncthreads();
    }
    */
    /*
      {
      if (tid == 0) {
      for (int i = 1; i < ctx.size; i++) {
      if (neighbor_dist[offset + i] < neighbor_dist[offset + i - 1]) assert(false);
      }
      }
      __syncwarp();
      }
    */
  }

  __global__ void unify_kernel
  (float* qdata, uint8_t* buffer, int32_t* request,
   const int num_dims,
   const int max_m, const int ef_search,  const int topk,
   int* nns, float* distances, int* found_cnt,
   uint32_t* neighbor_id, float* neighbor_dist,
   int nodes_per_page, int node_len, int data_len, Data* data,
   int qcnt) {
    
    int bid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (bid >= qcnt) return;    
    int tid = threadIdx.x % 32;
    int offset = (max_m + ef_search + 31) / 32 * 32 * bid;
    //if (bid == 0 && tid == 0) printf("U %d\n", bid);
    float* src_vec = qdata + num_dims * bid;
    
    int node_u = request[bid];
    if (node_u == -1) return;
    //printf("????? %d\n", tid);
    //if (threadIdx.x == 0) printf("%d\n", node_u);
    int buffer_offset = node_u % nodes_per_page * node_len;
    
    data_type* buffer_u = (data_type *)(buffer + bid * 4096 + buffer_offset);
    float dist = square_sum_32(src_vec, buffer_u, num_dims);    
    retset_push_32(distances + bid * topk, nns + bid * topk, found_cnt[bid],
                   topk, dist, node_u);
    
    //printf("?!# %f %d\n", dist, node_u);
    /*
      {
      if (tid == 0) {
      for (int i = 0; i < topk; i++) {
      printf("$$$ %lf %d\n", distances[bid * topk + i],
      nns[bid * topk + i]);
      }
      }
      __syncwarp();
      }
    */
    Data& ctx = data[bid];
    int sz = ctx.size;
    int id = sz + 1;
    int target = (sz + 31) / 32 * 32;
    int count = 0;
    for (int i = tid; i < target; i += 32) {
      bool flag = (i != 0) && (i < sz) &&
        (((neighbor_id[offset + i] ^ neighbor_id[offset + i - 1]) & 0x7fffffff) == 0);
      unsigned mask = __ballot_sync(0xffffffff, flag);
      int mv = i - count - __popc(mask & ((1u << tid) - 1));
      uint32_t tmp_id = neighbor_id[offset + i];
      float tmp_dist = neighbor_dist[offset + i];
      __syncwarp();

      if (i < sz && !flag) {
        //assert(0 <= mv && mv < sz);
        neighbor_id[offset + mv] = tmp_id;
        neighbor_dist[offset + mv] = tmp_dist;
      }
      __syncwarp();
      if (!flag && id > mv && ((tmp_id & 0x80000000u) == 0)) {
        id = mv;        
      }
      //printf("%d %d %d %d %u\n", i, id, mv, flag, (tmp_id & 0x80000000u) == 0);
      count += __popc(mask);
      __syncwarp();
    }
    //printf("?????? %d\n", tid);
    /*
      {
      if (tid == 0) {

      for (int i = 0; i < sz; i++) {

      printf("!$? %d %d %d %d %f\n", bid, i,
      neighbor_id[offset + i] & 0x7fffffff,
      neighbor_id[offset + i] >> 31,
      neighbor_dist[offset + i]);

      }


      for (int i = 1; i < sz; i++) {
      if (neighbor_dist[i] < neighbor_dist[i - 1]) assert(false);
      }
      }
      __syncwarp();
      }
    */
    //id = __reduce_min_sync(0xffffffff, id);
    
#pragma unroll
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
      int _id = __shfl_down_sync(0xffffffff, id, offset);
      id = min(_id, id);
    }
    //printf("??????? %d\n", tid);
    if (tid == 0)  {
      if (id >= ef_search) {
        request[bid] = -1;
        ctx.size = 0;
      } else {
        request[bid] = neighbor_id[offset + id];
        ctx.size = min(sz - count, ef_search);
        neighbor_id[offset + id] |= 0x80000000u;
      }
      //printf("!!? %d %d\n", id, ctx.size);
    }
    //printf("???????? %d\n", tid);
    /*
      {
      if (tid == 0) {
      for (int i = 1; i < ctx.size; i++) {
      if (neighbor_dist[offset + i] < neighbor_dist[offset + i - 1]) assert(false);
      }
      }
      __syncwarp();
      }
    */
  }  

}

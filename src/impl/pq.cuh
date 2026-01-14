#pragma once

#include "../pq_search.hpp"
#include "../common_cuda.cuh"
#include "util.cuh"

namespace gustann {
  __inline__ __device__ void PQSearchData::init_query(float *q, int batch_offset) {
    // AB_BUFFER
    float* dist_vec = pq_dists + (blockIdx.x + batch_offset) * num_pivots * num_chunks;

    for (size_t i = threadIdx.x; i < (size_t) num_pivots * num_chunks; i += blockDim.x) {
      dist_vec[i] = 0;
    }
    /*
      if (threadIdx.x == 0) {
      for (int i = 0; i < dim; i++) printf("%lf", q[i]);
      printf("\n");
      }
      __syncthreads();
    */
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
      q[i] -= centroid[i];
    }
    /*
      __syncthreads();
      if (threadIdx.x == 0) {
      for (int i = 0; i < dim; i++) printf("%lf", q[i]);
      printf("\n");
      }
    */
    __syncthreads();


    for (int i = 0; i < dim; i++) {
      int idx = chunk_id[i];
      for (int j = threadIdx.x; j < num_pivots; j += blockDim.x) {
        float dist = q[i] - pivots_t[i * num_pivots + j];
        //if (threadIdx.x == 0) printf("! %lf %lf %lf\n", q[0], q[i], dist);
        dist_vec[idx * num_pivots + j] += dist * dist;
      }
      __syncthreads();
    }
    /*
      for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 5; j++) {
      if (threadIdx.x == 0) {
      printf("%d %d %d %lf\n", blockIdx.x, i, j, dist_vec[i * num_pivots + j]);
      }
      }
      }
    */
    // restore for future data
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
      q[i] += centroid[i];
    }


  }

  __inline__ __device__ float PQSearchData::compute_dist(int idx, int batch_offset) {
    static __shared__ float shared[32];
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // AB_BUFFER
    float* dist_vec = pq_dists + (blockIdx.x + batch_offset) * num_pivots * num_chunks;
    float dist = 0;
    uint8_t* data = compressed_data + idx * num_chunks;
    for (int i = threadIdx.x; i < num_chunks; i += blockDim.x) {
      dist += dist_vec[i * num_pivots + data[i]];
    }
    float val = gustann::warp_reduce_sum(dist);
  
    // write out the partial reduction to shared memory if appropiate
    if (lane == 0) {
      shared[warp] = val;
    }
    __syncthreads();

    // if we we don't have multiple warps, we're done
    if (blockDim.x <= WARP_SIZE) {
      return shared[0];
    }

    // otherwise reduce again in the first warp
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane]: 0.0f;
    if (warp == 0) {
      val = gustann::warp_reduce_sum(val);
      // broadcast back to shared memory
      if (threadIdx.x == 0) {
        shared[0] = val;
      }
    }
    __syncthreads();
    return shared[0];
  }

  __inline__ __device__ void PQSearchData::compute_dist(int* idx, float* result, int cnt) {
    float* dist_vec = pq_dists + blockIdx.x * num_pivots * num_chunks;

    for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
      uint8_t* data = compressed_data + idx[i] * num_chunks;
      float dist = 0;
#pragma unroll
      for (int j = 0; j < num_chunks; j++) {
        dist += dist_vec[j * num_pivots + data[j]];
      }
      result[i] = dist;
    }
  }

  __inline__ __device__ void retset_push(float* distance, int* idx, int& size, int max_size, float value, int value_idx) {
    __shared__ bool found_flag;
    __shared__ unsigned int s[32];
    if (threadIdx.x == 0) found_flag = 0;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
  
    //if (threadIdx.x == 0) printf("!!!%d\n", size);
    for (int i = 0; i < size; i += blockDim.x) {
      int p = size - i - 1 - threadIdx.x;
      bool flag = p < size && p >= 0;
      __syncthreads();
      float tmp_d = flag ? distance[p] : 0;
      int tmp_i = flag ? idx[p] : 0;
      __syncthreads();
      if (flag && tmp_d > value && p + 1 < max_size) {
        distance[p + 1]  = tmp_d;
        idx[p + 1] = tmp_i;
      }
      __syncthreads();
      unsigned int mask = __ballot_sync(0xffffffff, flag && tmp_d > value);
      if (threadIdx.x == 0) s[warp] = mask;
      __syncthreads();

      if ((warp == 0 || s[warp - 1] == 0xffffffff) && (mask + 1) == (1u << lane)) {
        if (p + 1 < max_size) {
          distance[p + 1] = value;
          idx[p + 1] = value_idx;
        }
        //      printf("!!%d %x\n", p + 1, mask);
        found_flag = 1;
      }
      __syncthreads();
      if (found_flag) break;
      //if (mask != 0xffffffff) break;    
    }
    if (!found_flag && threadIdx.x == 0) {
      distance[0] = value;
      idx[0] = value_idx;
      //printf("!!0\n");
    }
    if (size + 1 <= max_size && threadIdx.x == 0) size++;
    /*
      if (threadIdx.x == 0) {
      for (int i = 0; i < size; i++) {
      printf("%lf(%d) ", distance[i], idx[i]);
      }
      printf("\n");
      }
    */
  }
}

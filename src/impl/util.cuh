#pragma once

#include "../common_cuda.cuh"

namespace gustann {
  // https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
  // from cuhnsw
  __inline__ __device__
  float warp_reduce_sum(float val) {
#if __CUDACC_VER_MAJOR__ >= 9
    // __shfl_down is deprecated with cuda 9+. use newer variants
    unsigned int active = __activemask();
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      val = val + __shfl_down_sync(active, val, offset);
    }
#else
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      val = val + __shfl_down(val, offset);
    }
#endif
    return val;
  }

  template <class T>
  __inline__ __device__
  float square_sum_32(const float * a, T* b, const int num_dims) {
    __syncwarp();
    
    // figure out the warp/ position inside the warp
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    float val = 0;
        
    for (int i = lane; i < num_dims; i += 32) {
      float _val = a[i] - (float)(b[i]);
      val += _val * _val;
    }
    __syncwarp();
#pragma unroll
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);
  }

  __inline__ __device__ void retset_push_32(float* distance, int* idx, int& size, int max_size, float value, int value_idx) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    bool found_flag = false;
    //if (threadIdx.x == 0) printf("!!!%d\n", size);
    for (int i = 0; i < size; i += 32) {
      int p = size - i - 1 - lane;
      bool flag = p < size && p >= 0;
      __syncwarp();
      float tmp_d = flag ? distance[p] : 0;
      int tmp_i = flag ? idx[p] : 0;
      __syncwarp();
      if (flag && tmp_d > value && p + 1 < max_size) {
        distance[p + 1]  = tmp_d;
        idx[p + 1] = tmp_i;
      }
      __syncwarp();
      unsigned int mask = __ballot_sync(0xffffffff, flag && tmp_d > value);
      __syncwarp();
    

      if ((mask + 1) == (1u << lane)) {
        if (p + 1 < max_size) {
          distance[p + 1] = value;
          idx[p + 1] = value_idx;
        }
        //      printf("!!%d %x\n", p + 1, mask);
        found_flag = 1;
      }
      found_flag = __any_sync(0xffffffff, found_flag);
      if (found_flag) break;
      //if (mask != 0xffffffff) break;    
    }
    if (!found_flag && lane == 0) {
      distance[0] = value;
      idx[0] = value_idx;
      //printf("!!0\n");
    }
    if (size + 1 <= max_size && lane == 0) size++;
    /*
      if (threadIdx.x == 0) {
      for (int i = 0; i < size; i++) {
      printf("%lf(%d) ", distance[i], idx[i]);
      }
      printf("\n");
      }
    */
  }

  // In kernel lower bound. 
  __inline__ __device__ int lower_bound(float* dist_arr, uint32_t* id_arr, int l, int r, float d, int x) {
    x = x & 0x7fffffff;
    while(l < r) {
      int mid = (l + r) / 2;
      if (dist_arr[mid] < d ||
          (dist_arr[mid] == d && (id_arr[mid] & 0x7fffffff) < x)) {
        l = mid + 1;
      } else {
        r = mid;
      }
    }
    return l;
  }
  

  
}

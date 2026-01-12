#pragma once
#include "common.h"
#include <page_cache.h>

using cuda_scalar = float;

const char *const ctrls_paths[6] = {"/dev/libnvm0","/dev/libnvm1","/dev/libnvm2","/dev/libnvm3","/dev/libnvm4","/dev/libnvm5"};

struct BaMConfig {
  int num_ctrls = 1;
  int queue_depth = 1024;
  int num_queues = 1;
  int cuda_device = 0;
  int nvm_namespace = 1;
  int page_size = 4096;
  int num_page = 1024; // cached page
} bam_config;

struct BaMContext {
  std::vector<Controller *> ctrls;
  page_cache_t *h_pc = nullptr;
  range_t<cuda_scalar> *h_range = nullptr;
  std::vector<range_t<cuda_scalar> *> vr;
  array_t <cuda_scalar> *a = nullptr;
  range_d_t <cuda_scalar> *d_range = nullptr;  
} bam_data;

inline void init_bam(long num_eles) {
  spdlog::set_pattern("[%^%-8l%$] %Y-%m-%d %H:%M:%S [%s:%#] %v");
  spdlog::set_level(spdlog::level::info);

  for (int i = 0; i < bam_config.num_ctrls; i++) {
    bam_data.ctrls.push_back
      (new Controller(ctrls_paths[i],
                      bam_config.nvm_namespace,
                      bam_config.cuda_device,
                      bam_config.queue_depth,
                      bam_config.num_queues));
  }
  assert(bam_config.page_size % sizeof(cuda_scalar) == 0);

  int ele_per_page = bam_config.page_size / sizeof(cuda_scalar);  
  //int num_pages = (num_eles + ele_per_page - 1) / ele_per_page;
  int num_pages = bam_config.num_page;

  DEBUG("Num page: {}", num_pages);
  DEBUG("Page size: {}", bam_config.page_size);
  DEBUG("Data size: {} B", num_eles * sizeof(cuda_scalar));

  bam_data.h_pc = new page_cache_t(bam_config.page_size,
                                   num_pages,
                                   bam_config.cuda_device,
                                   bam_data.ctrls[0][0],
                                   64,
                                   bam_data.ctrls);
  //page_cache_t* d_pc = (page_cache_t*) (bam_data.h_pc->d_pc_ptr);
  bam_data.h_range = new range_t<cuda_scalar>
    (0, num_eles, 0,
     (num_eles + ele_per_page - 1) / ele_per_page,
     0, bam_config.page_size, bam_data.h_pc,
     bam_config.cuda_device,
     STRIPE);
  bam_data.d_range = (range_d_t<cuda_scalar>*) bam_data.h_range->d_range_ptr;
  bam_data.vr.push_back(bam_data.h_range);
  bam_data.a = new array_t<cuda_scalar> (num_eles, 0, bam_data.vr, bam_config.cuda_device);\

}

template <class T>
inline __global__ void copy_data_to_ssd(array_d_t<T>* dest, T* src, size_t len, size_t offset = 0) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t thread_num = blockDim.x * gridDim.x;
  for (size_t i = tid; i < len; i += thread_num) {
    (*dest)(i + offset, src[i]);
  }
}

template <class T>
inline __global__ void copy_data_from_ssd(array_d_t<T>* src, T* dest, size_t len, size_t offset = 0) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t thread_num = blockDim.x * gridDim.x;
  for (size_t i = tid; i < len; i += thread_num) {
    dest[i] = (*src)[i + offset];
  }
}

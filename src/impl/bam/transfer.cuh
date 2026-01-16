#pragma once

#include "page_cache.h"

namespace gustann {
  template <class T>
  __global__ void copy_data_to_ssd(array_d_t<T> *dest, T *src,
                                   size_t len, size_t offset) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t thread_num = blockDim.x * gridDim.x;
    for (size_t i = tid; i < len; i += thread_num) {
      (*dest)(i + offset, src[i]);
    }
  }

  __global__ void copy_page_to_ssd(Controller **ctrls, page_cache_d_t *pc,
                                   uint64_t n_ctrls,
                                   uint8_t* src, size_t start_lba, size_t len, size_t page_size) {
#ifdef ASYNC_READ
    uint64_t cache_pages = pc->n_pages;
    
    
    uint32_t batch_per_block = cache_pages / gridDim.x;
    //if (blockIdx.x == 0 && threadIdx.x == 0) printf("%lu %lu %lu %u\n", start_lba, len, cache_pages, batch_per_block);
    uint32_t fetch_head = 0, fetch_tail = 0;
    
    for (int i = blockIdx.x; i < len; i += gridDim.x) {
      if (fetch_tail - fetch_head == batch_per_block) {
        if (threadIdx.x == 0) {
          uint32_t cache_idx = fetch_head % batch_per_block + batch_per_block * blockIdx.x;
          write_data_await(pc->cache_pages[cache_idx].qp,
                           &pc->cache_pages[cache_idx].ctx);
        }
        fetch_head++;
      }
      __syncthreads();
      uint32_t cache_idx = (fetch_tail++) % batch_per_block + batch_per_block * blockIdx.x;
      uint64_t* dest_p = (uint64_t *) (pc->base_addr + cache_idx * page_size);
      uint64_t* src_p = (uint64_t *) (src + i * page_size);
      for (int j = threadIdx.x; j < page_size / sizeof(uint64_t); j += blockDim.x) {
        dest_p[j] = src_p[j];
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        uint64_t dest_page = start_lba + i;
        uint32_t ctrl = dest_page % n_ctrls;
        uint32_t block = dest_page / n_ctrls;
        uint32_t queue = ctrls[ctrl]->queue_counter.fetch_add(1, simt::memory_order_relaxed) %
          (ctrls[ctrl]->n_qps);
        
        QueuePair *qp = &ctrls[ctrl]->d_qps[queue];
        write_data_async(pc, qp, block * pc->n_blocks_per_page, pc->n_blocks_per_page,
                         cache_idx, &pc->cache_pages[cache_idx].ctx);
        pc->cache_pages[cache_idx].qp = qp;
      }
    }
    //if (threadIdx.x == 0) printf("%u %u\n", fetch_head, fetch_tail);
    __syncthreads();
    for (uint32_t i = threadIdx.x + fetch_head; i < fetch_tail; i += blockDim.x) {
      uint32_t cache_idx = i % batch_per_block + batch_per_block * blockIdx.x;

      write_data_await(pc->cache_pages[cache_idx].qp,
                       &pc->cache_pages[cache_idx].ctx);
    }

#else

    for (size_t i = blockIdx.x; i < len; i += gridDim.x) {
      __syncthreads();
      uint32_t cache_idx = blockIdx.x;
      uint64_t* dest_p = (uint64_t *) (pc->base_addr + cache_idx * page_size);
      uint64_t* src_p = (uint64_t *) (src + i * page_size);
      for (int j = threadIdx.x; j < page_size / sizeof(uint64_t); j += blockDim.x) {
        dest_p[j] = src_p[j];
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        uint64_t dest_page = start_lba + i;
        uint32_t ctrl = dest_page % n_ctrls;
        uint32_t block = dest_page / n_ctrls;
        uint32_t queue = ctrls[ctrl]->queue_counter.fetch_add(1, simt::memory_order_relaxed) %
          (ctrls[ctrl]->n_qps);
        write_data(pc, (ctrls[ctrl]->d_qps) + queue, block * pc->n_blocks_per_page,
                   pc->n_blocks_per_page, cache_idx);

      }
    }
#endif
  }

  __global__ void f__k(array_d_t<uint8_t>* a) {
    int x = 159494;
    printf("%x %x %x %x\n", (*a)[x * 4096 + 128 * 4], (*a)[x * 4096 + 128 * 4 + 1], (*a)[x * 4096 + 128 * 4 + 2], (*a)[x * 4096 + 128 * 4 + 3]);
  }

  __global__ void fetch_all_data(array_d_t<uint8_t>* a, int num_pages) {
    for (int i = threadIdx.x; i < num_pages; i += blockDim.x) {
      (*a)[i * 4096];
    }
  }

}

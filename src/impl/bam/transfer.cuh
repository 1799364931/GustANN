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
    // 计算全局线程 ID
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算步长，用于处理数据量超过总线程数的情况
    uint64_t stride = blockDim.x * gridDim.x;

    // 负载均衡策略：与参考代码一致，使用 warp (tid/32) 粒度分配控制器和队列
    uint32_t ctrl = (tid / 32) % n_ctrls;
    // 注意：需确保 ctrls[ctrl] 已初始化
    uint32_t queue = (tid / 32) % (ctrls[ctrl]->n_qps);

    // 计算总共需要写入多少个 Page
    // 向上取整：(len + page_size - 1) / page_size
    size_t n_pages = (len + page_size - 1) / page_size;

    // 循环处理每个 Page
    for (size_t i = tid; i < n_pages; i += stride) {
      // 1. 计算 SSD 上的起始 Block 地址
      // 获取当前队列的 block_size_log (通常 block size 为 4KB, log 为 12)
      uint32_t blk_size_log = ctrls[ctrl]->d_qps[queue].block_size_log;
        
      // 计算一个 Page 包含多少个 Block
      uint64_t n_blocks = page_size >> blk_size_log;

      // 计算当前页在 SSD 上的起始 Block 编号
      // start_lba 是起始 Page 编号，i 是偏移量
      uint64_t current_page_idx = start_lba + i;
      uint64_t start_block = (current_page_idx * page_size) >> blk_size_log;

      // 2. 数据搬运 (从 src 到 page cache)
      // write_data 通常从 pc->base_addr 指定的 DMA 缓冲区读取数据
      // 我们假设 pc_idx 直接对应当前的各种处理索引 i (线性映射)
      // 注意：在实际系统中需确保 i 不超过 pc->n_pages
      uint64_t pc_idx = i; 

      // 计算源地址和目标地址
      // pc->base_addr 类型未知，强制转换为字节指针进行计算
      uint8_t* dst_ptr = (uint8_t*)pc->base_addr + (pc_idx * page_size);
      uint8_t* src_ptr = src + (i * page_size);

      // 将数据从源 src 复制到 page cache 的 DMA 缓冲区
      // 注意：在提交线程中进行内存拷贝效率较低，但为了满足函数签名要求需执行此步
      for (size_t b = 0; b < page_size; ++b) {
        // 处理最后一个 Page 可能不足 page_size 的情况
        if ((i * page_size + b) < len) {
          dst_ptr[b] = src_ptr[b];
        } else {
          dst_ptr[b] = 0; // 填充 0
        }
      }

      // 3. 提交写请求
      // 参数说明：
      // pc: page cache 结构体
      // qp: 队列指针 (ctrls[ctrl]->d_qps) + queue
      // start_block: SSD 起始块地址
      // n_blocks: 写入块数量
      // pc_idx: page cache 中的槽位索引 (用于定位 DMA 地址)
      write_data(pc, (ctrls[ctrl]->d_qps) + queue, start_block, n_blocks, pc_idx);
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

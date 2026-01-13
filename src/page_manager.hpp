#pragma once
#include "page_cache.h"

namespace gustann {
  struct SimpleCacheStat {
    uint64_t access_cnt;
    uint64_t prefetch_cnt;
    uint64_t access_hit;
    uint64_t prefetch_hit;
    uint64_t read_ready;
  };

  template <class T> struct NodeData {
    int deg;
    int* edge;
    T* data;
    __inline__ __device__ int operator[] (int x) { return edge[x];}
  };    
  
  struct SimpleCacheDevice {
    page_cache_d_t *cache;  
    uint64_t pages_per_block;
    SimpleCacheStat* stats;

    int nodes_per_page;
    int node_len;
    int data_len;
    int page_size;
/*
    __inline__ __device__ void init();
    __inline__ __device__ void prefetch (int nodeid) {
      int block = get_block(nodeid);
      acquire_page(block, true);
    }

    template <class T>
    __inline__ __device__ NodeData<T> read (int nodeid) {
      int block = get_block(nodeid);
      int offset = get_offset(nodeid);
      uint8_t* page = acquire_page(block, false) + offset;
      T* data = (T*) page;
      int* result = (int*)(page + data_len);
      return (NodeData<T>) {
        result[0],
        result + 1,
        data,
      };
    }

  private:
    __inline__ __device__ int get_block(int nodeid) {
      return nodeid / nodes_per_page;
    }
    __inline__ __device__ int get_offset(int nodeid) {
      return nodeid % nodes_per_page * node_len;
    }

    __inline__ __device__ uint8_t* acquire_page(int blockid, bool prefetch);
*/  
  };

  struct SimpleCache {
    SimpleCacheDevice device_data;
    SimpleCacheStat* stats = nullptr;
    uint64_t block_cnt;
    void init (page_cache_d_t *cache, uint64_t num_pages, uint64_t block_cnt, int nodes_per_page,
               int node_len, int data_len, int page_size);
    void print_stat();
    void reset_stat();
    
  };
}


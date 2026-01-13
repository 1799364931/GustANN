#pragma once

#include "def.cuh"
#include "page_cache.h"
#include "page_wrapper.cuh"

#include "../pq.cuh"
#include "../calc.cuh"

namespace gustann {
  template <class T>
  __inline__ __device__ void search_disk_graph_kernel_inner_with_pq
  (float* qdata, const int num_qnodes, DiskData* data,
   const int num_nodes, const int num_dims, const int max_m,
   const int ef_search, const int* entries, const int topk,
   int* nns, float* distances, int* found_cnt,
   int* visited_table, int* visited_list, 
   const int visited_table_size, const int visited_list_size, int64_t* acc_visited_cnt,
   Neighbor* neighbors, int nodes_per_page, int node_len, int data_len,
   PQSearchData* pq_data) {
    static __shared__ int size;
    const int MAX_SEARCH_EF = 300;
    //static __shared__ Neighbor ef_search_pq[MAX_SEARCH_EF];

    Neighbor* ef_search_pq = neighbors + ef_search * blockIdx.x;

    static __shared__ int visited_cnt;
    static __shared__ int prefetch_cnt;
    int* _visited_table = visited_table + visited_table_size * blockIdx.x;
    int* _visited_list = visited_list + visited_list_size * blockIdx.x;
    PageData pager;
    pager.nodes_per_page = nodes_per_page;
    pager.node_len = node_len;
    pager.disk = data;
    pager.data_len = data_len;
    pager.init();

    static __shared__ int final_size;
    
    int64_t start = clock64();
    int cnt_acc = 0, cnt_all = 0;
    __shared__ int arr[32];
    for (int i = blockIdx.x; i < num_qnodes; i += gridDim.x) {
      if (threadIdx.x == 0) {
        size = 0;
        visited_cnt = 0;
        prefetch_cnt = 0;
        final_size = 0;
      }
      __syncthreads();
      int cnt_fail = 0, cnt_read = 0, cnt_rounds = 0;
      // initialize entries
      float* src_vec = qdata + i * num_dims;
      pq_data->init_query(src_vec);
      //if (threadIdx.x == 0)printf("!!!! %lf\n", src_vec[0]);
      //ReadCtx ctx0;
      PushNodeToSearchPqBam(ef_search_pq, &size, ef_search, pq_data->compute_dist(entries[i]), entries[i]);
      //pager.drop(ctx0);
      if (CheckVisited(_visited_table, _visited_list, visited_cnt, entries[i], 
                       visited_table_size, visited_list_size)) 
        continue;
      __syncthreads();

      int idx = GetCand(ef_search_pq, size, false);
      //if (threadIdx.x == 0) printf("%d!!\n", idx);
      //static __shared__ int cand_arr[32];
      //static __shared__ int cand_cnt;
      //static __shared__ float cand_dist[32];

      int64_t t1 = 0;
      while (idx >= 0) {
        __syncthreads();
        if (threadIdx.x == 0) ef_search_pq[idx].checked = true;
        int entry = ef_search_pq[idx].nodeid;        
        __syncthreads();

        cnt_rounds++;



        //if (threadIdx.x == 0) pager.prefetch(entry);
        if (threadIdx.x == 0) arr[0] = entry;
        const int MAGIC = 1;
        for (int k = 1; k < MAGIC; k++) {
          int idx = GetCand(ef_search_pq, size, false);
          if (idx < 0) {
            if (threadIdx.x == 0) arr[k] = -1;
            continue;
          }
          if (threadIdx.x == 0) ef_search_pq[idx].checked = true;
          int entry = ef_search_pq[idx].nodeid;
          if (threadIdx.x == 0) arr[k] = entry;
        }
        __syncthreads();
        if (threadIdx.x > 0 && threadIdx.x < MAGIC) {
          if (arr[threadIdx.x] != -1) pager.prefetch(arr[threadIdx.x]);
        }
        __syncthreads();

        for (int k = 0; k < MAGIC; k++) {
          int entry = arr[k];
          if (arr[k] == -1) continue;
          //if (i == 0 && threadIdx.x == 0) printf("%d\n", entry);
          ReadCtx ctx1;
          t1 -= clock64();
          GraphData<T> graph = pager.get_graph<T>(entry, ctx1);
          t1 += clock64();
          float dist = square_sum(src_vec, graph.data, num_dims);
          retset_push(distances + i * topk, nns + i * topk, final_size, topk, dist, entry);
          //if (threadIdx.x == 0) printf("??? %d %lf %d\n", entry, dist, final_size);
          //if (threadIdx.x == 0) printf("%d %d %d %d\n", i, entry, graph.deg, graph[0]);

          //if (threadIdx.x == 0) cand_cnt = 0;
          //__syncthreads();
          for (int j = 0, tot = 0, end = graph.deg;
               j < end; j++, tot++) {
            int dstid = graph[j];
#if 0
            if (!CheckVisited(_visited_table, _visited_list, visited_cnt, dstid, 
                              visited_table_size, visited_list_size) &&
                !CheckAlreadyExists(ef_search_pq, size, dstid)) {
              if (threadIdx.x == 0) cand_arr[cand_cnt++] = dstid;
            }
            __syncthreads();
            if (cand_cnt == 32 || j + 1 == end) {
              pq_data->compute_dist(cand_arr, cand_dist, cand_cnt);
              __syncthreads();
              for (int k = 0; k < cand_cnt; k++) {
                PushNodeToSearchPqBam(ef_search_pq, &size, ef_search, cand_dist[k], cand_arr[k]);
              }
              __syncthreads();
              if (threadIdx.x == 0) cand_cnt = 0;
            }
#endif
            if (CheckVisited(_visited_table, _visited_list, visited_cnt, dstid, 
                             visited_table_size, visited_list_size) ||
                CheckAlreadyExists(ef_search_pq, size, dstid)) {
              continue;
            }
            float dist = pq_data->compute_dist(dstid);
            //if (threadIdx.x == 0) printf("!!!%d %d %lf\n", i, dstid, dist);
            PushNodeToSearchPqBam(ef_search_pq, &size, ef_search, dist, dstid);
            //if (threadIdx.x == 0) printf("!!! %d %d %lf\n", i, dstid, dist);
      
            __syncthreads();
            
          }

          //if (threadIdx.x == 0) printf("A\n");
          //if (threadIdx.x == 0) printf("??%d\n", i);
          pager.drop(ctx1);
        }
        __syncthreads();
        idx = GetCand(ef_search_pq, size, false);
        //cnt_acc += (idx == idx2), cnt_all++;
      }
      //if (threadIdx.x == 0) printf("#####%d\n", i);
      __syncthreads();
      

      for (int j = threadIdx.x; j < visited_cnt; j += blockDim.x) {
        _visited_table[_visited_list[j]] = -1;
      }
      __syncthreads();
      // get sorted neighbors
      /*
        if (threadIdx.x == 0) {
        int size2 = size;
        while (size > 0) {
        if (size <= topk) {
        nns[i * topk + size - 1] = ef_search_pq[0].nodeid;
        distances[i * topk + size - 1] = ef_search_pq[0].distance;
        }
        //printf("%lf\n", distances[i * topk + size - 1]);
        PqPop(ef_search_pq, &size);
        }
        found_cnt[i] = size2 < topk? size2: topk;
        }
      */
      if (threadIdx.x == 0) found_cnt[i] = final_size;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      acc_visited_cnt[blockIdx.x] += clock64() - start;
    }
  }

  
#ifdef ASYNC_READ
  template <class T>
  __inline__ __device__ void search_disk_graph_kernel_inner
  (const float* qdata, const int num_qnodes, DiskData* data,
   const int num_nodes, const int num_dims, const int max_m,
   const int ef_search, const int* entries, const int topk,
   int* nns, float* distances, int* found_cnt,
   int* visited_table, int* visited_list, 
   const int visited_table_size, const int visited_list_size, int64_t* acc_visited_cnt,
   Neighbor* neighbors, int nodes_per_page, int node_len, int data_len) {

    static __shared__ int size;
    const int PREFETCH_LENGTH = 24;
    static __shared__ int req_seq[PREFETCH_LENGTH];
    
    Neighbor* ef_search_pq = neighbors + ef_search * blockIdx.x;

    static __shared__ int visited_cnt;
    static __shared__ int prefetch_cnt;
    int* _visited_table = visited_table + visited_table_size * blockIdx.x;
    int* _visited_list = visited_list + visited_list_size * blockIdx.x;
    PageData pager;
    pager.nodes_per_page = nodes_per_page;
    pager.node_len = node_len;
    pager.disk = data;
    pager.data_len = data_len;
    

    int cnt_acc = 0, cnt_all = 0;
    for (int i = blockIdx.x; i < num_qnodes; i += gridDim.x) {
      if (threadIdx.x == 0) {
        size = 0;
        visited_cnt = 0;
        prefetch_cnt = 0;
      }
      __syncthreads();
      int cnt_fail = 0, cnt_read = 0, cnt_rounds = 0;
      // initialize entries
      const float* src_vec = qdata + i * num_dims;
      ReadCtx ctx0;
      PushNodeToSearchPqBam(ef_search_pq, &size, ef_search, pager.get_data<T>(entries[i], ctx0),
                            num_dims, src_vec, entries[i]);
      pager.drop(ctx0);
      if (CheckVisited(_visited_table, _visited_list, visited_cnt, entries[i], 
                       visited_table_size, visited_list_size)) 
        continue;
      __syncthreads();

      //printf("%d %f %d\n", ef_search_pq[0].nodeid, ef_search_pq[0].distance, size);
      
      // iterate until converge
      /*      
              {
              float* __ = pager.get_data(0, ctx0);
        
              float res = square_sum(src_vec, __, num_dims);
              if (threadIdx.x == 0) {
              for (int i = 0; i < 10; i++) printf("%f ", __[i]);
              printf("\n");
              }
              pager.drop(ctx0);
              }
      */
      int idx = GetCand(ef_search_pq, size, false);

      while (idx >= 0) {
        __syncthreads();
        if (threadIdx.x == 0) ef_search_pq[idx].checked = true;
        int entry = ef_search_pq[idx].nodeid;
        __syncthreads();

        cnt_rounds++;

        if (threadIdx.x < PREFETCH_LENGTH) {
          req_seq[threadIdx.x] = -1;
        }
        ReadCtx ctx1;

        /*
          int idx2 = GetCand(ef_search_pq, size, false);
          if (idx2 >= 0) {
          //if (blockIdx.x == 0 && threadIdx.x == 0)printf("%d %d\n", entry, ef_search_pq[idx2].nodeid);
          pager.prefetch(ef_search_pq[idx2].nodeid);
          }
        */
        GraphData<T> graph = pager.get_graph<T>(entry, ctx1);
        //if (threadIdx.x == 0) printf("%d %d %d %d\n", i, entry, graph.deg, graph[0]);

        for (int j = 0, tot = 0, end = graph.deg;
             j < end; j++, tot++) {
          int dstid = graph[j];

          if (!CheckVisited(_visited_table, _visited_list, visited_cnt, dstid, 
                            visited_table_size, visited_list_size)) {
            if (threadIdx.x == 0) req_seq[tot % PREFETCH_LENGTH] = dstid;
            //if (blockIdx.x == 0 && threadIdx.x == 0) printf("!!%d %d %d\n", entry, dstid, tot);
          }
          __syncthreads();

          if ((tot + 1) % PREFETCH_LENGTH == 0 || j == end - 1) {
            if (threadIdx.x < PREFETCH_LENGTH && req_seq[threadIdx.x] != -1) {
              pager.prefetch(req_seq[threadIdx.x]);
              prefetch_cnt++;
            }
            __syncthreads();
            for (int k = 0; k < PREFETCH_LENGTH; k++) {
              int dstid = req_seq[k];
              //if (blockIdx.x == 0 && threadIdx.x == 0) printf("%d %d\n", entry, dstid);
              if (dstid == -1) continue;
              //const float* dst_vec = data + num_dims * dstid;
              //float dist = GetDistanceByVecBam(src_vec, data, num_dims * dstid, num_dims, dist_type);

              ReadCtx ctx2;
              T* d_data = pager.get_data<T>(dstid, ctx2);
            
              cnt_fail += !PushNodeToSearchPqBam(ef_search_pq, &size, ef_search,
                                                 d_data, num_dims, src_vec, dstid);
              pager.drop(ctx2);
              cnt_read++;
            }
            __syncthreads();
            if (threadIdx.x < PREFETCH_LENGTH) {
              req_seq[threadIdx.x] = -1;
            }
          }
        }
        pager.drop(ctx1);
        __syncthreads();
        idx = GetCand(ef_search_pq, size, false);
        //cnt_acc += (idx == idx2), cnt_all++;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        acc_visited_cnt[blockIdx.x] += prefetch_cnt;
      }

      for (int j = threadIdx.x; j < visited_cnt; j += blockDim.x) {
        _visited_table[_visited_list[j]] = -1;
      }
      __syncthreads();
      // get sorted neighbors

      if (threadIdx.x == 0) {
        int size2 = size;
        while (size > 0) {
          if (size <= topk) {
            nns[i * topk + size - 1] = ef_search_pq[0].nodeid;
            distances[i * topk + size - 1] = ef_search_pq[0].distance;
          }
          //printf("%lf\n", distances[i * topk + size - 1]);
          PqPop(ef_search_pq, &size);
        }
        found_cnt[i] = size2 < topk? size2: topk;
      }

      __syncthreads();
    }
  }
#else
  template <class T>
  __inline__ __device__ void search_disk_graph_kernel_inner
  (const float* qdata, const int num_qnodes, DiskData* data,
   const int num_nodes, const int num_dims, const int max_m,
   const int ef_search, const int* entries, const int topk,
   int* nns, float* distances, int* found_cnt,
   int* visited_table, int* visited_list, 
   const int visited_table_size, const int visited_list_size, int64_t* acc_visited_cnt,
   Neighbor* neighbors, int nodes_per_page, int node_len, int data_len) {

    static __shared__ int size;
    
    Neighbor* ef_search_pq = neighbors + ef_search * blockIdx.x;

    static __shared__ int visited_cnt;
    static __shared__ int prefetch_cnt;
    int* _visited_table = visited_table + visited_table_size * blockIdx.x;
    int* _visited_list = visited_list + visited_list_size * blockIdx.x;
    PageData pager;
    pager.nodes_per_page = nodes_per_page;
    pager.node_len = node_len;
    pager.disk = data;
    pager.data_len = data_len;
    

    int cnt_acc = 0, cnt_all = 0;
    for (int i = blockIdx.x; i < num_qnodes; i += gridDim.x) {
      if (threadIdx.x == 0) {
        size = 0;
        visited_cnt = 0;
        prefetch_cnt = 0;
      }
      __syncthreads();
      int cnt_fail = 0, cnt_read = 0, cnt_rounds = 0;
      // initialize entries
      const float* src_vec = qdata + i * num_dims;
      ReadCtx ctx0;
      PushNodeToSearchPqBam(ef_search_pq, &size, ef_search, pager.get_data<T>(entries[i], ctx0),
                            num_dims, src_vec, entries[i]);
      pager.drop(ctx0);
      if (CheckVisited(_visited_table, _visited_list, visited_cnt, entries[i], 
                       visited_table_size, visited_list_size)) 
        continue;
      __syncthreads();

      int idx = GetCand(ef_search_pq, size, false);

      while (idx >= 0) {
        __syncthreads();
        if (threadIdx.x == 0) ef_search_pq[idx].checked = true;
        int entry = ef_search_pq[idx].nodeid;
        __syncthreads();

        cnt_rounds++;
        ReadCtx ctx1;

        GraphData<T> graph = pager.get_graph<T>(entry, ctx1);

        for (int j = 0, tot = 0, end = graph.deg;
             j < end; j++, tot++) {
          int dstid = graph[j];

          if (CheckVisited(_visited_table, _visited_list, visited_cnt, dstid, 
                           visited_table_size, visited_list_size)) {
            continue;
          }
          __syncthreads();
          ReadCtx ctx2;
          T* d_data = pager.get_data<T>(dstid, ctx2);
          
          cnt_fail += !PushNodeToSearchPqBam(ef_search_pq, &size, ef_search,
                                             d_data, num_dims, src_vec, dstid);
          pager.drop(ctx2);
          cnt_read++;

          __syncthreads();
        }
        pager.drop(ctx1);
        __syncthreads();
        idx = GetCand(ef_search_pq, size, false);
        //cnt_acc += (idx == idx2), cnt_all++;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        acc_visited_cnt[blockIdx.x] += prefetch_cnt;
      }

      for (int j = threadIdx.x; j < visited_cnt; j += blockDim.x) {
        _visited_table[_visited_list[j]] = -1;
      }
      __syncthreads();
      // get sorted neighbors

      if (threadIdx.x == 0) {
        int size2 = size;
        while (size > 0) {
          if (size <= topk) {
            nns[i * topk + size - 1] = ef_search_pq[0].nodeid;
            distances[i * topk + size - 1] = ef_search_pq[0].distance;
          }
          //printf("%lf\n", distances[i * topk + size - 1]);
          PqPop(ef_search_pq, &size);
        }
        found_cnt[i] = size2 < topk? size2: topk;
      }

      __syncthreads();
    }
  }
#endif
  
  __global__ void search_disk_graph_kernel
  (float* qdata, const int num_qnodes,
   DiskData* data,   
   const int num_nodes, const int num_dims, const int max_m,
   const int ef_search, const int* entries, const int topk,
   int* nns, float* distances, int* found_cnt,
   int* visited_table, int* visited_list, 
   const int visited_table_size, const int visited_list_size, int64_t* acc_visited_cnt,
   Neighbor* neighbors, int nodes_per_page, int node_len, int data_len, DataType data_type,
   PQSearchData* pq) {
    search_disk_graph_kernel_inner_with_pq<uint8_t>
      (qdata, num_qnodes, data, num_nodes, num_dims, max_m, ef_search, entries, topk,
       nns, distances, found_cnt, visited_table, visited_list,
       visited_table_size, visited_list_size, acc_visited_cnt,
       neighbors, nodes_per_page, node_len, data_len, pq
      );
  }
}

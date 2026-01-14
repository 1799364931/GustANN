#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <string>
#include <map>

#include <sys/time.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

//#include "ssd_io.hpp"
#include "hybrid.hpp"

#include "common.hpp"
#include "common_cuda.cuh"

#include "nav_graph.hpp"
#include "ssd_search.hpp"
//#include "ssd_search_kernel.hpp"

#include "io/interface.hpp"

#include "impl/hyd.cuh"
#include "impl/util.cuh"
#include "impl/nav.cuh"
#include "impl/pq.cuh"

#include "impl/hyd_search_v1.cuh"
#include "impl/hyd_search_v2.cuh"

#define REPORT(fmt, ...) printf("[REPORT] " fmt "\n", __VA_ARGS__)

#define NAV_GRAPH
//#define LOAD_PROBE
//#define COPY_DATA
//#define CACHE_START

namespace gustann {

  __global__ void copy_page(uint8_t* dest, uint8_t* src, int32_t* request, const int data_len, const int data_cnt) {
    int tid = threadIdx.x;
    assert(data_len % 4 == 0);
    

    int x = request[blockIdx.x];
    if (x == -1) return;
    uint32_t* local_dest = (uint32_t*)(dest + blockIdx.x * PAGE_SIZE);
    uint32_t* local_src = (uint32_t*)(src + blockIdx.x * PAGE_SIZE);
    int len = data_len / 4;
    int offset = x % data_cnt * len;
    for (int i = tid; i < len; i += blockDim.x) {
      local_dest[i + offset] = local_src[i + offset];
    }
  }

  
  struct TaskRunner {

    float* d_qdata; // (mini_batch * num_dims_);
    int* d_nns; //(mini_batch * topk);
    float* d_distances; //(mini_batch * topk);
    int* d_found_cnt; //(mini_batch, 0);
    
    thrust::device_vector<float> d_tmp_dist; //(mini_batch * max_m0_);
    thrust::device_vector<int> d_tmp_id; //(mini_batch * max_m0_);
    thrust::device_vector<int64_t> d_acc_visited_cnt; //(mini_batch, 0);
    thrust::device_vector<uint32_t> d_neighbors_id; //(aligned_ef * mini_batch);
    thrust::device_vector<float> d_neighbors_dist; //(aligned_ef * mini_batch);
    thrust::device_vector<Data> d_ctx; //(mini_batch, (Data){0, 0});
    
    uint8_t* buffer;
    int32_t* request;

    cudaStream_t stream;
    double t0;

    int mini_batch, num_dims, topk, max_m0, ef_search, aligned_ef;
    int enter_point;
    int num_reads = 0;
    int qcnt;
    int nodes_per_page_, node_size_, data_size_;
    int stream_offset;
    int cid;
    int tid;
    int *nns;
    float *distances;
    int *found_cnt;
    int tcnt;
    int64_t num_data;
    PQSearch* pq;
#ifdef COPY_DATA
    uint8_t* buffer_dev;
    int32_t* request_dev;
#endif
    std::shared_ptr<IndexLoader> loader;
    
    enum {
      Q_INIT,
      Q_GPU,
      Q_SSD,
      Q_RES,
      Q_FIN,
    } state;
    uint8_t* starter;
    double time_gpu;
    double time_ssd;
    double latency;
    int cnt_query;

    double time_init_issue;
    double time_gpu_issue;
    double time_ssd_issue;
    double time_fin_issue;

    unsigned int seed;
    int round;

    int* start_pt;
    NavGraph* nav_graph;

    /// GGG
    uint8_t* test_pool;
#ifdef LOAD_PROBE
    std::vector<std::vector<double>> ssd_overall;
    static constexpr int sample_ssd = 6;
    std::map<std::pair<int, int>, int> freq;
#endif    
    
    void init_query(const float* qdata, int _qcnt, int* _nns, float* _dis, int *_found_cnt, int *_start_pt) {
      time_init_issue -= elapsed();
      latency -= elapsed();
      cnt_query++;
      qcnt = _qcnt;
      nns = _nns;
      distances = _dis;
      found_cnt = _found_cnt;

      round = 0;
      start_pt = _start_pt;

      
      //thrust::fill(d_found_cnt.begin(), d_found_cnt.end(), 0);
      //thrust::copy(qdata, qdata + qcnt * num_dims, d_qdata.begin());
      //thrust::fill()
      CHECK_CUDA(cudaStreamSynchronize(stream));
      time_gpu -= elapsed();
      CHECK_CUDA(cudaMemsetAsync(d_found_cnt,
                                 0, sizeof(int) * qcnt, stream));
      CHECK_CUDA(cudaMemcpyAsync(d_qdata,
                                 qdata, sizeof(float) * qcnt * num_dims,
                                 cudaMemcpyHostToDevice, stream));
      init_search<<<qcnt, 64, 0, stream>>>
        (d_qdata,
         pq->get_device_ptr(),
         stream_offset, num_dims);
#ifdef NAV_GRAPH
      int init_ef = std::min(ef_search, 5);
      int dim = nav_graph->data_len;

      get_entry_kernel<<<(qcnt + 1) / 2, 64, 0, stream>>>
        (d_qdata, nav_graph->data_dev, nav_graph->graph_dev, qcnt,
         nav_graph->num_node, dim, nav_graph->max_m,
         init_ef, nav_graph->start, request,
         thrust::raw_pointer_cast(d_neighbors_id.data()),
         thrust::raw_pointer_cast(d_neighbors_dist.data())
         );
#else

      for (int j = 0; j < qcnt; j++) {
#ifdef CACHE_START
        request[j] = start_pt[j] == -1 ? enter_point : start_pt[j];
#else
        //read_page(enter_point_, local_buf + PAGE_SIZE * j);
        memcpy(buffer + PAGE_SIZE * j, starter, PAGE_SIZE);
        //CHECK_CUDA(cudaMemcpyAsync(buffer + PAGE_SIZE * j, starter,
        //                           PAGE_SIZE, cudaMemcpyHostToHost,
        //                           stream));
        request[j] = enter_point;
#endif
      }
#endif
      state = Q_INIT;
      time_init_issue += elapsed();
    //submit_gpu();
    }

    void finish_query() {
      time_fin_issue -= elapsed();
#if 1
      //std::vector<int64_t> acc_visited_cnt(mini_batch);
      //thrust::copy(d_acc_visited_cnt.begin(), d_acc_visited_cnt.end(), acc_visited_cnt.begin());
      //thrust::copy(d_nns.begin(), d_nns.begin() + qcnt * topk, nns);
      //thrust::copy(d_distances.begin(), d_distances.begin() + qcnt * topk, distances);
      //thrust::copy(d_found_cnt.begin(), d_found_cnt.begin() + qcnt, found_cnt);
      CHECK_CUDA(cudaMemcpyAsync(nns, d_nns,
                                 qcnt * topk * sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA(cudaMemcpyAsync(distances, d_distances,
                                 qcnt * topk * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA(cudaMemcpyAsync(found_cnt, d_found_cnt,
                                 qcnt * sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
#endif
      //CHECK_CUDA(cudaStreamSynchronize(stream));
      state = Q_RES;
      time_fin_issue += elapsed();
    }
    static const int rept = 100;
    void submit_gpu() {
      time_gpu_issue -= elapsed();
      CHECK_CUDA(cudaStreamSynchronize(stream));
      time_gpu -= elapsed();
      /*
      // GGG

      double start = elapsed();
      for (int i = 0; i < rept; i++) {
        uint8_t* buffer = test_pool + 1ll * i * PAGE_SIZE * mini_batch;
        merge_data_kernel<<<qcnt, tcnt,
          ((sizeof(int) * 3 + sizeof(float) * 2) * (ef_search + max_m0)
           + ((pq->get_data().num_chunks + 1) * max_m0)
           ), stream>>>
          (buffer, request,
           //pq->get_device_ptr(),
           pq->device_data.num_chunks, pq->device_data.pq_dists, pq->device_data.compressed_data,
           nodes_per_page_, node_size_, data_size_,
           stream_offset, max_m0, ef_search,
           thrust::raw_pointer_cast(d_neighbors_id.data()),
           thrust::raw_pointer_cast(d_neighbors_dist.data()),
           thrust::raw_pointer_cast(d_ctx.data()));
        

        unify_kernel<<<(qcnt + 1 / 2), 64, 0, stream>>>
          (d_qdata, buffer, request,
           num_dims, max_m0, ef_search, topk,
           d_nns, d_distances, d_found_cnt,
           thrust::raw_pointer_cast(d_neighbors_id.data()),
           thrust::raw_pointer_cast(d_neighbors_dist.data()),
           nodes_per_page_, node_size_, data_size_,
           thrust::raw_pointer_cast(d_ctx.data()), qcnt
           );

      }
      CHECK_CUDA(cudaStreamSynchronize(stream));
      double end = elapsed();
      printf("Time: %lf\n", end - start);
      state = Q_FIN;
      return;

      /// END GGG XXX
      */
#if 0
      // VERSION A
      get_pq_dist_kernel<<<qcnt, tcnt, 0, stream>>>
        (buffer, request,
         thrust::raw_pointer_cast(d_tmp_dist.data()),
         thrust::raw_pointer_cast(d_tmp_id.data()),
         pq->get_device_ptr(),
         nodes_per_page_, node_size_, data_size_,
         stream_offset, max_m0
         );
      //DEBUG("XXXX {}", i);
      update_kernel<<<(qcnt + 1) / 2, 64, 0, stream>>>
        (d_qdata,
         buffer, request,
         thrust::raw_pointer_cast(d_tmp_dist.data()),
         thrust::raw_pointer_cast(d_tmp_id.data()),
         num_data, num_dims, max_m0, ef_search, 
         topk,
         d_nns,
         d_distances,
         d_found_cnt,
         thrust::raw_pointer_cast(d_acc_visited_cnt.data()),
         thrust::raw_pointer_cast(d_neighbors_id.data()),
         thrust::raw_pointer_cast(d_neighbors_dist.data()),
         nodes_per_page_, node_size_, data_size_,
         thrust::raw_pointer_cast(d_ctx.data()), qcnt
         );
#else
#ifdef COPY_DATA
      
      //CHECK_CUDA(cudaMemcpyAsync(buffer_dev, buffer, PAGE_SIZE * qcnt, cudaMemcpyDefault, stream));
      CHECK_CUDA(cudaMemcpyAsync(request_dev, request, sizeof(int32_t) * qcnt, cudaMemcpyDefault, stream));
      /*
      for (int i = 0; i < qcnt; i++) {
        if (request[i] != -1) {
          CHECK_CUDA(cudaMemcpyAsync(buffer_dev + PAGE_SIZE * i, buffer + PAGE_SIZE * i, PAGE_SIZE, cudaMemcpyDefault, stream));
        }
      }p
      */
      copy_page<<<qcnt, 32, 0, stream>>>(buffer_dev, buffer, request_dev, node_size_, nodes_per_page_);
      uint8_t* buffer_t = buffer_dev; // Overwriting `buffer` below
      int32_t* request_t = request_dev;
#else
      uint8_t* buffer_t = buffer;
      int32_t* request_t = request;

#endif
      

      //printf("!!!!!\n");
      merge_data_kernel<<<qcnt, tcnt,
        ((sizeof(int) * 3 + sizeof(float) * 2) * (ef_search + max_m0)
         //+ ((pq->get_data().num_chunks + 1) * max_m0)
         ), stream>>>
        (buffer_t, request_t,
         pq->device_data.num_chunks, pq->device_data.pq_dists, pq->device_data.compressed_data,
         nodes_per_page_, node_size_, data_size_,
         stream_offset, max_m0, ef_search,
         thrust::raw_pointer_cast(d_neighbors_id.data()),
         thrust::raw_pointer_cast(d_neighbors_dist.data()),
         thrust::raw_pointer_cast(d_ctx.data()));

      unify_kernel<<<(qcnt + 1 / 2), 64, 0, stream>>>
        (d_qdata, buffer_t, request_t,
         num_dims, max_m0, ef_search, topk,
         d_nns, d_distances, d_found_cnt,
         thrust::raw_pointer_cast(d_neighbors_id.data()),
         thrust::raw_pointer_cast(d_neighbors_dist.data()),
         nodes_per_page_, node_size_, data_size_,
         thrust::raw_pointer_cast(d_ctx.data()), qcnt
         );

#endif

#ifdef COPY_DATA
      CHECK_CUDA(cudaMemcpyAsync(request, request_dev, sizeof(int32_t) * qcnt, cudaMemcpyDefault, stream));
#endif
      state = Q_GPU;
      time_gpu_issue += elapsed();
    }

    void submit_ssd() {
      time_ssd_issue -= elapsed();
      CHECK_CUDA(cudaStreamSynchronize(stream));
      //DEBUG("YYYY {}", i);
      bool finished = true;
      time_ssd -= elapsed();
      
      /*
        for (int j = 0; j < qcnt; j++) {
        if (local_req[j] != -1) {
        finished = false;
        read_page(local_req[j], local_buf + PAGE_SIZE * j);
        num_reads++;
        }
        }
      */
#ifdef LOAD_PROBE
      std::vector<int> ssd_cnt(sample_ssd);
#endif
      std::vector<std::pair<int, void*>> pages;
      for (int j = 0; j < qcnt; j++) {
        //request[j] = rand_r(&seed) % num_data; /// !!! REMOVE !!! ///
        if (request[j] != -1) {
          //printf("%d\n", request[j]);
          if (!(request[j] >= 0 && request[j] < num_data)) {
            fprintf(stderr, "??? %d\n", request[j]);
            throw;
          }
          //printf("%d %d ", j, request[j]);
          finished = false;
          int blockid = request[j] / nodes_per_page_;
          pages.emplace_back(blockid, buffer + PAGE_SIZE * j);
          num_reads++;
#ifdef LOAD_PROBE
          int ssd = blockid % sample_ssd;
          ssd_cnt[ssd]++;
          freq[{round, request[j]}]++;
#endif
        }
      }
      //printf("\n");
      //printf("F %d\n", finished);
      //printf("%d\n", qcnt);
      //uring.read_pages(threadid, pages);

      loader->submit_task(pages, tid, cid);

#ifdef LOAD_PROBE
      std::sort(ssd_cnt.begin(), ssd_cnt.end());
      int sum = std::accumulate(ssd_cnt.begin(), ssd_cnt.end(), 0);
      if (round == ssd_overall.size()) {
        ssd_overall.push_back(std::vector<double>(sample_ssd));
      }
      if (sum != 0) {
        for (int i = 0; i < sample_ssd; i++) {
          ssd_overall[round][i] += 1. * ssd_cnt[i] / sum;
          //printf("%lf %d %d\n", ssd_overall[round][i], ssd_cnt[i], sum);
        }
      }
#endif

      if (++round == 6) {       
        for (int j = 0; j < qcnt; j++) {
          if (start_pt[j] == -1) start_pt[j] = request[j];
        }
        //finished = true; /// !!! REMOVE !!! 
      }
      //if (round == 35) {
      //finished = true;
      //}
      /*
      if (round > 1000) {
        for (int i = 0; i < qcnt; i++) {
          if (request[i] != -1) {
            printf("%d %d %d\n", request[i], i, cid);
            int offset = (max_m0 + ef_search + 31) / 32 * 32 * i;
            thrust::host_vector<uint32_t> v = d_neighbors_id;
            thrust::host_vector<float> t = d_neighbors_dist;
            for (int i = 0; i < max_m0 + ef_search; i++) {
              printf("%d %d %f\n", v[offset + i] & 0x7fffffff, v[offset + i] >> 31, t[offset + i]);
            }
            if (round > 1002) throw;
          }
        }
        
        printf("\n");         
      }
      {
        thrust::host_vector<uint32_t> v = d_neighbors_id;
        thrust::host_vector<float> t = d_neighbors_dist;
        thrust::host_vector<Data> g = d_ctx;
        for (int i = 0; i < qcnt; i++) {
          int offset = (max_m0 + ef_search + 31) / 32 * 32 * i;
          if (request[i] != -1) {
            for (int j = 1; j < g[i].size; j++) {
              if (!(t[offset + j - 1] <= t[offset + j])) {
                printf("%d %d\n", i, cid);
                throw;
              }
            }
          }
        }
      }
      */
      if (finished) {
        time_ssd += elapsed();
        finish_query();
        //printf("FIN\n");
      } else {
        state = Q_SSD;
      }
      time_ssd_issue += elapsed();
    }

    bool update_state() {
      switch(state) {
      case Q_INIT:
      case Q_GPU:
      case Q_RES: {
        auto err = cudaStreamQuery(stream);
        if (err == cudaSuccess) {
          if (state == Q_INIT) {
            time_gpu += elapsed();
#ifdef NAV_GRAPH
            nav_graph->translate(request, qcnt);
            submit_ssd();
#else
#ifdef CACHE_START
            submit_ssd();
#else
            //printf("!!!! INIT->GPU\n");
            submit_gpu();
#endif
#endif
          } else if (state == Q_GPU) {
            //printf("!!!! GPU->SSD\n");
            time_gpu += elapsed();
            submit_ssd();
          } else {
            //printf("!!!!!\n");
            latency += elapsed();
            state = Q_FIN;
            return true;
          }
        } else if (err == cudaErrorNotReady) {
          //printf("Wait!\n");
        } else {
          CHECK_CUDA(err);
        }
        break;
      }
      case Q_SSD: {
        bool ready = loader->poll_task(cid);
        if (ready) {

          //printf("!!!! SSD->GPU \n");
          time_ssd += elapsed();
          submit_gpu();
        }
        break;
      }
      case Q_FIN: {
        return true;
      }
      }
      return false;
    }
    
    TaskRunner(int _tid, int _cid, int _mini_batch, int _num_dims,
               int _topk, int64_t _num_data, int _max_m0, int _ef_search,
               int _enter_point, uint8_t* _starter, PQSearch *_pq,
               int nodes_per_page, int node_size, int data_size,
               std::shared_ptr<IndexLoader> _loader,
               NavGraph* _nav_graph) {

      tid = _tid;
      cid = _cid;
      mini_batch = _mini_batch;
      stream_offset = cid * mini_batch;
      
      num_dims = _num_dims;
      topk = _topk;
      num_data = _num_data;
      max_m0 = _max_m0;
      ef_search = _ef_search;
      aligned_ef = (ef_search + max_m0 + 31) / 32 * 32; // MODIFIED IN VERSION B
      starter = _starter;
      enter_point = _enter_point;
      pq = _pq;

      nodes_per_page_ = nodes_per_page;
      node_size_ = node_size;
      data_size_ = data_size;
      loader = _loader;

      nav_graph = _nav_graph;

      tcnt = (max_m0 + 31) / 32 * 32;

      state = Q_FIN;

      num_reads = 0;
      time_gpu = 0;
      time_ssd = 0;
      latency = 0;
      cnt_query = 0;

      seed = tid * 100000 + cid;
      
      CHECK_CUDA(cudaStreamCreate(&stream));
#ifdef MEM_PROFILE
      size_t free_mem, tot_mem;
      CHECK_CUDA(cudaMemGetInfo(&free_mem, &tot_mem));
      printf("Now %lf/%lf B free mem\n",
             1.0 * free_mem, 1.0 * tot_mem );
#endif
      //CHECK_CUDA(cudaMallocHost(&buffer, sizeof(uint8_t) * PAGE_SIZE * mini_batch));


      buffer = loader->create_buffer((int64_t)PAGE_SIZE * mini_batch);
      
      CHECK_CUDA(cudaHostRegister(buffer, sizeof(uint8_t) * PAGE_SIZE * mini_batch, cudaHostRegisterDefault));
#ifdef COPY_DATA
      CHECK_CUDA(cudaMalloc(&buffer_dev, PAGE_SIZE * mini_batch));
      CHECK_CUDA(cudaMalloc(&request_dev, sizeof(int32_t) * mini_batch));
#else
      
#endif

      CHECK_CUDA(cudaMallocHost(&request, sizeof(int32_t) * mini_batch));
      CHECK_CUDA(cudaMalloc(&d_qdata, sizeof(float) * mini_batch * num_dims));
      CHECK_CUDA(cudaMalloc(&d_nns, sizeof(int) * mini_batch * topk));
      CHECK_CUDA(cudaMalloc(&d_distances, sizeof(float) * mini_batch * topk));
      CHECK_CUDA(cudaMalloc(&d_found_cnt, sizeof(int) * mini_batch));
      d_tmp_dist.resize(mini_batch * max_m0);
      d_tmp_id.resize(mini_batch * max_m0);
      d_acc_visited_cnt.resize(mini_batch, 0);
      d_neighbors_id.resize(aligned_ef * mini_batch);
      d_neighbors_dist.resize(aligned_ef * mini_batch);
      d_ctx.resize(mini_batch, (Data){0, 0});
#ifdef MEM_PROFILE
      CHECK_CUDA(cudaMemGetInfo(&free_mem, &tot_mem));
      printf("Now %lf/%lf B free mem\n",
             1.0 * free_mem, 1.0 * tot_mem );
#endif
      /*
      // GGG
      CHECK_CUDA(cudaMallocHost(&test_pool, 1l * PAGE_SIZE * mini_batch * rept));
      for (int i = 0; i < rept * mini_batch; i++) {
        for (int j = 0; j < nodes_per_page_; j++) {
          uint8_t* start = test_pool + 1l * i * PAGE_SIZE + j * node_size_;
          for (int k = 0; k < data_size_; k++) {
            start[k] = rand_r(&seed) % 128;
          }
          int* edge = (int*)(start + data_size_);
          edge[0] = max_m0;
          edge++;
          for (int k = 0; k < max_m0; k++) {
            edge[k] = rand_r(&seed) % num_data;
          }
        }
      }
      // END GGG
      */
      
      //printf("Ready %d!!!\n", seed);
    }
  };


  HybridExecutor::HybridExecutor(const Layout &layout,
                                 const DataType &data_type,
                                 const std::string &fpath,
                                 const HybridExecutorConfig &config)
    : layout_(layout), data_type_(data_type) {
    mini_batch_ = config.mini_batch;
    thread_cnt_ = config.thread_cnt;
    ctx_per_thread_ = config.ctx_per_thread;
    
    FILE *input = fopen(fpath.c_str(), "rb");
    if (!input) {
      ERROR("Index File Not found!");
      exit(-1);
    }
    

    starter_ = new uint8_t[PAGE_SIZE];
    fseek(input, (long) PAGE_SIZE * (layout_.enter_point / layout_.nodes_per_page + 1), SEEK_SET);
    fread((char *)starter_, sizeof(char), PAGE_SIZE, input);
    

    fclose(input);


    if (config.use_backend == HybridExecutorConfig::SPDK) {
      const auto& ssds = config.ssd_lists;
      if (ssds.empty()) {
        ERROR("NO SSD IN USE!");
        exit(-1);
      }
      loader_ = create_spdk_loader(ssds, mini_batch_ * ctx_per_thread_,
                                   thread_cnt_, thread_cnt_ * ctx_per_thread_);
    } else if (config.use_backend == HybridExecutorConfig::MEMORY) {
      loader_ = create_mem_loader_sync(fpath.c_str(), layout_.num_pages);
    } else {
      ERROR("Wrong IO Backend setting!");
      exit(-1);
    }

  }

  void HybridExecutor::search(const float *qdata, int num_queries, int topk,
                             int ef_search, int *nns, float *distances, int *found_cnt,
                             const Config& config, PQSearch* pq
                             ) {
    int batch_cnt = mini_batch_ * thread_cnt_ * ctx_per_thread_;
    //num_queries = 1;
    CHECK_CUDA(cudaHostRegister((void*)qdata, sizeof(float) * num_queries * layout_.num_dims, cudaHostRegisterDefault));
    
    if (pq) pq->init_device(layout_.num_dims, layout_.num_data, batch_cnt, ef_search);
    
    std::atomic<int> tot_reads(0);
    
    NavGraph nav_graph;
#ifdef NAV_GRAPH
    const std::string nav_index_file = config.nav_data + "/" + "nav_index";
    const std::string nav_data_file = config.nav_data + "/"+ "nav_index.data";
    const std::string nav_map_file = config.nav_data + "/" + "map.txt";

    INFO("Use small navigation graph!");
    nav_graph.init(nav_index_file, nav_data_file, nav_map_file);
#endif

    int* start_pts = new int [num_queries];
    memset(start_pts, -1, sizeof(int) * num_queries);

    std::atomic<int> cur(0);
    //cur = 56521;
    auto worker = [&](int threadid) {
      bind_core(threadid * 2 + 21);

      bool finished = false;
      int tot_task = 0;
      std::vector<TaskRunner> tasks;
      for (int i = 0; i < ctx_per_thread_; i++) {
        tasks.emplace_back(threadid, threadid * ctx_per_thread_ + i,
                           mini_batch_, (int)layout_.num_dims, topk,
                           layout_.num_data, (int)layout_.max_m0, ef_search,
                           (int)layout_.enter_point, starter_, pq,
                           (int)layout_.nodes_per_page, (int)layout_.node_size,
                           (int)layout_.data_size, loader_, &nav_graph);       
      }
      double t0 = elapsed();
      INFO("Thread {} started", threadid);

      while(!finished) {
        finished = true;
        for (auto& task: tasks) {
          if (task.update_state()) {
            if (cur.load() < num_queries) {
              int qstart = cur.fetch_add(mini_batch_);             
              int qend = std::min(qstart + mini_batch_, num_queries);
              int qcnt = qend - qstart;
              //printf("!!! %d %d\n", qstart, qcnt);
              if (qstart < num_queries) {
#ifdef CACHE_START
                for (int i = qstart; i < qend; i++) {
                  if (start_pts[i % 10000] != -1) {
                    start_pts[i] = start_pts[i % 10000];
                  }
                }
#endif
                task.init_query(qdata + (int64_t) qstart * layout_.num_dims, qcnt,
                                nns + qstart * topk,
                                distances + qstart * topk,
                                found_cnt + qstart,
                                start_pts + qstart);
                finished = false;
              }
              tot_task += qcnt;
            }
          } else {
            finished = false;
          }
        }
      }

      double tot_gpu = 0;
      double tot_ssd = 0;

      double tot_init_issue = 0;
      double tot_gpu_issue = 0;
      double tot_ssd_issue = 0;
      double tot_fin_issue = 0;

      double tot_lat = 0;
      int cnt_batch = 0;
#ifdef LOAD_PROBE
      std::vector<std::vector<double>> round_probe;
      std::map<std::pair<int, int>, int> tot_freq;
#endif
      for (auto& task: tasks) {
        tot_reads.fetch_add(task.num_reads);
        tot_gpu += task.time_gpu;
        tot_ssd += task.time_ssd;

        tot_init_issue += task.time_init_issue;
        tot_gpu_issue += task.time_gpu_issue;
        tot_ssd_issue += task.time_ssd_issue;
        tot_fin_issue += task.time_fin_issue;
        tot_lat += task.latency;
        cnt_batch += task.cnt_query;
#ifdef LOAD_PROBE
        for (int i = 0; i < (int) task.ssd_overall.size(); i++) {
          if (i == (int) round_probe.size()) {
            round_probe.push_back(std::vector<double>(task.sample_ssd));
          }
          for (int j = 0; j < task.sample_ssd; j++) {
            round_probe[i][j] += task.ssd_overall[i][j];
            //printf("%lf\n", task.ssd_overall[i][j]);
          }
        }
        for (auto x: task.freq) {
          tot_freq[x.first] += x.second;
        }
#endif
      }
      double t1 = elapsed();
      INFO("Thread {}: {} / {} s, {} queries in {} s, {} qps",
           threadid, tot_gpu, tot_ssd, tot_task, t1 - t0, tot_task / (t1 - t0));
      INFO("Thread {}: Init {}, GPU {}, SSD {}, FIN {}",
           threadid, tot_init_issue, tot_gpu_issue, tot_ssd_issue, tot_fin_issue);
      INFO("Thread {}: Latency {} ms",
           threadid, tot_lat / cnt_batch * 1000);
      REPORT("LAT%d %lf", threadid, tot_lat / cnt_batch * 1000);
#ifdef LOAD_PROBE
#if 1
      int tot_r = round_probe.size();
      for (int i = 0; i < tot_r; i++) {
        printf("%lf ", round_probe[i].front() / cnt_batch);
      }
      printf("\n");
      for (int i = 0; i < tot_r; i++) {
        printf("%lf ", round_probe[i].back() / cnt_batch);
      }
      printf("\n");
#endif
#if 1
      std::map<int, std::vector<int>> g;
      for (auto x: tot_freq) {
        g[x.first.first].push_back(x.second);
      }
      int y = 0;
      for (auto &x: g) {
        auto &v = x.second;
        std::sort(v.begin(), v.end());
        std::reverse(v.begin(), v.end());
        printf("%d\t%d\t%lu\n", v[0], v[(v.size() - 1) / 2], v.size());
        if (y++ < 3) {
          const int G = 50;
          for (int i = 0; i < G; i++) {
            printf("%d ", v[i * v.size() / G]);
          }
          
          printf("%d \n", v.back());
          const int X = 500;
          const int Y = 50;
          for (int i = 0; i <= Y; i++) {
            printf("%d ", v[i * v.size() / X]);
          }
          printf("%d\n", v.back());
        }        

      }
#endif
#endif
    };

    std::vector<std::thread> th;
    CHECK_CUDA(cudaDeviceSynchronize());
    double start = elapsed();
    for (int i = 0; i < thread_cnt_; i++) {
      th.emplace_back(worker, i);
    }

    for (int i = 0; i < thread_cnt_; i++) {
      th[i].join();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double end = elapsed();
    DEBUG("End Search");
    INFO("Use time: {}", end - start);
    INFO("Total reads: {}", tot_reads.load());
    REPORT("Time %lf", end - start);
    REPORT("IO %d", tot_reads.load());
    CHECK_CUDA(cudaDeviceSynchronize());

  }
}

#pragma once

#include <cstdint>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include "common.hpp"
#include "common_cuda.cuh"
#include "io/interface.hpp"

#include "nav_graph.hpp"
#include "impl/hyd.cuh"

//#define LOAD_PROBE
//#define COPY_DATA
//#define CACHE_START


namespace gustann {
  
  class TaskRunner {

  public:
    void init_query(const float* qdata, int _qcnt, int* _nns, float* _dis, int *_found_cnt, int *_start_pt);
    void finish_query();
    void submit_gpu();
    void submit_ssd();
    bool update_state();

    TaskRunner(int _tid, int _cid, int _mini_batch, int _num_dims, int _topk,
               int64_t _num_data, int _max_m0, int _ef_search, int _enter_point,
               uint8_t *_starter, PQSearch *_pq, int nodes_per_page,
               int node_size, int data_size, DataType data_type,
               std::shared_ptr<IndexLoader> _loader, NavGraph *_nav);
    ~TaskRunner();
    TaskRunner(const TaskRunner&) = delete;
    TaskRunner& operator=(const TaskRunner&) = delete;
    TaskRunner(TaskRunner&&) = delete;
    TaskRunner& operator=(TaskRunner&&) = delete;

  public:
    double time_gpu;
    double time_ssd;
    double latency;
    int cnt_query;
    int num_reads = 0;
    double time_init_issue;
    double time_gpu_issue;
    double time_ssd_issue;
    double time_fin_issue;

  private:
    float* d_qdata = nullptr; // (mini_batch * num_dims_);
    int* d_nns = nullptr; //(mini_batch * topk);
    float* d_distances = nullptr; //(mini_batch * topk);
    int* d_found_cnt = nullptr; //(mini_batch, 0);
    
    thrust::device_vector<float> d_tmp_dist; //(mini_batch * max_m0_);
    thrust::device_vector<int> d_tmp_id; //(mini_batch * max_m0_);
    thrust::device_vector<int64_t> d_acc_visited_cnt; //(mini_batch, 0);
    thrust::device_vector<uint32_t> d_neighbors_id; //(aligned_ef * mini_batch);
    thrust::device_vector<float> d_neighbors_dist; //(aligned_ef * mini_batch);
    thrust::device_vector<Data> d_ctx; //(mini_batch, (Data){0, 0});

    DataType data_type;
      
    uint8_t* buffer = nullptr;
    int32_t* request = nullptr;

    cudaStream_t stream = nullptr;
    double t0;

    int mini_batch, num_dims, topk, max_m0, ef_search, aligned_ef;
    int enter_point;

    int qcnt;
    int nodes_per_page_, node_size_, data_size_;
    int stream_offset;
    int cid;
    int tid;
    int *nns = nullptr;
    float *distances = nullptr;
    int *found_cnt = nullptr;
    int tcnt;
    int64_t num_data;
    PQSearch* pq = nullptr;
#ifdef COPY_DATA
    uint8_t* buffer_dev = nullptr;
    int32_t* request_dev = nullptr;
#endif
    std::shared_ptr<IndexLoader> loader;
    
    enum {
      Q_INIT,
      Q_GPU,
      Q_SSD,
      Q_RES,
      Q_FIN,
    } state;
    uint8_t *starter = nullptr;
    unsigned int seed;
    int round;

    int* start_pt = nullptr;
    NavGraph* nav_graph = nullptr;

    /// GGG
    uint8_t* test_pool = nullptr;
#ifdef LOAD_PROBE
    std::vector<std::vector<double>> ssd_overall;
    static constexpr int sample_ssd = 6;
    std::map<std::pair<int, int>, int> freq;
#endif
    template <class T> void gpu_search_v1();
    template <class T> void gpu_search_v2();

  };
}

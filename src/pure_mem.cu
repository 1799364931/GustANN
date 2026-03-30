#include <algorithm>
#include <numeric>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>


#include "common.hpp"
#include "common_cuda.cuh"
#include "nav_graph.hpp"
#include "pure_mem.hpp"
//#include "ssd_search_kernel.hpp"

#include "impl/pure_mem.cuh"
#include "impl/nav.cuh"



namespace gustann {

  PureMemExecutor::PureMemExecutor(const std::string &fpath, const Layout &layout,
                           const DataType& data_type,
                           bool use_gpu_mem)
    : layout_(layout) {
    block_cnt_ = 112 * 100;
    //block_cnt_ = 1;
    block_dim_ = 32;
    data_type_ = data_type;
    use_gpu_mem_ = use_gpu_mem;
    read_to_mem(fpath);
  }

  PureMemExecutor::~PureMemExecutor() {
    CHECK_CUDA(cudaFreeHost(mem_data_host_));
    if (use_gpu_mem_) {
      CHECK_CUDA(cudaFree(mem_data_dev_));
    }
  }
  void PureMemExecutor::read_to_mem(const std::string &fpath) {
    FILE* input = fopen(fpath.c_str(), "rb");
    if (!input) {
      ERROR("Failed to open file: {}", fpath);
      exit(-1);
    }
    fseek(input, PAGE_SIZE, SEEK_SET);
    CHECK_CUDA(cudaHostAlloc(&mem_data_host_, layout_.num_pages * PAGE_SIZE, cudaHostAllocPortable));

    ASSERT(fread(mem_data_host_, 1, layout_.num_pages * PAGE_SIZE, input) ==
           layout_.num_pages * PAGE_SIZE);
    if (use_gpu_mem_) {
      INFO("USE GPU MEMORY");
      CHECK_CUDA(cudaMalloc(&mem_data_dev_, layout_.num_pages * PAGE_SIZE));
      CHECK_CUDA(cudaMemcpy(mem_data_dev_, mem_data_host_,
                            layout_.num_pages * PAGE_SIZE,
                            cudaMemcpyHostToDevice));
      mem_data_ = mem_data_dev_;
    } else {
      INFO("USE DRAM {}", (uint64_t)mem_data_host_);
      mem_data_ = mem_data_host_;
    }
    INFO("Data loaded to CPU");
    fclose(input);
  }

  void PureMemExecutor::search(const float *qdata, const int num_queries_,
                           const int topk, const int ef_search, int *nns,
                           float *distances, int *found_cnt,
                           PQSearch *pq, NavGraph *nav_graph) {
    int num_queries = num_queries_;
    
    thrust::device_vector<float> d_qdata(num_queries * layout_.num_dims);
    thrust::copy(qdata, qdata + num_queries * layout_.num_dims, d_qdata.begin());

    int aligned_ef = (ef_search + layout_.max_m0 + 31) / 32 * 32;
    int block_cnt = block_cnt_;//num_queries_;

    thrust::device_vector<int> d_entries(num_queries);
    thrust::device_vector<int> d_nns(num_queries * topk);
    thrust::device_vector<float> d_distances(num_queries * topk);
    thrust::device_vector<int> d_found_cnt(num_queries);
    thrust::device_vector<uint32_t> d_neighbors_id(aligned_ef * block_cnt);
    thrust::device_vector<float> d_neighbors_dist(aligned_ef * block_cnt);


    DEBUG("Start Search");

    if (pq) {
      pq->init_device(layout_.num_dims, layout_.num_data, block_cnt,
                      ef_search);
    } else {
      ERROR("PQ is not inited!");
      throw;
    }
    double start = elapsed();


    int init_ef = std::min(ef_search, 5);

    if (nav_graph) {
      get_entry_kernel(data_type_)<<<(num_queries + 1) / 2, 64, 0>>>(
          thrust::raw_pointer_cast(d_qdata.data()), nav_graph->data_dev,
          nav_graph->graph_dev, num_queries, nav_graph->num_node,
          layout_.num_dims, nav_graph->max_m, init_ef, nav_graph->start,
          thrust::raw_pointer_cast(d_entries.data()),
          thrust::raw_pointer_cast(d_neighbors_id.data()),
          thrust::raw_pointer_cast(d_neighbors_dist.data()));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<int> entries(num_queries);
    thrust::copy(d_entries.begin(), d_entries.end(), entries.begin());
    if (nav_graph) {
      nav_graph->translate(entries.data(), entries.size());
    } else {
      std::fill(entries.begin(), entries.end(), layout_.enter_point);
    }
    thrust::copy(entries.begin(), entries.end(), d_entries.begin());
    double t1 = elapsed();
    INFO("Init: {}", t1 - start);

    auto kernel_func = search_disk_graph_kernel<uint8_t>;
#pragma GCC diagnostics push
#pragma GCC diagnostics error "-Wswtich"
    switch (data_type_) {
    case UINT8:
      break;
    case FLOAT:
      kernel_func = search_disk_graph_kernel<float>;
      break;
    }
    int block_dim = std::max((layout_.max_m0 + 31) / 32 * 32, 64l);
#pragma GCC diagnostics pop
    kernel_func<<<
      block_cnt, block_dim,
      (sizeof(int) * 3 + sizeof(float) * 2) * (ef_search + layout_.max_m0)
        >>>
      (
        mem_data_,
        thrust::raw_pointer_cast(d_qdata.data()),
        layout_.num_dims,
        pq->get_device_ptr(),
        layout_.nodes_per_page, layout_.node_size, layout_.data_size,
        thrust::raw_pointer_cast(d_entries.data()),
        layout_.max_m0, ef_search, topk,
        thrust::raw_pointer_cast(d_nns.data()), 
        thrust::raw_pointer_cast(d_distances.data()), 
        thrust::raw_pointer_cast(d_found_cnt.data()), 
        num_queries
      );

    CHECK_CUDA(cudaDeviceSynchronize());
    double end = elapsed();

    DEBUG("End Search");
    INFO("Use time: {}", end - start);
    std::vector<int64_t> acc_visited_cnt(block_cnt);
    thrust::copy(d_nns.begin(), d_nns.end(), nns);
    thrust::copy(d_distances.begin(), d_distances.end(), distances);
    thrust::copy(d_found_cnt.begin(), d_found_cnt.end(), found_cnt);
    CHECK_CUDA(cudaDeviceSynchronize());
  }

}

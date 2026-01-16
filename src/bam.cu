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
#include "bam.hpp"
//#include "ssd_search_kernel.hpp"

#include "impl/bam/def.cuh"
#include "impl/bam/transfer.cuh"
//#include "impl/bam/search_v1.cuh"
#include "impl/bam/search.cuh"
#include "impl/nav.cuh"



namespace gustann {

  const char *const ctrls_paths[6] = {"/dev/libnvm0","/dev/libnvm1","/dev/libnvm2","/dev/libnvm3","/dev/libnvm4","/dev/libnvm5"};

  BaMExecutor::BaMExecutor(const std::string &fpath, const Layout &layout,
                           const DataType& data_type,
                           const BaMConfig &config, bool copy_data)
    : layout_(layout) {
    block_cnt_ = 112 * 100;
    block_dim_ = 32;
    visited_list_size_ = 8192 * 8;
    visited_table_size_ = visited_list_size_ * 2;
    data_type_ = data_type;
    
    init(config);
    if (copy_data) {
      copy_to_bam(fpath);
    }
#ifdef _USE_MEM
    read_to_mem(fpath);
#endif
  }

  void BaMExecutor::copy_to_bam(const std::string &fpath) {
    DEBUG("start copy, total page {}", layout_.num_pages);
    //std::ifstream input(fpath, std::ios::binary);
    FILE* input = fopen(fpath.c_str(), "rb");
    if (!input) {
      ERROR("Failed to open file: {}", fpath);
      exit(-1);
    }
    uint8_t* buff;

    uint64_t copy_block_cnt = 112 * 4;
    uint64_t copy_batch = 1000 * 1024 / 4 / copy_block_cnt;
    uint64_t copy_pages = copy_block_cnt * copy_batch; // 100MB

    /*
      {
      input.seekg(0, std::ios::end);      
      auto pos = input.tellg();
      uint64_t len = pos;
      DEBUG("Len: {}", len);
      }
    */
    CHECK_CUDA(cudaMallocHost(&buff, page_size_ * copy_pages));
    fseek(input, page_size_, SEEK_SET);
    //input.seekg(page_size_, std::ios::beg);
    CHECK_CUDA(cudaDeviceSynchronize());
    double start = elapsed();

    uint64_t tot_cnt = 0;
    int gb_cnt = 1;
    for (int64_t i = 0; i < layout_.num_pages; i += copy_pages) {
      // readsome may cause problems when reading large datasets
      // https://bugzilla.redhat.com/show_bug.cgi?id=1122595
      //size_t size = input.readsome((char*) buff, copy_pages * page_size_);
      size_t size = fread((char*) buff, sizeof(char), copy_pages * page_size_, input);
      //copy_data_to_ssd<<<copy_block_cnt, block_dim_>>>
      //  (bam_data_.a->d_array_ptr, buff, size, page_size_ * i);
      copy_page_to_ssd<<<copy_block_cnt, block_dim_>>>
        (bam_data_.h_pc->pdt.d_ctrls, bam_data_.h_pc->d_pc_ptr,
         bam_data_.ctrls.size(), buff, i, (size + page_size_ - 1) / page_size_, page_size_);

      auto pos = ftell(input);
            
      tot_cnt += (size + page_size_ - 1) / page_size_;
      if (tot_cnt >= gb_cnt * 10ll * 1024 * 1024 * 1024 / page_size_) {
        DEBUG("{} GB Copied", tot_cnt * page_size_ / 1024 / 1024 / 1024);
        gb_cnt++;
      }
      CHECK_CUDA(cudaDeviceSynchronize());
      /*
        if (i <= 159494 && 159494 < i + block_cnt_) {
        int a = 159494 - i;
        printf("%d\n", *(int*)(buff + a * 4096 + 128 * 4));
        }
      */
    }
    DEBUG("{} {}", tot_cnt, layout_.num_pages);
    ASSERT((tot_cnt == layout_.num_pages));
#ifdef ASYNC_READ
    bam_data_.h_pc->clear_cache();
#endif
    CHECK_CUDA(cudaDeviceSynchronize());
    double end = elapsed();
    CHECK_CUDA(cudaFreeHost(buff));
    f__k<<<1, 1>>>(bam_data_.a->d_array_ptr);
    DEBUG("finish copy, bandwidth = {} GB/s", 1. * layout_.num_pages * page_size_ / 1024 / 1024 / 1024 / (end - start));
    fclose(input);

  }

  void BaMExecutor::init(const BaMConfig &config) {
    int64_t ssd_pages = layout_.num_pages;
    for (int i = 0; i < config.num_ctrls; i++) {
      bam_data_.ctrls.push_back
        (new Controller(ctrls_paths[i],
                        config.nvm_namespace,
                        config.cuda_device,
                        config.queue_depth,
                        config.num_queues));
    }
    
    int ele_per_page = config.page_size;  
    
    int num_pages = config.num_page;
    
    DEBUG("Cache page: {}", num_pages);
    DEBUG("Page size: {}", config.page_size);
    DEBUG("Data page: {}", ssd_pages);
    
    bam_data_.h_pc = new page_cache_t(config.page_size,
                                      num_pages,
                                      config.cuda_device,
                                      bam_data_.ctrls[0][0],
                                      64,
                                      bam_data_.ctrls);
    int64_t data_size = ssd_pages * config.page_size;

    page_size_ = config.page_size;

    if (config.use_simple_cache) {
      this->use_simple_cache = true;
    } else {
      this->use_simple_cache = false;
      bam_data_.h_range = new range_t<uint8_t>
        (0, data_size, 0,
         ssd_pages,
         0, config.page_size, bam_data_.h_pc,
         config.cuda_device,
         STRIPE);
      bam_data_.d_range = (range_d_t<uint8_t>*) bam_data_.h_range->d_range_ptr;
      bam_data_.vr.push_back(bam_data_.h_range);
      bam_data_.a = new array_t<uint8_t> (data_size, 0, bam_data_.vr, config.cuda_device);
    }
    
    INFO("Inited BaM device");
  }

  void BaMExecutor::read_to_mem(const std::string &fpath) {
    FILE* input = fopen(fpath.c_str(), "rb");
    if (!input) {
      ERROR("Failed to open file: {}", fpath);
      exit(-1);
    }
    fseek(input, page_size_, SEEK_SET);
    CHECK_CUDA(cudaHostAlloc(&mem_data_, layout_.num_pages * page_size_, cudaHostAllocPortable));
    uint8_t* mem_data_host = mem_data_; //new uint8_t[num_pages_ * page_size_];
    ASSERT(fread(mem_data_host, 1, layout_.num_pages * page_size_, input) == layout_.num_pages * page_size_);
    //CHECK_CUDA(cudaMalloc(&mem_data_, num_pages_ * page_size_));
    //CHECK_CUDA(cudaMemcpy(mem_data_, mem_data_host, num_pages_ * page_size_, cudaMemcpyHostToDevice));
    fclose(input);
    //delete [] mem_data_host;
  }

  void BaMExecutor::search(const float *qdata, const int num_queries_,
                           const int topk, const int ef_search, int *nns,
                           float *distances, int *found_cnt,
                           PQSearch *pq, NavGraph *nav_graph) {
    int num_queries = num_queries_;
    //num_queries = 10;   
    
    thrust::device_vector<float> d_qdata(num_queries * layout_.num_dims);
    thrust::copy(qdata, qdata + num_queries * layout_.num_dims, d_qdata.begin());

    int aligned_ef = (ef_search + layout_.max_m0 + 31) / 32 * 32;

    thrust::device_vector<int> d_entries(num_queries);
    thrust::device_vector<int> d_nns(num_queries * topk);
    thrust::device_vector<float> d_distances(num_queries * topk);
    thrust::device_vector<int> d_found_cnt(num_queries);
    thrust::device_vector<uint32_t> d_neighbors_id(aligned_ef * block_cnt_);
    thrust::device_vector<float> d_neighbors_dist(aligned_ef * block_cnt_);


    DEBUG("Start Search");

    //fetch_all_data<<<1, 32>>>(bam_data_.a->d_array_ptr, num_pages_);
    if (pq) {
      pq->init_device(layout_.num_dims, layout_.num_data, block_cnt_,
                      ef_search);
    } else {
      ERROR("PQ is not inited!");
      throw;
    }
    bam_data_.a->print_reset_stats();
    CHECK_CUDA(cudaDeviceSynchronize());
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
#pragma GCC diagnostics pop
    kernel_func<<<
      block_cnt_, (layout_.max_m0 + 31) / 32 * 32,
      (sizeof(int) * 3 + sizeof(float) * 2) * (ef_search + layout_.max_m0)
        >>>
      (
#ifdef _USE_MEM
        mem_data_,
#else
        bam_data_.a->d_array_ptr,
#endif
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
    std::vector<int64_t> acc_visited_cnt(block_cnt_);
    thrust::copy(d_nns.begin(), d_nns.end(), nns);
    thrust::copy(d_distances.begin(), d_distances.end(), distances);
    thrust::copy(d_found_cnt.begin(), d_found_cnt.end(), found_cnt);
    CHECK_CUDA(cudaDeviceSynchronize());

    bam_data_.a->print_reset_stats();

  }

}

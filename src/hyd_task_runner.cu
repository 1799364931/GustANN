#include "hyd_task_runner.cuh"

#include "impl/nav.cuh"
#include "impl/pq.cuh"
#include "impl/util.cuh"

#include "impl/hyd_search_v1.cuh"
#include "impl/hyd_search_v2.cuh"

namespace gustann {

static __global__ void init_search(float *qdata, PQSearchData *pq_data,
                                   int stream_offset, int dim) {
  pq_data->init_query(qdata + blockIdx.x * dim, stream_offset);
}

static __global__ void copy_page(uint8_t *dest, uint8_t *src, int32_t *request,
                                 const int data_len, const int data_cnt) {
  int tid = threadIdx.x;
  assert(data_len % 4 == 0);

  int x = request[blockIdx.x];
  if (x == -1)
    return;
  uint32_t *local_dest = (uint32_t *)(dest + blockIdx.x * PAGE_SIZE);
  uint32_t *local_src = (uint32_t *)(src + blockIdx.x * PAGE_SIZE);
  int len = data_len / 4;
  int offset = x % data_cnt * len;
  for (int i = tid; i < len; i += blockDim.x) {
    local_dest[i + offset] = local_src[i + offset];
  }
}

void TaskRunner::init_query(const float *qdata, int _qcnt, int *_nns,
                            float *_dis, int *_found_cnt, int *_start_pt) {
  time_init_issue -= elapsed();
  latency -= elapsed();
  cnt_query++;
  qcnt = _qcnt;
  nns = _nns;
  distances = _dis;
  found_cnt = _found_cnt;

  round = 0;
  start_pt = _start_pt;

  // thrust::fill(d_found_cnt.begin(), d_found_cnt.end(), 0);
  // thrust::copy(qdata, qdata + qcnt * num_dims, d_qdata.begin());
  // thrust::fill()
  CHECK_CUDA(cudaStreamSynchronize(stream));
  time_gpu -= elapsed();
  CHECK_CUDA(cudaMemsetAsync(d_found_cnt, 0, sizeof(int) * qcnt, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_qdata, qdata, sizeof(float) * qcnt * num_dims,
                             cudaMemcpyHostToDevice, stream));
  init_search<<<qcnt, 64, 0, stream>>>(d_qdata, pq->get_device_ptr(),
                                       stream_offset, num_dims);
  if (nav_graph) {
    int init_ef = std::min(ef_search, 5);
    int dim = nav_graph->data_len;

    get_entry_kernel(data_type)<<<(qcnt + 1) / 2, 64, 0, stream>>>(
        d_qdata, nav_graph->data_dev, nav_graph->graph_dev, qcnt,
        nav_graph->num_node, dim, nav_graph->max_m, init_ef, nav_graph->start,
        request, thrust::raw_pointer_cast(d_neighbors_id.data()),
        thrust::raw_pointer_cast(d_neighbors_dist.data()));

  } else {

    for (int j = 0; j < qcnt; j++) {
#ifdef CACHE_START
      request[j] = start_pt[j] == -1 ? enter_point : start_pt[j];
#else
      // read_page(enter_point_, local_buf + PAGE_SIZE * j);
      memcpy(buffer + PAGE_SIZE * j, starter, PAGE_SIZE);
      // CHECK_CUDA(cudaMemcpyAsync(buffer + PAGE_SIZE * j, starter,
      //                            PAGE_SIZE, cudaMemcpyHostToHost,
      //                            stream));
      request[j] = enter_point;
#endif
    }
  }

  state = Q_INIT;
  time_init_issue += elapsed();
}

void TaskRunner::finish_query() {
  time_fin_issue -= elapsed();
#if 1
  // std::vector<int64_t> acc_visited_cnt(mini_batch);
  // thrust::copy(d_acc_visited_cnt.begin(), d_acc_visited_cnt.end(),
  // acc_visited_cnt.begin()); thrust::copy(d_nns.begin(), d_nns.begin() + qcnt
  // * topk, nns); thrust::copy(d_distances.begin(), d_distances.begin() + qcnt
  // * topk, distances); thrust::copy(d_found_cnt.begin(), d_found_cnt.begin() +
  // qcnt, found_cnt);
  CHECK_CUDA(cudaMemcpyAsync(nns, d_nns, qcnt * topk * sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaMemcpyAsync(distances, d_distances,
                             qcnt * topk * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaMemcpyAsync(found_cnt, d_found_cnt, qcnt * sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
#endif
  // CHECK_CUDA(cudaStreamSynchronize(stream));
  state = Q_RES;
  time_fin_issue += elapsed();
}

template <class T> void TaskRunner::gpu_search_v1() {
  // VERSION A
  get_pq_dist_kernel<<<qcnt, tcnt, 0, stream>>>(
      buffer, request, thrust::raw_pointer_cast(d_tmp_dist.data()),
      thrust::raw_pointer_cast(d_tmp_id.data()), pq->get_device_ptr(),
      nodes_per_page_, node_size_, data_size_, stream_offset, max_m0);
  // DEBUG("XXXX {}", i);
  update_kernel<T><<<(qcnt + 1) / 2, 64, 0, stream>>>(
      d_qdata, buffer, request, thrust::raw_pointer_cast(d_tmp_dist.data()),
      thrust::raw_pointer_cast(d_tmp_id.data()), num_data, num_dims, max_m0,
      ef_search, topk, d_nns, d_distances, d_found_cnt,
      thrust::raw_pointer_cast(d_acc_visited_cnt.data()),
      thrust::raw_pointer_cast(d_neighbors_id.data()),
      thrust::raw_pointer_cast(d_neighbors_dist.data()), nodes_per_page_,
      node_size_, data_size_, thrust::raw_pointer_cast(d_ctx.data()), qcnt);
}

template <class T> void TaskRunner::gpu_search_v2() {
#ifdef COPY_DATA

  // CHECK_CUDA(cudaMemcpyAsync(buffer_dev, buffer, PAGE_SIZE * qcnt,
  // cudaMemcpyDefault, stream));
  CHECK_CUDA(cudaMemcpyAsync(request_dev, request, sizeof(int32_t) * qcnt,
                             cudaMemcpyDefault, stream));
  /*
    for (int i = 0; i < qcnt; i++) {
    if (request[i] != -1) {
    CHECK_CUDA(cudaMemcpyAsync(buffer_dev + PAGE_SIZE * i, buffer + PAGE_SIZE *
    i, PAGE_SIZE, cudaMemcpyDefault, stream));
    }
    }p
  */
#ifdef FULL_READ
  cudaMemcpyAsync(buffer_dev, buffer, PAGE_SIZE * qcnt, cudaMemcpyHostToDevice,
                  stream);
#else
  copy_page<<<qcnt, 32, 0, stream>>>(buffer_dev, buffer, request_dev,
                                     node_size_, nodes_per_page_);
#endif
  uint8_t *buffer_t = buffer_dev; // Overwriting `buffer` below
  int32_t *request_t = request_dev;
#else
  uint8_t *buffer_t = buffer;
  int32_t *request_t = request;

#endif
  // printf("!!!!!\n");
  merge_data_kernel<<<
      qcnt, tcnt,
      ((sizeof(int) * 3 + sizeof(float) * 2) * (ef_search + max_m0)
       //+ ((pq->get_data().num_chunks + 1) * max_m0)
       ),
      stream>>>(buffer_t, request_t, pq->device_data.num_chunks,
                pq->device_data.pq_dists, pq->device_data.compressed_data,
                nodes_per_page_, node_size_, data_size_, stream_offset, max_m0,
                ef_search, thrust::raw_pointer_cast(d_neighbors_id.data()),
                thrust::raw_pointer_cast(d_neighbors_dist.data()),
                thrust::raw_pointer_cast(d_ctx.data()));

  unify_kernel<T><<<(qcnt + 1 / 2), 64, 0, stream>>>(
      d_qdata, buffer_t, request_t, num_dims, max_m0, ef_search, topk, d_nns,
      d_distances, d_found_cnt, thrust::raw_pointer_cast(d_neighbors_id.data()),
      thrust::raw_pointer_cast(d_neighbors_dist.data()), nodes_per_page_,
      node_size_, data_size_, thrust::raw_pointer_cast(d_ctx.data()), qcnt);

#ifdef COPY_DATA
  CHECK_CUDA(cudaMemcpyAsync(request, request_dev, sizeof(int32_t) * qcnt,
                             cudaMemcpyDefault, stream));
#endif
}
void TaskRunner::submit_gpu() {
  time_gpu_issue -= elapsed();
  CHECK_CUDA(cudaStreamSynchronize(stream));
  time_gpu -= elapsed();
#define gpu_search gpu_search_v2 // Some tricks

  if (data_type == UINT8) {
    gpu_search<uint8_t>();
  } else if (data_type == FLOAT) {
    gpu_search<float>();
  } else {
    ERROR("Invalid Data Type!");
    exit(-1);
  }

#undef gpu_search

  state = Q_GPU;
  time_gpu_issue += elapsed();
}

void TaskRunner::submit_ssd() {
  time_ssd_issue -= elapsed();
  CHECK_CUDA(cudaStreamSynchronize(stream));
  // DEBUG("YYYY {}", i);
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
  std::vector<std::pair<int, void *>> pages;
  for (int j = 0; j < qcnt; j++) {
    // request[j] = rand_r(&seed) % num_data; /// !!! REMOVE !!! ///
    if (request[j] != -1) {
      // printf("%d\n", request[j]);
      if (!(request[j] >= 0 && request[j] < num_data)) {
        fprintf(stderr, "??? %d\n", request[j]);
        throw;
      }
      // printf("%d %d ", j, request[j]);
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
  // printf("\n");
  // printf("F %d\n", finished);
  // printf("%d\n", qcnt);
  // uring.read_pages(threadid, pages);

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
      // printf("%lf %d %d\n", ssd_overall[round][i], ssd_cnt[i], sum);
    }
  }
#endif

  if (++round == 6) {
    for (int j = 0; j < qcnt; j++) {
      if (start_pt[j] == -1)
        start_pt[j] = request[j];
    }
  }
  if (finished) {
    time_ssd += elapsed();
    finish_query();
    // printf("FIN\n");
  } else {
    state = Q_SSD;
  }
  time_ssd_issue += elapsed();
}

bool TaskRunner::update_state() {
  switch (state) {
  case Q_INIT:
  case Q_GPU:
  case Q_RES: {
    auto err = cudaStreamQuery(stream);
    if (err == cudaSuccess) {
      if (state == Q_INIT) {
        time_gpu += elapsed();

        if (nav_graph) {
          nav_graph->translate(request, qcnt);
          submit_ssd();
        } else {
#ifdef CACHE_START
          submit_ssd();
#else
          submit_gpu();
#endif
        }
      } else if (state == Q_GPU) {
        // printf("!!!! GPU->SSD\n");
        time_gpu += elapsed();
        submit_ssd();
      } else {
        // printf("!!!!!\n");
        latency += elapsed();
        state = Q_FIN;
        return true;
      }
    } else if (err == cudaErrorNotReady) {
      // printf("Wait!\n");
    } else {
      CHECK_CUDA(err);
    }
    break;
  }
  case Q_SSD: {
    bool ready = loader->poll_task(cid);
    if (ready) {

      // printf("!!!! SSD->GPU \n");
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

TaskRunner::TaskRunner(int _tid, int _cid, int _mini_batch, int _num_dims,
                       int _topk, int64_t _num_data, int _max_m0,
                       int _ef_search, int _enter_point, uint8_t *_starter,
                       PQSearch *_pq, int nodes_per_page, int node_size,
                       int data_size, DataType data_type_,
                       std::shared_ptr<IndexLoader> _loader, NavGraph *_nav) {

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

  nodes_per_page_ = nodes_per_page;
  node_size_ = node_size;
  data_size_ = data_size;
  data_type = data_type_;
  loader = _loader;

  nav_graph = _nav;
  pq = _pq;

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
  printf("Now %lf/%lf B free mem\n", 1.0 * free_mem, 1.0 * tot_mem);
#endif
  // CHECK_CUDA(cudaMallocHost(&buffer, sizeof(uint8_t) * PAGE_SIZE *
  // mini_batch));

  buffer = loader->create_buffer((int64_t)PAGE_SIZE * mini_batch);

  CHECK_CUDA(cudaHostRegister(buffer, sizeof(uint8_t) * PAGE_SIZE * mini_batch,
                              cudaHostRegisterDefault));
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
  printf("Now %lf/%lf B free mem\n", 1.0 * free_mem, 1.0 * tot_mem);
#endif
}

TaskRunner::~TaskRunner() {
  if (stream) {
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

#ifdef COPY_DATA
  if (buffer_dev) {
    CHECK_CUDA(cudaFree(buffer_dev));
  }
  if (request_dev) {
    CHECK_CUDA(cudaFree(request_dev));
  }
#endif

  if (d_qdata) {
    CHECK_CUDA(cudaFree(d_qdata));
  }
  if (d_nns) {
    CHECK_CUDA(cudaFree(d_nns));
  }
  if (d_distances) {
    CHECK_CUDA(cudaFree(d_distances));
  }
  if (d_found_cnt) {
    CHECK_CUDA(cudaFree(d_found_cnt));
  }
  if (request) {
    CHECK_CUDA(cudaFreeHost(request));
  }
  if (buffer) {
    CHECK_CUDA(cudaHostUnregister(buffer));
    loader->destroy_buffer(buffer);
  }
  if (stream) {
    CHECK_CUDA(cudaStreamDestroy(stream));
  }
}
} // namespace gustann

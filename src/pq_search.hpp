#pragma once
#include <string>

struct PQData {
  int idx;
  float distance;
};


struct PQSearchData {
  static const int num_pivots = 256;
  int dim;
  int num_chunks;
  int num_pts;
  float* centroid; // 1 * dim;
  float* pivots; // num_pivots * dim
  float* pivots_t; // dim * num_pivots;
  int* chunk_id; // dim
  uint8_t* compressed_data; // num_pts * num_chunks
  float* pq_dists; // num_blocks * num_pivots * num_chunks
  PQData* pq_retset;
  
  __inline__ __device__ void init_query(float* q, int offset = 0);
  __inline__ __device__ float compute_dist(int idx, int offset = 0);
  __inline__ __device__ void compute_dist(int* idx, float* result, int cnt);
};


class PQSearch {
public:
  void read_data(std::string table_file, std::string vec_file);
  void init_device(int dim, int num_pts, int num_thread_blocks, int ef_search);
  inline PQSearchData* get_device_ptr() { return device_ptr; }
  const PQSearchData& get_data() const { return host_data; }
  PQSearchData device_data;
private:
  PQSearchData host_data;
  PQSearchData* device_ptr;
};


#include "pq_search.hpp"
#include <fstream>

#include "common.hpp"
#include "common_cuda.cuh"

namespace gustann {
  template <class T>
  static void read_bin(std::string file, int &npts, int &ndim, size_t offset,
                       T *&data);

  void PQSearch::read_data(std::string table_file, std::string vec_file) {
    DEBUG("Reading Compressed data");
    read_bin(vec_file, host_data.num_pts, host_data.num_chunks, 0, host_data.compressed_data);
    /*
      if (host_data.num_chunks % 4 != 0) {
      int padded_chunk = (host_data.num_chunks + 3) / 4;
      uint8_t* padded = new uint8_t[host_data.num_pts * padded_chunk];
    
      for (int i = 0; i < host_data.num_pts; i++) {
      for (int j = 0; j < host_data.num_chunks; j++) {
      padded[i * padded_chunk + j] = host_data.compressed_data[i * host_data.num_chunks + j];
      }
      }
      }
    */
    size_t* basic_offsets;
    int nr, nc;
    // read data from file
    DEBUG("Reading metadata");
    read_bin(table_file, nr, nc, 0, basic_offsets);
    ASSERT((nr == 4 || nr == 5) &&
           nc == 1); // PipeANN (and older version of DiskANN) use (nr == 5),
                     // while new verison of DiskANN use nr == 4.
    bool use_legacy = (nr == 5);
    DEBUG("Metadata: {} {} {} {} {}", basic_offsets[0], basic_offsets[1], basic_offsets[2], basic_offsets[3], (nr == 5 ? basic_offsets[4] : -1));

    DEBUG("Reading pivots");
    read_bin(table_file, nr, host_data.dim, basic_offsets[0], host_data.pivots);
    ASSERT(nr == host_data.num_pivots);
    host_data.pivots_t = new float[(size_t) host_data.num_pivots * host_data.dim];
    for (int i = 0; i < host_data.num_pivots; i++) {
      for (int j = 0; j < host_data.dim; j++) {
        host_data.pivots_t[j * host_data.num_pivots + i] = host_data.pivots[i * host_data.dim + j];
      }
    }
    DEBUG("Reading centroid");
    read_bin(table_file, nr, nc, basic_offsets[1], host_data.centroid);
    ASSERT(nr == host_data.dim && nc == 1);

    // Older version of DiskANN have the `rearrangement field` in
    // `basic_offsets[2]`;
    // For compatibility, PipeANN still contains this field.
    // Therefore for new DiskANN (nr == 4), chunk offsets is in basic_offset[2]
    // but in PipeANN, it's in basic_offset[3]
    int chunk_offset_idx = use_legacy ? 3 : 2;
    
    DEBUG("Reading chunk offsets");
    int* chunk_offsets;
    read_bin(table_file, nr, nc, basic_offsets[chunk_offset_idx],
             chunk_offsets);
    DEBUG( "{} {} {}", nr, nc, chunk_offset_idx);
    ASSERT(nr == host_data.num_chunks + 1 && nc == 1);
    host_data.chunk_id = new int[host_data.dim];
    memset(host_data.chunk_id, -1, sizeof(int) * host_data.dim);
    for (int i = 0; i < host_data.num_chunks; i++) {
      for (int j = chunk_offsets[i]; j < chunk_offsets[i + 1]; j++) {
        host_data.chunk_id[j] = i;
      }
    }
    delete [] basic_offsets;
    delete [] chunk_offsets;
    DEBUG("PQ data load ok: dim: {}, num_pts: {}, num_chunks: {}", host_data.dim, host_data.num_pts, host_data.num_chunks);
  }

  void PQSearch::init_device(int dim, int num_pts, int num_thread_blocks, int ef_search) {
    ASSERT(dim == host_data.dim);
    ASSERT(num_pts == host_data.num_pts);

#ifdef MEM_PROFILE
    // move data to device
    size_t free_mem, tot_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &tot_mem));
    printf("Now %lf/%lf B free mem\n",
           1.0 * free_mem, 1.0 * tot_mem );
#endif
  
    device_data = host_data;
    copy_to_dev(host_data.centroid, device_data.centroid, host_data.dim);
    copy_to_dev(host_data.pivots, device_data.pivots,
                (size_t) host_data.num_pivots * host_data.dim);
    copy_to_dev(host_data.pivots_t, device_data.pivots_t,
                (size_t) host_data.num_pivots * host_data.dim);
    copy_to_dev(host_data.chunk_id, device_data.chunk_id, host_data.dim);

#ifdef MEM_PROFILE
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &tot_mem));
    printf("Now %lf/%lf B free mem\n",
           1.0 * free_mem, 1.0 * tot_mem );
#endif
    copy_to_dev(host_data.compressed_data, device_data.compressed_data,
                (size_t) host_data.num_pts * host_data.num_chunks);
#ifdef MEM_PROFILE
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &tot_mem));
    printf("Now %lf/%lf B free mem\n",
           1.0 * free_mem, 1.0 * tot_mem );
#endif
    /*
      CHECK_CUDA(cudaMallocHost(&device_data.compressed_data,
      sizeof(uint8_t) * host_data.num_pts * host_data.num_chunks));
      memcpy(device_data.compressed_data, host_data.compressed_data,
      sizeof(uint8_t) * host_data.num_pts * host_data.num_chunks);
    */
    // AB_BUFFER
    CHECK_CUDA(cudaMalloc(&device_data.pq_dists, sizeof(float) * num_thread_blocks * host_data.num_pivots * host_data.num_chunks));
    //CHECK_CUDA(cudaMalloc(&device_data.pq_retset, sizeof(PQData) * num_thread_blocks * ef_search));
  
  
    DEBUG("F");
    copy_to_dev(&device_data, device_ptr, 1);
    DEBUG("PQ data moved to device");

#ifdef MEM_PROFILE
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &tot_mem));
    printf("Now %lf/%lf B free mem\n",
           1.0 * free_mem, 1.0 * tot_mem );
#endif
  }

  template<class T>
  static void read_bin(std::string filename, int& npts, int& ndim, size_t offset, T* &data) {
    std::ifstream ifile(filename);
    if (!ifile.is_open()) {
      ERROR("Cannot open file: {}", filename);
      exit(-1);
    }
    ifile.seekg(offset, std::ios::beg);
    ifile.read((char*)&npts, sizeof(int));
    ifile.read((char*)&ndim, sizeof(int));
    size_t len = (size_t)npts * ndim;
    data = new T [len];
    ifile.read((char*)data, sizeof(T) * len);
    DEBUG("Read {} x {} data from file: {}", npts, ndim, filename);
  }
}

#pragma once

#ifdef CHECK_CUDA
#undef CHECK_CUDA
#endif


static constexpr int WARP_SIZE = 32;

#define CHECK_CUDA(code) { checkCuda((code), __FILE__, __LINE__); }
inline void checkCuda(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::stringstream err;
    err << "Cuda Error: " << cudaGetErrorString(code) << " (" << file << ":" << line << ")";
    throw std::runtime_error(err.str());
  }
}

template <class T>
inline void copy_to_dev(T* host_data, T* &dev_data, size_t len) {
  CHECK_CUDA(cudaMalloc(&dev_data, sizeof(T) * len));
  CHECK_CUDA(cudaMemcpy(dev_data, host_data, sizeof(T) * len, cudaMemcpyHostToDevice));
}

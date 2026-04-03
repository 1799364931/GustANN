#pragma once

#include <string>
#include <vector>

namespace gustann {

struct HybridExecutorConfig {
  int mini_batch;
  int thread_cnt;
  int ctx_per_thread;
  enum {
    SPDK,
    MEMORY,
    URING,
    AIO,
  } use_backend;
  std::vector<std::string> ssd_lists;
};

} // namespace gustann

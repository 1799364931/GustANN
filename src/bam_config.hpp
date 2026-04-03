#pragma once

namespace gustann {

struct BaMConfig {
  int num_ctrls = 1;
  int queue_depth = 1024;
  int num_queues = 1;
  int cuda_device = 0;
  int nvm_namespace = 1;
  int page_size = 4096;
  int num_page = 1024;
  bool use_simple_cache = false;
};

} // namespace gustann

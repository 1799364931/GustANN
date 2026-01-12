#pragma once
#include <vector>
#include <utility>
#include <memory>

namespace gustann {
  using IoRequest = std::pair<int, void *>; // (blockId, dest_buffer)

  static constexpr int64_t PAGE_SIZE = 4096;
  class IndexLoader {
  public:
    // requests: list of IO requests
    // thread_id: thread ID
    // ctx_id: coroutine ID, must be distinct *globally*.
    virtual void submit_task(const std::vector<IoRequest>& requests,
                             int thread_id, int ctx_id) = 0;
    virtual bool poll_task(int ctx_id) = 0;
    virtual void log_latency(const std::vector<double> &percentages) {}
    virtual void clear_stats() {}
    virtual ~IndexLoader() {}
  };

  std::shared_ptr<IndexLoader> create_mem_loader_sync(const char* filename, int64_t num_pages);
  
} // namespace gustann


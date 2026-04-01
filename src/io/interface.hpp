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
    virtual uint8_t *create_buffer(int64_t size) = 0;
    virtual void destroy_buffer(uint8_t *) = 0;
    virtual ~IndexLoader() {}
  };

  std::shared_ptr<IndexLoader> create_mem_loader_sync(const char* filename, int64_t num_pages);

#ifdef USE_SPDK
  std::shared_ptr<IndexLoader>
  create_spdk_loader(const std::vector<std::string> &ssds, int queue_cap,
                     int thread_cnt, int ctx_cnt
  );
#endif

#ifdef USE_URING
  std::shared_ptr<IndexLoader> create_uring_loader(const char* filename, int ctx_cnt);
#endif

#ifdef USE_AIO
  std::shared_ptr<IndexLoader> create_aio_loader(const char* filename, int ctx_cnt);
#endif
} // namespace gustann


#include "interface.hpp"
#include "../common.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <algorithm>
#include <libaio.h>

#define AIO_MAX_EVENTS 256

namespace gustann {

  class AioLoader : public IndexLoader {
    int fd_;
    std::atomic<int>* pending_;
    io_context_t* aio_ctxs_;
    int ctx_cnt_;

  public:
    AioLoader(const char* filename, int ctx_cnt) : ctx_cnt_(ctx_cnt) {
      fd_ = ::open(filename, O_DIRECT | O_RDONLY);
      if (fd_ < 0) {
        ERROR("Failed to open index file for AIO: {}", filename);
        exit(-1);
      }
      pending_ = new std::atomic<int>[ctx_cnt_];
      aio_ctxs_ = new io_context_t[ctx_cnt_];
      for (int i = 0; i < ctx_cnt_; i++) {
        pending_[i].store(0);
        aio_ctxs_[i] = 0;
        int ret = io_setup(AIO_MAX_EVENTS, &aio_ctxs_[i]);
        if (ret != 0) {
          ERROR("io_setup failed for ctx {}: {}", i, strerror(-ret));
          exit(-1);
        }
      }
      INFO("AioLoader initialized: fd={}, ctx_cnt={}", fd_, ctx_cnt_);
    }

    void submit_task(const std::vector<IoRequest>& requests,
                     int thread_id, int ctx_id) override {
      size_t n_ops = requests.size();
      pending_[ctx_id].store(n_ops);

      size_t submitted = 0;
      while (submitted < n_ops) {
        size_t batch = std::min(n_ops - submitted, (size_t)AIO_MAX_EVENTS);
        struct iocb cbs[AIO_MAX_EVENTS];
        struct iocb* cb_ptrs[AIO_MAX_EVENTS];
        for (size_t j = 0; j < batch; j++) {
          auto& [blk, dst] = requests[submitted + j];
          int64_t offset = ((int64_t)blk + 1) * PAGE_SIZE;
          io_prep_pread(&cbs[j], fd_, dst, PAGE_SIZE, offset);
          cb_ptrs[j] = &cbs[j];
        }
        int64_t ret = io_submit(aio_ctxs_[ctx_id], (int64_t)batch, cb_ptrs);
        if (ret < 0) {
          ERROR("io_submit failed: returned {}, errno={}",
                ret, strerror(-ret));
          exit(-1);
        }
        submitted += ret;
      }
    }

    bool poll_task(int ctx_id) override {
      int remaining = pending_[ctx_id].load();
      if (remaining == 0) return true;

      struct io_event events[AIO_MAX_EVENTS];
      struct timespec timeout = {0, 0}; // non-blocking
      int ret = io_getevents(aio_ctxs_[ctx_id], 0, remaining, events, &timeout);
      if (ret > 0) {
        pending_[ctx_id].fetch_sub(ret);
      }
      return pending_[ctx_id].load() == 0;
    }

    uint8_t* create_buffer(int64_t size) override {
      void* buf = aligned_alloc(PAGE_SIZE, size);
      if (!buf) {
        ERROR("aligned_alloc failed for AIO buffer");
        exit(-1);
      }
      return (uint8_t*)buf;
    }

    void destroy_buffer(uint8_t* buf) override {
      free(buf);
    }

    ~AioLoader() {
      for (int i = 0; i < ctx_cnt_; i++) {
        io_destroy(aio_ctxs_[i]);
      }
      if (fd_ >= 0) ::close(fd_);
      delete[] pending_;
      delete[] aio_ctxs_;
    }
  };

  std::shared_ptr<IndexLoader> create_aio_loader(const char* filename,
                                                  int ctx_cnt) {
    return std::make_shared<AioLoader>(filename, ctx_cnt);
  }

} // namespace gustann

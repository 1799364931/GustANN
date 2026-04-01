#include "interface.hpp"
#include "../common.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <liburing.h>

#define URING_QD 256

namespace gustann {

  class IoUringLoader : public IndexLoader {
    int fd_;
    std::atomic<int>* pending_;
    struct io_uring* rings_;
    int ctx_cnt_;

  public:
    IoUringLoader(const char* filename, int ctx_cnt) : ctx_cnt_(ctx_cnt) {
      fd_ = ::open(filename, O_DIRECT | O_RDONLY);
      if (fd_ < 0) {
        ERROR("Failed to open index file for io_uring: {}", filename);
        exit(-1);
      }
      pending_ = new std::atomic<int>[ctx_cnt_];
      rings_ = new struct io_uring[ctx_cnt_];
      for (int i = 0; i < ctx_cnt_; i++) {
        pending_[i].store(0);
        int ret = io_uring_queue_init(URING_QD, &rings_[i], 0);
        if (ret < 0) {
          ERROR("io_uring_queue_init failed for ctx {}: {}", i, strerror(-ret));
          exit(-1);
        }
      }
      INFO("IoUringLoader initialized: fd={}, ctx_cnt={}", fd_, ctx_cnt_);
    }

    void submit_task(const std::vector<IoRequest>& requests,
                     int thread_id, int ctx_id) override {
      struct io_uring* ring = &rings_[ctx_id];
      pending_[ctx_id].store(requests.size());

      for (auto& [blk, dst] : requests) {
        auto sqe = io_uring_get_sqe(ring);
        if (!sqe) {
          io_uring_submit(ring);
          sqe = io_uring_get_sqe(ring);
          if (!sqe) {
            ERROR("io_uring SQE exhausted on ctx {}", ctx_id);
            exit(-1);
          }
        }
        int64_t offset = ((int64_t)blk + 1) * PAGE_SIZE;
        io_uring_prep_read(sqe, fd_, dst, PAGE_SIZE, offset);
        sqe->user_data = 0;
      }
      io_uring_submit(ring);
    }

    bool poll_task(int ctx_id) override {
      struct io_uring* ring = &rings_[ctx_id];
      // Drain all available completions
      while (pending_[ctx_id].load() > 0) {
        struct io_uring_cqe* cqe = nullptr;
        int ret = io_uring_peek_cqe(ring, &cqe);
        if (ret == -EAGAIN) break; // nothing ready
        if (ret < 0) break;
        if (cqe->res < 0) {
          ERROR("io_uring read error: {}", strerror(-cqe->res));
        }
        io_uring_cqe_seen(ring, cqe);
        pending_[ctx_id].fetch_sub(1);
      }
      return pending_[ctx_id].load() == 0;
    }

    uint8_t* create_buffer(int64_t size) override {
      void* buf = aligned_alloc(PAGE_SIZE, size);
      if (!buf) {
        ERROR("aligned_alloc failed for io_uring buffer");
        exit(-1);
      }
      return (uint8_t*)buf;
    }

    void destroy_buffer(uint8_t* buf) override {
      free(buf);
    }

    ~IoUringLoader() {
      for (int i = 0; i < ctx_cnt_; i++) {
        io_uring_queue_exit(&rings_[i]);
      }
      if (fd_ >= 0) ::close(fd_);
      delete[] pending_;
      delete[] rings_;
    }
  };

  std::shared_ptr<IndexLoader> create_uring_loader(const char* filename,
                                                    int ctx_cnt) {
    return std::make_shared<IoUringLoader>(filename, ctx_cnt);
  }

} // namespace gustann

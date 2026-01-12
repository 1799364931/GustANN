#include "interface.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gustann {
  // Make a naiive fully synchornous memcpy for simplexity
  // TODO: Avoid DRAM memcpy (using cudaMemcpyAsync) while enabling selective transfer
  class MemLoaderSync : public IndexLoader {
    uint8_t *index;
  public:
    MemLoaderSync(const char* filename, int64_t num_pages) {
      FILE *file = fopen(filename, "rb");
      if (!file) {
        printf("Faile to open Mem Index");
        exit(-1);
      }
      index = new uint8_t [num_pages * PAGE_SIZE];
      fseek(file, PAGE_SIZE, SEEK_SET);
      int64_t ret = fread(index, PAGE_SIZE, num_pages, file);
      if (ret != num_pages) {
        printf("Mem Index Load FAILED!\n");
        printf("%ld %ld\n", ret, num_pages);
        exit(-1);
      }
      fclose(file);
      printf("Mem Index Loaded!\n");
    }
    void submit_task(const std::vector<IoRequest>& pages, int, int) override {
      for (auto [blk, dst] : pages) {
        memcpy(dst, index + (int64_t) blk * PAGE_SIZE, PAGE_SIZE);
      }
    }
    bool poll_task(int) override { return true; }
    ~MemLoaderSync() {
      delete [] index;
    }
  };

  std::shared_ptr<IndexLoader> create_mem_loader_sync(const char *filename,
                                                      int64_t num_pages) {
    std::shared_ptr<IndexLoader> mem_io =
        std::make_shared<MemLoaderSync>(filename, num_pages);
    return mem_io;
  }

}

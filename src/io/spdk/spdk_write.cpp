#include "spdk_wrapper.h"
#include <fstream>

constexpr long PG_SIZE = 4096;

int main(int argc, char** argv) {

  if (argc != 3) {
    printf("%s: <input> <ssd_list>\n", argv[0]);
    return -1;
  }
  std::unique_ptr<spdk_wrapper::SpdkWrapper> spdk;
  spdk = spdk_wrapper::SpdkWrapper::create(32);
  std::vector<std::string> ssds;
  std::fstream stream(argv[2]);
  printf("%s\n", argv[2]);
  
  if (stream) {
    std::string s;
    while(stream >> s) {
      printf("USE SSD: \"%s\"\n", s.c_str());
      ssds.push_back(s);
    }
  }

  spdk->Init(ssds);

  FILE* f0 = fopen(argv[1], "rb");
  fseek(f0, PAGE_SIZE, SEEK_SET);

  const int BATCH_SIZE = 512;
  char* buff = (char*) spdk_dma_zmalloc(BATCH_SIZE * PG_SIZE, PG_SIZE, NULL);
  
  
  int shards = ssds.size();
  long in_flight = 0;
  long cnt = 0;

  while(true) {
    int block = cnt % BATCH_SIZE;
    if (fread(buff + block * PG_SIZE, 1, PG_SIZE, f0) != PG_SIZE)
      break;
    spdk->SubmitWriteCommand(buff + block * PG_SIZE, PG_SIZE, cnt / shards,
                             [](void *ctx, const struct spdk_nvme_cpl *cpl) {
                                 if (spdk_nvme_cpl_is_error(cpl)) {
                                   printf("Error!!!!");
                                 }
                                 int* in_flight = (int*) ctx;
                                 (*in_flight)--;
                             }, &in_flight, cnt % shards, 0);
    in_flight++;
    if (++cnt % BATCH_SIZE == 0) {
      while(in_flight > 0) {
        for (int i = 0; i < shards; i++) {
          spdk->PollCompleteQueue(i, 0);
        }
      }
    }
    if (cnt % 1000000 == 0) {
      printf("%ld Blocks\n", cnt);
    }
  }
  while(in_flight > 0) {
    for (int i = 0; i < shards; i++) {
      spdk->PollCompleteQueue(i, 0);
    }
  }
  printf("%ld Block in Total!\n", cnt);

}

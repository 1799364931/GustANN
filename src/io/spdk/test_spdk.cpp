#include "spdk_wrapper.h"
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#include <functional>
#include <queue>

constexpr int PG_SIZE = 4096;

std::unique_ptr<spdk_wrapper::SpdkWrapper> spdk;

template <class T>
struct Queue {
  int head, tail, size, cap;
  T* data;
  Queue(int cap): cap(cap) {
    data = new T[cap];
    head = 0, tail = 0, size = 0;
  }
  bool empty() {
    return size == 0;
  }
  void push(const T &x) {
    data[tail] = x;
    tail = (tail + 1) % cap;
    size++;
  }
  T pop() {
    int r = head;
    head = (head + 1) % cap;
    size--;
    return data[r];
  }
};  

struct Ctx {
  unsigned int seed;
  int ns_id;
  int qp_id;
  long r;
  uint8_t* buffer;
  std::atomic<int>* ssd_req;
  Queue<Ctx*>* queue;
};

const int range = 20 * 1024 * 1024; // 20M * 4KB = 80 GB
bool submit(Ctx* ctx);
std::atomic<int>* all;

void callback(void* ctx, const struct spdk_nvme_cpl *cpl) {
  if (spdk_nvme_cpl_is_error(cpl)) {
    printf("Error!!!!");
  }

  Ctx* cur = (Ctx*) ctx;
  //printf("Done!!! %d\n", cur->ssd_req->load());
  cur->r = rand_r(&cur->seed) % range;

  all[cur->ns_id]++;
  //cur->ns_id = rand_r(&cur->seed) % 3;
  submit(cur);
}

bool submit(Ctx* ctx) {
  int ret = spdk->TrySubmitReadCommand(ctx->buffer, PG_SIZE, ctx->r, callback, ctx,
                                       ctx->ns_id, ctx->qp_id);
  if (ret != 0) {
    if (ret == -ENOMEM) {
      ctx->queue->push(ctx);
      return false;
    } else {
      printf("Error!!!\n");
    }
  }
  ctx->ssd_req->fetch_add(1);
  return true;
}


int main() {
  spdk = spdk_wrapper::SpdkWrapper::create(16);
  std::vector<std::string> ssds = {
    "0000:8b:00.0",
    "0000:8c:00.0",
    "0000:8d:00.0",
    "0000:8e:00.0",
    "0000:22:00.0",
    "0000:23:00.0",
  };
  spdk->Init(ssds);
  int num_ssd = ssds.size();
  
  all = (std::atomic<int>*) new int [ssds.size()];
  memset(all, 0, sizeof (int) * ssds.size());

  std::atomic<int> ssd_req(0);
    
  auto worker = [&](int threadid) {
    const int BATCH_SIZE = 1024;
    uint8_t *buffer =
      (uint8_t*) spdk_dma_zmalloc(BATCH_SIZE * PG_SIZE, PG_SIZE, NULL);
    //  (uint8_t *) 
    // spdk_malloc(BATCH_SIZE * PG_SIZE, 0, NULL,
    //             SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(threadid * 2 + 4, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set) != 0) {
      perror("pthread_setaffinity_np");
      exit(-1);
    }


    std::atomic<int> ok_cnt(0);
    unsigned int seed = threadid;
    printf("%d started\n", threadid);
    printf("%d\n", num_ssd);

    
/*
    while(true) {

      for (int i = 0; i < BATCH_SIZE; i++) {
        long d = rand_r(&seed) % num_ssd;
        long r = rand_r(&seed) % range;
        spdk->SubmitReadCommand(buffer + i * PG_SIZE, PG_SIZE, r,
                                [](void* ctx, const struct spdk_nvme_cpl *cpl) {
                                    if (spdk_nvme_cpl_is_error(cpl)) {
                                      printf("Error!!!!");
                                    }
                                  auto cnt = (std::atomic<int>*) ctx;
                                  cnt->fetch_add(1);
                                }, 
                                &ok_cnt, d, threadid);
        ssd_req++;
      }
      while(ok_cnt.load() < BATCH_SIZE) {
        for (int i = 0; i < num_ssd; i++) {
          spdk->PollCompleteQueue(i, threadid);
        }
      }
      ok_cnt = 0;
    }
 */
    std::vector<Ctx> ctxs(BATCH_SIZE);
    //std::queue<Ctx*> queue;
    Queue<Ctx*> queue(BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++) {
      ctxs[i].buffer = buffer + i * PG_SIZE;
      ctxs[i].qp_id = threadid;

      ctxs[i].seed = threadid * BATCH_SIZE + i;
      ctxs[i].ssd_req = &ssd_req;
      ctxs[i].queue = &queue;
      ctxs[i].ns_id = (threadid ) % ssds.size(); //threadid; //rand_r(&ctxs[i].seed) % ssds.size();
      ctxs[i].r = rand_r(&ctxs[i].seed) % range;
      submit(&ctxs[i]);
      
    }
    while(true) {
      
      while (!queue.empty()) {
        auto c = queue.pop();
        if (!submit(c)) {
          break;
        }
      }
      //usleep(1000);
      for (int i = 0; i < num_ssd; i++) {
        
      }
      
      spdk->PollCompleteQueue(threadid, threadid) ;

    }
  };
  std::vector<std::thread> th;
  int thread_num = ssds.size();
  for (int i = 0; i < thread_num; i++) {
    th.emplace_back(worker, i);
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  long last = 0;
  while(true) {
    sleep(1);
    long cur = ssd_req;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    long r = cur - last;
    last = cur;

    printf("Bandwidth: %lf GB/s, %ld\n", r * PG_SIZE / elapsed.count() / 1024 / 1024 / 1024, r);
    for (int i = 0; i < (int) ssds.size(); i++) {
      printf("%d ", all[i].load());
    }
    printf("\n");
    
    start = end;
  }
  for (int i = 0; i < thread_num; i++) {
    th[i].join();
  }
  
}

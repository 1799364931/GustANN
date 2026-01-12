
#include "folly_wrapper.h"
#include "spdk/nvme.h"
#include "spdk_wrapper.h"

#include <algorithm>
#include <stdint.h>

//const char *using_ssd = "0000:8c:00.0";

namespace spdk_wrapper {

class SpdkWrapperImplementation : public SpdkWrapper {
private:
  static constexpr int MAX_QPAIR_NUM = 32;
  int queue_cnt;

 public:
  void Init(const std::vector<std::string>& _use_ssds) override {
    
    INFO0("Initializing NVMe Controllers");

    // See: https://github.com/spdk/spdk/commit/57fd99b91e71a4baa5543e19ff83958dc99d4dac
    opts_.opts_size = sizeof(opts_); 
    spdk_env_opts_init(&opts_);

    ASSERT(spdk_env_init(&opts_) >= 0 && "Unable to initialize SPDK env\n");



    g_use_ssds = _use_ssds;
    g_namespaces_.resize(g_use_ssds.size());
    for (auto &ns_entry: g_namespaces_) {
      ns_entry.ctrlr = nullptr;
    }

    spdk_nvme_trid_populate_transport(&g_trid_, SPDK_NVME_TRANSPORT_PCIE);
    snprintf(g_trid_.subnqn, sizeof(g_trid_.subnqn), "%s",
             SPDK_NVMF_DISCOVERY_NQN);

    ASSERT(spdk_nvme_probe(
                 &g_trid_, this, SpdkWrapperImplementation::ProbeCallBack,
                 SpdkWrapperImplementation::AttachCallBack, nullptr) == 
           0 && "spdk_nvme_probe failed");

    ASSERT(g_controllers_.size() != 0 && "no NVMe controllers found");
    INFO0("Initialization complete");

    for (auto &ns_entry : g_namespaces_) {
      ASSERT(ns_entry.ctrlr != nullptr);
      for(int i = 0; i < queue_cnt; i++) {
        struct spdk_nvme_io_qpair_opts opts;
        spdk_nvme_ctrlr_get_default_io_qpair_opts(ns_entry.ctrlr, &opts, sizeof(opts));
        
        opts.delay_cmd_submit = true;
        //opts.create_only = true;
        opts.async_mode = true;
        ns_entry.qpair[i] = spdk_nvme_ctrlr_alloc_io_qpair(ns_entry.ctrlr, &opts, sizeof(opts));
        //printf("!!!%d\n", i);
        ASSERT(ns_entry.qpair[i] != nullptr &&
               "ERROR: spdk_nvme_ctrlr_alloc_io_qpair() failed");
      }
    }
    
    INFO("Allocated {} qpairs", queue_cnt);
  }

  SpdkWrapperImplementation(int queue_cnt) : queue_cnt(queue_cnt) {
    spdlog::set_pattern("[%^%-8l%$] %Y-%m-%d %H:%M:%S %v");
    logger_ = spdlog::default_logger();
    
    ASSERT(queue_cnt <= MAX_QPAIR_NUM);
  }

  ~SpdkWrapperImplementation() {
    for (auto ns_entry : g_namespaces_) {
      for(int qp_id = 0; qp_id < queue_cnt; qp_id++){
        spdk_nvme_ctrlr_free_io_qpair(ns_entry.qpair[qp_id]);
      }
      struct spdk_nvme_detach_ctx *detach_ctx = NULL;
      spdk_nvme_detach_async(ns_entry.ctrlr, &detach_ctx);
      if (detach_ctx) {
        spdk_nvme_detach_poll(detach_ctx);
      }
    }
  }

  
  
  void SubmitWriteCommand(const void *pinned_src, const int64_t bytes,
                         const int64_t lba, spdk_nvme_cmd_cb func,
                         void *ctx, int ns_id, int qp_id) override {
    auto ns_entry = g_namespaces_.at(ns_id);
    uint32_t lba_count = (bytes + kLBASize_ - 1) / kLBASize_;

    int64_t real_lba = lba * (LBASize_ / kLBASize_);

    while(true) {
      int ret =
        spdk_nvme_ns_cmd_write(ns_entry.ns, ns_entry.qpair[qp_id], (void *)pinned_src,
                               real_lba, lba_count, func, ctx, 0);

      if (ret == 0) {
        return;
      } else if (ret == -ENOMEM) {
        //FB_LOG_EVERY_MS(ERROR, 10000)
        //    << "SubmitReadCommand return with ENOMEM, let's poll CQ";
        PollCompleteQueue(ns_id, qp_id);
      } else {
        CRITICAL("SubmitReadCommand Error {}", ret);
      }

    }
  }

  int TrySubmitReadCommand(void *pinned_dst, const int64_t bytes,
                           const int64_t lba, spdk_nvme_cmd_cb func,
                           void *ctx, int ns_id, int qp_id) override {
    auto ns_entry = g_namespaces_.at(ns_id);
    //printf("%d %d\n", ns_id, gettid());
    uint32_t lba_count = (bytes + kLBASize_ - 1) / kLBASize_;
    int64_t real_lba = lba * (LBASize_ / kLBASize_);

    return spdk_nvme_ns_cmd_read(ns_entry.ns, ns_entry.qpair[qp_id], pinned_dst,
                                 real_lba, lba_count, func, ctx, 0);


  }
  
  
  void SubmitReadCommand(void *pinned_dst, const int64_t bytes,
                         const int64_t lba, spdk_nvme_cmd_cb func,
                         void *ctx, int ns_id, int qp_id) override {
    auto ns_entry = g_namespaces_.at(ns_id);
    uint32_t lba_count = (bytes + kLBASize_ - 1) / kLBASize_;
    int64_t real_lba = lba * (LBASize_ / kLBASize_);
    while (1) {
      auto ret = spdk_nvme_ns_cmd_read(ns_entry.ns, ns_entry.qpair[qp_id], pinned_dst,
      
                                 real_lba, lba_count, func, ctx, 0);
      if (ret == 0) {
        return;
      } else if (ret == -ENOMEM) {
        //FB_LOG_EVERY_MS(ERROR, 10000)
        //    << "SubmitReadCommand return with ENOMEM, let's poll CQ";
        PollCompleteQueue(ns_id, qp_id);
      } else {
        CRITICAL("SubmitReadCommand Error {}", ret);
      }
    }
  }

  inline void PollCompleteQueue(int ns_id, int qp_id) override {
    auto ns_entry = g_namespaces_[ns_id];
    spdk_nvme_qpair_process_completions(ns_entry.qpair[qp_id], 0);
  }

  inline int GetLBASize() const override { return LBASize_; }

  inline uint64_t GetLBANumber() const override {
    // uint64_t capacity = 3200631791616LL;
    uint64_t capacity = 300 * 1024 * 1024 * 1024LL;
    ASSERT(0 == capacity % GetLBASize());
    return capacity / GetLBASize();
  }

  
 private: 
  static bool ProbeCallBack(void *cb_ctx,
                            const struct spdk_nvme_transport_id *trid,
                            struct spdk_nvme_ctrlr_opts *opts) {
    INFO("Found {}", trid->traddr);


    SpdkWrapperImplementation *ptr = (SpdkWrapperImplementation *)(cb_ctx);
    std::string find_name = trid->traddr;
    if (std::find(ptr->g_use_ssds.begin(),
                  ptr->g_use_ssds.end(),
                  find_name) == ptr->g_use_ssds.end()) return false;
    INFO("Default Queue Depth: {}", opts->io_queue_size);
    // max io_queue_size, as same in SPDK perf.
    opts->io_queue_size = UINT16_MAX;
    INFO("Attaching to {}", trid->traddr);
    return true;
  }

  static void AttachCallBack(void *cb_ctx,
                             const struct spdk_nvme_transport_id *trid,
                             struct spdk_nvme_ctrlr *ctrlr,
                             const struct spdk_nvme_ctrlr_opts *opts) {
    SpdkWrapperImplementation *ptr = (SpdkWrapperImplementation *)(cb_ctx);
    INFO("Attached to {}", trid->traddr);

    /*
     * spdk_nvme_ctrlr is the logical abstraction in SPDK for an NVMe
     *  controller.  During initialization, the IDENTIFY data for the
     *  controller is read using an NVMe admin command, and that data
     *  can be retrieved using spdk_nvme_ctrlr_get_data() to get
     *  detailed information on the controller.  Refer to the NVMe
     *  specification for more details on IDENTIFY for NVMe controllers.
     */
    const struct spdk_nvme_ctrlr_data *cdata = spdk_nvme_ctrlr_get_data(ctrlr);

    std::unique_ptr<char[]> buf(new char[1024]);
    std::snprintf(buf.get(), 1024, "%-20.20s (%-20.20s)", cdata->mn, cdata->sn);
    std::string name(buf.get(), buf.get() + 1024 - 1);

    ptr->g_controllers_[name] = ctrlr;

    /*
     * Each controller has one or more namespaces.  An NVMe namespace is
     * basically equivalent to a SCSI LUN.  The controller's IDENTIFY data tells
     * us how many namespaces exist on the controller.  For Intel(R) P3X00
     * controllers, it will just be one namespace.
     *
     * Note that in NVMe, namespace IDs start at 1, not 0.
     */
    for (int nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); nsid != 0;
         nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid)) {
      struct spdk_nvme_ns *ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
      if (ns == NULL || !spdk_nvme_ns_is_active(ns)) {
        continue;
      }

      int sector_size = spdk_nvme_ns_get_sector_size(ns);
      //printf("%d\n", sector_size);


      std::string target_name = trid->traddr;
      auto pos = std::find(ptr->g_use_ssds.begin(),
                           ptr->g_use_ssds.end(),
                           target_name) - ptr->g_use_ssds.begin();

      ns_entry& entry = ptr->g_namespaces_.at(pos);
      entry.ctrlr = ctrlr;
      entry.ns = ns;
     
      INFO("Namespace ID: {} size: {} GB (SSD {})",
           spdk_nvme_ns_get_id(ns),
           spdk_nvme_ns_get_size(ns) / 1000000000,
           pos);

       // use only first namespace
      break;

    }
  }
  spdk_env_opts opts_;
  spdk_nvme_transport_id g_trid_  = {};
  int kLBASize_ = 512;
  const int LBASize_ = 4096;
  std::unordered_map<std::string, spdk_nvme_ctrlr *> g_controllers_;

  struct ns_entry {
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;
    struct spdk_nvme_qpair *qpair[MAX_QPAIR_NUM];
  };

  std::vector<ns_entry> g_namespaces_;
  std::vector<std::string> g_use_ssds;
};

std::unique_ptr<SpdkWrapper> SpdkWrapper::create(int queue_cnt) {
  std::unique_ptr<SpdkWrapper> ret =
      std::make_unique<SpdkWrapperImplementation>(queue_cnt);
  return ret;
}

}

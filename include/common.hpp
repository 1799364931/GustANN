#pragma once
#include <sstream>

#define SPDLOG_EOL ""
#define SPDLOG_TRACE_ON
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#define INFO SPDLOG_INFO
#define DEBUG SPDLOG_DEBUG
#define WARN SPDLOG_WARN
#define ERROR SPDLOG_ERROR

#define ASSERT(x) do { if (!(x)) { ERROR("Assertion failed {}", #x); abort();}} while(0)


namespace gustann {
  struct BaMConfig {
    int num_ctrls = 1;
    int queue_depth = 1024;
    int num_queues = 1;
    int cuda_device = 0;
    int nvm_namespace = 1;
    int page_size = 4096;
    int num_page = 1024; // cached page
    bool use_simple_cache = false;
  };
  enum DataType {
    FLOAT,
    UINT8,
  };
}

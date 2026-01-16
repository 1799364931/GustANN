#pragma once
#include <sstream>

#include <sys/time.h>

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

  enum DataType {
    UINT8 = 0,
    FLOAT = 1,
  };


  enum DistFunc {
    L2,
  };

  static double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
  }

  
  static void bind_core(int core_num) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core_num, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set) != 0) {
      perror("pthread_setaffinity_np");
      exit(-1);
    }
  }
}

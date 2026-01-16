#pragma once

#include "pq.cuh"

namespace gustann {
  struct __align__(128) Data {
    int visited_cnt;
    int size;
  };
}

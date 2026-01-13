#pragma once
//#include "types.hpp"
#include "common.hpp"
#include "pq_search.hpp"

namespace gustann {  
#ifdef _USE_BAM
#endif

  struct __align__(128) Data {
    int visited_cnt;
    int size;
  };

}

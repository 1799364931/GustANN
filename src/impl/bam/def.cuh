#pragma once

#include "page_cache.h"

namespace gustann {
  
  using data_type = uint8_t;
  //#define _IN_MEM
  static const int WARP_SIZE = 32;
#ifdef _IN_MEM
  using DiskData = uint8_t;
#else
  using DiskData = array_d_t<uint8_t>;
#endif


}

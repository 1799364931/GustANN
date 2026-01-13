
add_executable(split_file split_file.cpp)
add_executable(analyze analyze.cpp)
add_executable(gen_small_file gen_small_file.cpp)
add_executable(trans_vec transform_vec.cpp)
add_executable(check_recall check_recall.cpp)

set(HYBRID_LINK_LIBRARY spdlog json11 argparse)
set(HYBRID_INCLUDE_LIBRARY ../include)

add_executable(search_disk_hybrid search_disk.cu
  ../src/ssd_search.cu ../src/ssd_search_kernel_hybrid.cu ../src/pq_search.cu ../src/ssd_search_hybrid.cu ../src/nav_graph.cu
  ../src/io/mem_io.cpp
)
target_include_directories(search_disk_hybrid PRIVATE ${HYBRID_INCLUDE_LIBRARY})
target_link_libraries(search_disk_hybrid PRIVATE ${HYBRID_LINK_LIBRARY})
target_link_libraries(search_disk_hybrid PRIVATE index_loader)
target_compile_options(search_disk_hybrid
  PRIVATE "-lineinfo"
  PRIVATE "-DHYBRID_CALC"
)

add_executable(search_disk_hybrid_float search_disk.cu
  ../src/ssd_search.cu ../src/ssd_search_kernel_hybrid.cu ../src/pq_search.cu ../src/ssd_search_hybrid.cu ../src/nav_graph.cu
)
target_include_directories(search_disk_hybrid_float PRIVATE ${HYBRID_INCLUDE_LIBRARY})
target_link_libraries(search_disk_hybrid_float PRIVATE ${HYBRID_LINK_LIBRARY})
target_link_libraries(search_disk_hybrid_float PRIVATE index_loader)
target_compile_options(search_disk_hybrid_float
  PRIVATE "-lineinfo"
  PRIVATE "-DHYBRID_CALC"
  PRIVATE "-DFLOAT_DATA"
)

IF(GUSTANN_USE_BAM)
  message("ENABLE BAM")
  set(APP_LINK_LIBRARY spdlog json11 argparse
    ${CMAKE_SOURCE_DIR}/deps/bam/build/lib/libnvm.so)
  set(APP_INCLUDE_LIBRARY ../include
    ../deps/bam/include ../deps/bam/include/freestanding/include)
  
  add_executable(search_disk search_disk.cu
    ../src/ssd_search.cu ../src/pq_search.cu
    ../src/nav_graph.cu ../src/bam.cu
  )
  target_include_directories(search_disk PRIVATE ${APP_INCLUDE_LIBRARY})
  target_link_libraries(search_disk PRIVATE ${APP_LINK_LIBRARY})
  target_compile_options(search_disk
    #PRIVATE "-DASYNC_READ"
    PRIVATE "-D_USE_BAM"
    PRIVATE "-lineinfo"
    PRIVATE "-O2"
  )
ENDIF()

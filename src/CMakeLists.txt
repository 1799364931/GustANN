add_subdirectory(io)

add_library(gustann layout.cpp ssd_search.cu pq_search.cu hybrid.cu nav_graph.cu hyd_task_runner.cu pure_mem.cu)
target_link_libraries(gustann
  PUBLIC spdlog argparse
  PRIVATE index_loader)

if (GUSTANN_USE_BAM)
  target_compile_definitions(gustann PUBLIC USE_BAM)
  target_link_libraries(gustann PUBLIC
    ${CMAKE_SOURCE_DIR}/deps/bam/build/lib/libnvm.so)
  target_include_directories(gustann PRIVATE
    ../deps/bam/include ../deps/bam/include/freestanding/include)
  target_sources(gustann PRIVATE bam.cu)
endif()

target_compile_options(gustann
  PRIVATE "-lineinfo"
  PUBLIC "-O2" "-g"
)

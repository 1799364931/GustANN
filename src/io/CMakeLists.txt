if (GUSTANN_USE_SPDK)
  add_subdirectory(spdk/)
endif()

add_library(index_loader mem_io.cpp)
target_compile_options(index_loader PUBLIC "-O2" "-g")
target_link_libraries(index_loader PUBLIC spdlog)

if (GUSTANN_USE_SPDK)
  target_link_directories(index_loader PUBLIC spdk_io)
endif()

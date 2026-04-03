# Helper programs

add_executable(split_file split_file.cpp)
add_executable(analyze analyze.cpp)
add_executable(gen_small_file gen_small_file.cpp)
add_executable(trans_vec transform_vec.cpp)
add_executable(check_recall check_recall.cpp)



add_executable(search_disk_hybrid search_disk.cpp)
target_include_directories(search_disk_hybrid PRIVATE "../src" ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(search_disk_hybrid PRIVATE gustann)
target_compile_options(search_disk_hybrid
  PRIVATE "-DHYBRID_CALC"
  PRIVATE "-O2" "-g"
)
set_target_properties(search_disk_hybrid PROPERTIES LINKER_LANGUAGE CUDA)

IF(GUSTANN_USE_BAM)
  message("ENABLE BAM")
  
  add_executable(search_disk search_disk.cpp)
  target_include_directories(search_disk PRIVATE "../src" ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(search_disk PRIVATE gustann)
  target_compile_options(search_disk
    #PRIVATE "-DASYNC_READ"
    PRIVATE "-D_USE_BAM"
    PRIVATE "-O2" "-g"
  )
  set_target_properties(search_disk PROPERTIES LINKER_LANGUAGE CUDA)
ENDIF()

add_executable(search_mem search_disk.cpp)
target_include_directories(search_mem PRIVATE "../src" ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(search_mem PRIVATE gustann)
target_compile_options(search_mem
  PRIVATE "-DUSE_PURE_MEM"
  PRIVATE "-O2" "-g"
)
set_target_properties(search_mem PROPERTIES LINKER_LANGUAGE CUDA)

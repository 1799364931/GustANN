# Helper programs

add_executable(split_file split_file.cpp)
add_executable(analyze analyze.cpp)
add_executable(gen_small_file gen_small_file.cpp)
add_executable(trans_vec transform_vec.cpp)
add_executable(check_recall check_recall.cpp)



add_executable(search_disk_hybrid search_disk.cu)
target_include_directories(search_disk_hybrid PRIVATE "../src")
target_link_libraries(search_disk_hybrid PRIVATE gustann)
target_compile_options(search_disk_hybrid
  PRIVATE "-lineinfo"
  PRIVATE "-DHYBRID_CALC"
  PRIVATE "-O2" "-g"
)

IF(GUSTANN_USE_BAM)
  message("ENABLE BAM")
  
  add_executable(search_disk search_disk.cu)
  target_include_directories(search_disk PRIVATE "../src")
  target_link_libraries(search_disk PRIVATE gustann)
  target_compile_options(search_disk
    #PRIVATE "-DASYNC_READ"
    PRIVATE "-D_USE_BAM"
    PRIVATE "-lineinfo"
    PRIVATE "-O2" "-g"
  )
ENDIF()

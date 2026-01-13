#pragma once
#include <vector>
#include <cstdint>
#include <string>

struct NavGraph {
  std::vector<int> mapping;
  uint8_t* data;
  uint8_t* data_dev;
  int* graph;
  int* graph_dev;
  int data_len;
  int num_node;
  int start;
  int max_m;
  
  void init(std::string index_file,
            std::string data_file,
            std::string map_file);

  void translate(int* entry, int qcnt);
};

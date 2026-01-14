#include "nav_graph.hpp"
#include "common.hpp"
#include "common_cuda.cuh"
#include <cassert>
#include <algorithm>

namespace gustann {
#ifdef FLOAT_DATA
  const int data_size = 4;
#else
  const int data_size = 1;
#endif

  void NavGraph::init
  (std::string index_file,
   std::string data_file,
   std::string map_file) {
    FILE* f = fopen(data_file.c_str(), "r");
    fread(&num_node, sizeof(int), 1, f);
    fread(&data_len, sizeof(int), 1, f);
    data = new uint8_t [1ll * num_node * data_len * data_size];
  
    fread(data, 1, 1ll * num_node * data_len * data_size, f);
 
    fclose(f);
    f = fopen(map_file.c_str(), "r");

    for (int i = 0; i < num_node; i++) {
      int x;
      assert(fscanf(f, "%d", &x) == 1);
      mapping.push_back(x);
    }
 

    fclose(f);
    f = fopen(index_file.c_str(), "r");
    int dummy;
    fread(&dummy, sizeof(int), 1, f);
    fread(&dummy, sizeof(int), 1, f);
    fread(&max_m, sizeof(int), 1, f);
    fread(&start, sizeof(int), 1, f);
    fread(&dummy, sizeof(int), 1, f);
    fread(&dummy, sizeof(int), 1, f);
    graph = new int [1ll * num_node * max_m];
    //  printf("!!!%d %d\n", max_m, start);

    assert(0 < max_m && max_m <= 32);
    for (int i = 0; i < num_node; i++) {
      int d;
      fread(&d, 1, sizeof(int), f);
      assert(0 < d && d <= max_m);
      fread(graph + max_m * i, sizeof(int), d, f);
      //std::random_shuffle(graph + max_m * i, graph + max_m * i + d);
      for (int j = d; j < max_m; j++) graph[max_m * i + j] = -1;
    }
  

    fclose(f);
    copy_to_dev(data, data_dev, 1l * num_node * data_len * data_size);
    copy_to_dev(graph, graph_dev, 1l * max_m * num_node);
  }



  void NavGraph::translate(int* entry, int qcnt) {
    for (int i = 0; i < qcnt; i++) {
      //printf("%d %d\n", entry[i], mapping[entry[i]]);
      entry[i] = mapping[entry[i]];
    }
  }
}

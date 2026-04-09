# GustANN

GustANN is a high-throughput billion-scale graph-based vector store on GPU, based on our SIGMOD'26 paper: [*High-Throughput, Cost-Effective Billion-Scale Vector Search with a Single GPU.*](https://dl.acm.org/doi/10.1145/3769799)

**Features:**
+ **High Throughput**: ~250K QPS for billion-scale dataset (SIFT1B, top-10, recall=0.9), 7.81x of DiskANN
+ **Memory Efficient**: ~40GB memory usage for both GPU and CPU on billion-scale dataset.
+ **Flexible Interface**: supports flexible search-mode (SSD-based, all-in-DRAM, all-in-GPU) with multiple storage backends (SPDK, io-uring, aio).



## Quickstart

You can quickly set up a 1M vector database to try GustANN using the script `scripts/quick_start.sh`.

### Basic Configurations

+ CPU: X86 CPU supporting huge page (You may verify this through `grep pdpe1gb /proc/cpuinfo`), 
+ DRAM: ~40GB for vector search. Additional memory space is needed for building the index.
+ GPU: ~40GB GPU memory for billion-scale vector search (e.g., NVIDIA A100)
+ Vector dataset: less than 2B vectors to avoid integer overflow, each record size (`vector_size + 4 + 4 * num_neighbors`) is less than 4KB. 
+ SSD (optional): ~700GB for SIFT and ~1TB for DEEP (both containing 1B vectors). Multiple SSDs are supported. 
  - We suggest to use SPDK to achieve best performance of GustANN. 
  - Alternatively, you can use other backend (io-uring or aio), or in-memory index. 

### Software Dependencies

We use [DiskANN](https://github.com/microsoft/DiskANN) to build the vector index. 
To build DiskANN, install the following dependencies (for Ubuntu 22.04):

``` shell-session
# apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev libjemalloc-dev
```

Also, install CUDA according to the instruction from [NVIDIA](https://developer.nvidia.com/cuda-downloads).

Other dependencies of GustANN is listed in `deps/` directory.

### Build the Repository

First, clone the repository:

``` shell-session
$ git clone https://github.com/thustorage/GustANN.git --recursive
$ cd GustANN
```

Then, build DiskANN (for index building only):

``` shell-session
$ cd deps/DiskANN
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ cd ../../..
```

Finally, build GustANN:

``` shell-session
$ mkdir -p build
$ cd build
$ cmake ..
$ make -j
$ cd ..
```

To build different storage backend, you can turn on the switch `-DCMAKE_USE_{SPDK,URING,AIO}=ON`

## Dataset and Index Preparation

For complete dataset preparation instructions, you may refer to [PipeANN's repository](https://github.com/thustorage/PipeANN?tab=readme-ov-file#for-others-starting-from-scratch). 
Note that PipeANN uses a different argument format to DiskANN.

### Build DiskANN Index

**If you have built the index, please skip this step.**

To build a DiskANN index, you need to prepare a dataset in `bin` format.
To convert the dataset, DiskANN provides some utilities to convert from `bvec/fvec`(format that SIFT dataset uses):

``` shell-session
$ ./deps/DiskANN/build/apps/utils/fvecs_to_bin <float/uint8> input_vecs output_bin
```

Then, you can build the index using the following command:

``` bash
$ ./deps/DiskANN/build/apps/build_disk_index --data_type uint8/float --dist_fn l2 --index_path_prefix <index_prefix> --data_path <dataset_file> -B <pq_size> -M <memory> -R 128 -L 200 
```

The key arguments are specified like this:

+ `index_prefix`: the directory and the name of the index. For example, if you use `/data/index`, then DiskANN will create index files with this prefix (e.g., `/data/index_disk.index`).
+ `dataset_file`: the dataset in `bin` format
+ `pq_size`: Size of the compressed product quantilization (PQ) vectors. Type 3.3 for 100M-scale datasets, 33 for 1B-scale datasets. This setting will generate 32-bit PQ vectors.
+ `memory`: The maximum memory available for building the index. 

Alternatively, after modifying the `scripts/setup.sh`, you can also execute the script:

``` shell-session
$ ./scripts/build_disann_index.sh <pq_size> <memory>
```

### Prepare GustANN Index

In addition to the original DiskANN index, GustANN needs the build a pivot graph.

We have provided scripts to build the pivot graph easily.
Please modify the `scripts/setup.sh` according to the instruction in it, and run:

``` shell-session
$ ./scripts/gen_pivot_graph.sh
```


## Run In-memory GustANN

**For small datasets (e.g., < 100M vectors), we suggest to store the dataset in DRAM or in GPU memory to achieve better performance.**

``` shell-session
$ ./build/bin/search_mem \
      --query <query_file> --index <index_file> --ground_truth <ground_truth> \
      --pq_data <pq_file> --nav_graph <nav_graph> --data_type <data_type> \
      --topk <topk> --ef_search <L> -R <R> [-G]
```

The meanings of each arguments are shown as follows:

+ `query_file`: The query vectors (in `bvecs`/`fvecs` format)
+ `index_file`: The DiskANN index
+ `ground_truth`: The ground truth (in `ivecs` format)
+ `pq_file`: The product quantilization (PQ) of all vectors (only need to type `<prefix>_pq`)
+ `nav_graph`: The additional GustANN index (the `nav/` directory)
+ `data_type`: `uint8` for SIFT dataset, `float` for DEEP dataset. For other datasets, please refer to their documents. Only these two data types are supported.
+ `topk`: How many top-k vectors are searched
+ `L`: How many vectors are stored during the search (The higher, the more accurate)
+ `R`: Repeat the query `R` times. Set it to greater than 1 for a more accurate throughput benchmark, if the query set is small.
+ `G`: When turned on, it is searched purely on the GPU. Best performance, but only for small datasets.

Alternatively, after modifying the `scripts/setup.sh`, you can also execute the script:

``` shell-session
$ ./scripts/run_mem.sh --topk <topk> --ef_search <L> -R <R> [-G]
```

### Run on-SSD GustANN

You can use SPDK (fastest, supports multi-SSD), liburing or libaio.

### Use SPDK

To use SPDK, **root privillege is needed**. There should be no partitions or filesystems on SSDs. You may use `nvme format` to format the disk. **This will erase all data on the disk. Do this at your own risk!**

#### Build SPDK

```
$ git clone https://github.com/spdk/spdk deps/spdk # we have tested GustANN on git commit 7c0720d1d
$ cd deps/spdk
$ sudo scripts/pkgdep.sh # Install the dependency of SPDK
$ ./configure
$ make -j
$ cd ../..
```

Then rebuild GustANN with `-DCMAKE_USE_SPDK=ON`

#### Setup SPDK

``` shell-session
# ./deps/spdk/scripts/setup.sh # Setup SPDK Environment
# ./deps/spdk/build/examples/hello_world # To check whether SPDK works fine
```

Ideally, you will see outputs similar to this:

``` plain
Attaching to 0000:8b:00.0
Attaching to 0000:8d:00.0
Attaching to 0000:8e:00.0
Attached to 0000:8d:00.0
  Namespace ID: 1 size: 3840GB
Attached to 0000:8e:00.0
  Namespace ID: 1 size: 3840GB
Attached to 0000:8b:00.0
  Namespace ID: 1 size: 3840GB
Initialization complete.
INFO: using host memory buffer for IO
Hello world!
INFO: using host memory buffer for IO
Hello world!
INFO: using host memory buffer for IO
Hello world!
```

Collect all PCIe addresses for the SSDs you want you use in the format of XXXX:XX:XX.X, and write them into a file (`ssd_list.txt` for instance):

```plain
0000:8b:00.0
0000:8d:00.0
0000:8e:00.0
```

#### Write Index to SSD

Then, write the index contents into the SSD using the following utility:

``` shell-session
# ./build/spdk/spdk_write <index_file> <ssd_list>
```

The `index_file` is the DiskANN index file (`<prefix>_disk.index`), `ssd_list` is the SSD list collected in the previous step.

Alternatively, after modifying the `scripts/setup.sh`, you can also execute the script:

``` shell-session
# ./scripts/write_spdk.sh
```

### Setup LibUring

#### Download and build

``` bash
git clone https://github.com/axboe/liburing.git deps/liburing # We have tested on commit 20b3fe67
cd deps/liburing
./configure && make -j
cd ../..
```

Then rebuild GustANN with `-DCMAKE_USE_URING=ON`

### Run GustANN

``` shell-session
# ./build/bin/search_disk_hybrid --query <query_file> --index <index_file> --ground_truth <ground_truth> --pq_data <pq_file> --nav_graph <nav_graph> --data_type <data_type> --io_backend [spdk/uring/aio/memory] --topk <topk> --ef_serach <L> -B <B> -T <T> -C <C> -R <R> --ssd_list_file <ssd_list>
```


The meanings of the arguments are shown as follows:

+ `query_file`: The query vectors (in `bvecs`/`fvecs` format)
+ `index_file`: The DiskANN index
+ `ground_truth`: The ground truth (in `ivecs` format)
+ `pq_file`: The product quantilization (PQ) of all vectors (only need to type `<prefix>_pq`)
+ `nav_graph`: The additional GustANN index (the `nav/` directory)
+ `data_type`: `uint8` for SIFT dataset, `float` for DEEP dataset. For other datasets, please refer to their documents.
+ `topk`: How many top-k vectors are searched
+ `L`: How many vectors are stored during the search (The higher, the more accurate)
+ `B`: The minibatch size (1120 in the evaluation)
+ `T`: How many worker threads (2 in the evaluation)
+ `C`: How many minibatches for each thread (20 in the evaluation)
+ `R`: Repeat the query `R` times. Set it to greater than 1 for a more accurate throughput benchmark, if the query set is small.
+ `io_backend`: The IO backend GustANN uses. You can also setup in-memory search, which is a little slower than the mode mentioned before.
+ `ssd_list`: The SSD list file. (SPDK only)

Different I/O backends favor different worker configurations:
+ SPDK: `T=2 C=20 B>=1000`
+ IOUring/memory: `T=20 C=1 B>=1000`
+ AIO: `T=2 C=10 B=256`
You may adjust these settings based on the performace of the SSDs.

After the search finishes, the runtime, total SSD I/Os, and the recall will be printed on the stdout.

Alternatively, after modifying the `scripts/setup.sh`, you can also execute the script:

``` shell-session
# ./scripts/run_spdk.sh --topk <topk> --ef_serach <L> -B <B> -T <T> -C <C> -R <R> # for SPDK
$ ./scripts/run_uring.sh --topk <topk> --ef_serach <L> -B <B> -T <T> -C <C> -R <R> # for IOUring
$ ./scripts/run_aio.sh --topk <topk> --ef_serach <L> -B <B> -T <T> -C <C> -R <R> # for AIO
$ ./scripts/run.sh --topk <topk> --ef_serach <L> -B <B> -T <T> -C <C> -R <R> # for in-memory search
```

### GPU Direct-IO Support (Experimental)

See [bam.md](bam.md)

## Paper

If you find GustANN useful, please cite our paper:

``` bibtex
@inproceedings{sigmod26gustann,
author = {Haodi Jiang and Hao Guo and Minhui Xie and Jiwu Shu and Youyou Lu},
title = {{High-Throughput, Cost-Effective Billion-Scale Vector Search with a Single GPU}},
year = {2026},
publisher = {Association for Computing Machinery},
booktitle = {Proceedings of the 2026 International Conference on Management of Data},
address = {Bengaluru, India},
series = {SIGMOD '26}
}
```

## Acknowledgement

Some GPU kernel implementations are from [CuHNSW](https://github.com/js1010/cuhnsw). We really appreciate it.



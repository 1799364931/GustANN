# GPU Direct-IO Support

Note: this setup is mainly for research purpose.

This document shows how to run GustANN with BaM GPU-SSD direct access engine[^1].

## Build

First, prepare BaM as shown in the [repository](https://github.com/ZaidQureshi/bam).

``` shell-session
$ git clone https://github.com/ZaidQureshi/bam.git deps/bam
$ # Then build BaM and its Linux kernel module as instructed in original repo
```

We tested BaM with git commit hash `2edd33c`. You may checkout this version. 
To compile and link successfully, you may need to add an `inline` at the beginning of line 584 in `dep/bam/include/page_cache.h`

Then, to enable BaM in GustANN, please recompile GustANN with:

``` shell-session
$ # in ./build
$ cmake -DGUSTANN_USE_BAM=ON ..
$ make -j
```

## Run GustANN-BaM

**Root previllege is required!**

``` shell-session
# ./bin/search_disk --query <query_file>  --index <index_file> --ground_truth <ground_truth>   --pq_data <pq_file> --nav_graph <nav_graph> --data_type <data_type> --topk <topk> --ef_search <L> --num_ctrls <num_ctrls> --cache_page <cache_pages> [--copy_data] [--copy_only] 
```

Most of the arguments are the same as original GustANN, here lists the difference:

+ `num_ctrls`: Number of SSDs to use. By default, BaM maps SSDs to `/dev/libnvm*`. If it is not the case, you can modify line 29 in `src/bam.cu`.
+ `cache_pages`: The number of SSD pages cached in GPU Memory.
+ `copy_data`: Copies the DiskANN index in `index_file` to SSD.
+ `copy_only`: Only copies index to SSD without searching it.

## Reference

[^1]: Zaid Qureshi, Vikram Sharma Mailthody, Isaac Gelado, Seungwon Min, Amna Masood, Jeongmin Park, Jinjun Xiong, C. J. Newburn, Dmitri Vainbrand, I-Hsin Chung, Michael Garland, William Dally, and Wen-mei Hwu. 2023. GPU-Initiated On-Demand High-Throughput Storage Access in the BaM System Architecture. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS 2023). Association for Computing Machinery, New York, NY, USA, 325–339. https://doi.org/10.1145/3575693.3575748

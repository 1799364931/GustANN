# Advanced Usage & Detailed Configuration

This document provides under-the-hood manual commands and exhaustive parameter definitions for **GustANN**. It is intended for advanced users who prefer a **step-by-step approach** instead of using the automated `scripts/` directory, or for those who need to debug the system manually.

---

## 1. Manual Index Preparation

If you do not want to use `scripts/build_disann_index.sh`, you can execute the DiskANN binary manually after converting your dataset (as described in the README).

Execute the following command to build the DiskANN index:

```bash
./deps/DiskANN/build/apps/build_disk_index \
    --data_type <uint8/float> \
    --dist_fn l2 \
    --index_path_prefix <index_prefix> \
    --data_path <dataset_file> \
    -B <pq_size> \
    -M <memory> \
    -R 128 -L 200 
```

**Parameter Breakdown:**
*   `data_type`: `uint8` for SIFT datasets, `float` for DEEP datasets.
*   `index_prefix`: The directory and base name for the index outputs. (e.g., `/data/index` will generate files like `/data/index_disk.index`).
*   `dataset_file`: Path to your dataset in `bin` format.
*   `pq_size`: Size of the compressed Product Quantization (PQ) vectors. Use `3.3` for 100M-scale datasets, or `33` for 1B-scale datasets (generates 32-bit PQ vectors).
*   `memory`: Maximum memory limit (in GB) available for building the index.

---

## 2. Manual Execution: In-Memory GustANN

For small datasets (< 100M vectors) or pure GPU search, you can manually run the in-memory binary bypassing `scripts/run_mem.sh`:

```bash
./build/bin/search_mem \
    --query <query_file> \
    --index <index_file> \
    --ground_truth <ground_truth> \
    --pq_data <pq_file> \
    --nav_graph <nav_graph> \
    --data_type <data_type> \
    --topk <topk> \
    --ef_search <L> [<L2> ... ] \
    -R <R> [-G]
```

**Argument Definitions:**
*   `query_file`: Path to the query vectors (must be in `bvecs`/`fvecs` format).
*   `index_file`: The DiskANN index file (usually `<index_prefix>_disk.index`).
*   `ground_truth`: Path to the ground truth file (in `ivecs` format).
*   `pq_file`: The PQ data of all vectors (You only need to provide the `<index_prefix>_pq`).
*   `nav_graph`: The additional GustANN index directory (usually the `nav/` directory).
*   `data_type`: `uint8` for SIFT, `float` for DEEP. *(Note: Only these two types are currently supported).*
*   `topk`: The number of nearest neighbors to search for.
*   `ef_search` (`L`): The search queue size. Higher values increase accuracy but reduce speed. You can input multiple `L` values separated by spaces to test different accuracies in a single run.
*   `-R <R>`: Repeat the queries `R` times. Useful for smoothing out throughput metrics if the query set is small.
*   `-G` (Optional): Forces pure GPU search. Yields the best performance but is strictly limited by GPU VRAM (only suitable for small datasets).

---

## 3. Manual Execution: On-SSD GustANN (Billion-Scale)

To run the full hybrid SSD-GPU pipeline manually without the scripts, use the `search_disk_hybrid` binary:

```bash
sudo ./build/bin/search_disk_hybrid \
    --query <query_file> \
    --index <index_file> \
    --ground_truth <ground_truth> \
    --pq_data <pq_file> \
    --nav_graph <nav_graph> \
    --data_type <data_type> \
    --io_backend <spdk|uring|aio|memory> \
    --topk <topk> \
    --ef_search <L> [<L2> ...] \
    -B <B> -T <T> -C <C> -R <R> \
    --ssd_list_file <ssd_list>
```

**Hardware-Specific Tuning Parameters:**
*   `-T <T>`: Number of worker threads managing I/O.
*   `-C <C>`: Number of minibatches allocated per thread.
*   `-B <B>`: Minibatch size.
*   `io_backend`: Defines the storage backend (`spdk`, `uring`, `aio`, or `memory`).
*   `ssd_list_file`: Text file containing PCIe addresses (Required ONLY for `spdk`).

*(For descriptions of `query_file`, `index`, `pq_data`, `L`, `R`, etc., please refer to Section 2 above).*

---

## 4. SPDK Setup Details

### 4.1 Verifying SPDK Initialization
When you run `sudo ./deps/spdk/build/examples/hello_world`, a successful SPDK environment setup will output text similar to this:

```text
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
```

### 4.2 Formatting `ssd_list.txt`
Based on the SPDK output above, collect the PCIe addresses into a raw text file (e.g., `ssd_list.txt`). Each line must strictly follow the `XXXX:XX:XX.X` format:

```text
0000:8b:00.0
0000:8d:00.0
0000:8e:00.0
```

### 4.3 Manually Writing Index to SSD via SPDK
The DiskANN index must be directly written to the raw SSDs bypassing the filesystem. To do this manually:

```bash
sudo ./build/spdk/spdk_write <index_file> <ssd_list_file>
```
*   `index_file`: e.g., `/data/index_disk.index`
*   `ssd_list_file`: The path to the text file created in step 4.2.


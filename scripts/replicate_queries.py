import io_utils
import sys
import numpy as np

# usage:
# python3 replicate_queries.py <original_data> <new_data> <replicate_times>
# It repeats all vectors in the original file k times, in the format <replica 1> <replica 2> ... <replica k>.

# A key note is that original queries have two types: *vecs, *bin (i.e., ivecs, fvecs, bbin, fbin), you need to use the corresponding reader function in io_utils
# A special format is `bin`, please look at deps/DiskANN/apps/utils/compute_groundtruth.cpp for its details.

READERS = {
    'fvecs': io_utils.fvecs_read,
    'ivecs': io_utils.ivecs_read,
    'bvecs': io_utils.bvecs_read,
    'fbin':  io_utils.fbin_read,
    'ibin':  io_utils.ibin_read,
    'bbin':  io_utils.bbin_read,
}

WRITERS = {
    'fvecs': io_utils.fvecs_write,
    'ivecs': io_utils.ivecs_write,
    'bvecs': io_utils.bvecs_write,
    'fbin':  io_utils.fbin_write,
    'ibin':  io_utils.ibin_write,
    'bbin':  io_utils.bbin_write,
}

def get_ext(filename):
    return filename.rsplit('.', 1)[-1]

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 replicate_queries.py <original_data> <new_data> <replicate_times>")
        sys.exit(1)

    original_file = sys.argv[1]
    new_file = sys.argv[2]
    k = int(sys.argv[3])

    in_ext = get_ext(original_file)
    out_ext = get_ext(new_file)

    if in_ext == 'bin':
        ids, dists = io_utils.gt_bin_read(original_file)
        print(f"Read groundtruth bin: {ids.shape[0]} queries, {ids.shape[1]} neighbors from {original_file}")
        ids_rep = np.tile(ids, (k, 1))
        dists_rep = np.tile(dists, (k, 1))
        print(f"Replicated to {ids_rep.shape[0]} queries, writing to {new_file}")
        io_utils.gt_bin_write(new_file, ids_rep, dists_rep)
    else:
        if in_ext not in READERS:
            print(f"Unsupported input format: {in_ext}")
            sys.exit(1)
        if out_ext not in WRITERS:
            print(f"Unsupported output format: {out_ext}")
            sys.exit(1)

        vecs = READERS[in_ext](original_file)
        print(f"Read {vecs.shape[0]} vectors of dim {vecs.shape[1]} from {original_file}")

        replicated = np.tile(vecs, (k, 1))
        print(f"Replicated to {replicated.shape[0]} vectors, writing to {new_file}")

        WRITERS[out_ext](new_file, replicated)

    print("Done.")

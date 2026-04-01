#!/bin/bash

HOME_DIR=$(readlink -f $(dirname $0)/..)
source $HOME_DIR/scripts/setup.sh

$HOME_DIR/build/bin/$EXEC --query $QUERY_FILE --index $INDEX_FILE --ground_truth $GT_FILE --pq_data $PQ_FILE --nav_graph $PIVOT_GRAPH --io_backend uring --data_type $DATA_TYPE $*

#!/bin/bash


HOME_DIR=$(readlink -f $(dirname $0)/..)
DISKANN_DIR=$HOME_DIR/deps/DiskANN

pushd $DISKANN_DIR
mkdir build
cd build
cmake ..
make -j$(nproc)

popd 

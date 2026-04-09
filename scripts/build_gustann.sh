#!/bin/bash

HOME_DIR=$(readlink -f $(dirname $0)/..)

pushd $HOME_DIR

mkdir build
cd build
cmake .. "$@"
make -j$(nproc)

popd

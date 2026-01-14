#!/bin/bash

HOME_DIR=$(readlink -f $(dirname $0)/..)
source $HOME_DIR/scripts/setup.sh

$HOME_DIR/build/src/io/spdk/spdk_write $INDEX_FILE $SSD_LIST

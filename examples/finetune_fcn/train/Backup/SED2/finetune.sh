#!/usr/bin/env sh
export HDF5_DISABLE_VERSION_CHECK=1
/home/liming/project/caffe/build/tools/caffe train --solver=./solver.prototxt --weights=../MSRA-DUT/train_iter_80000.caffemodel --gpu=1

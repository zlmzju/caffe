#!/usr/bin/env sh
export HDF5_DISABLE_VERSION_CHECK=1
/home/liming/project/caffe/build/tools/caffe train --solver=./solver.prototxt --weights=./surgery_net.caffemodel --gpu=1
#--weights=./train_iter_53000.caffemodel --gpu=1

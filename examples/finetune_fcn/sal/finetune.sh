#!/usr/bin/env sh
export HDF5_DISABLE_VERSION_CHECK=1
/home/liming/project/caffe/build/tools/caffe train --solver=./solver.prototxt --snapshot=./models/train_iter_3000.solverstate --gpu=1

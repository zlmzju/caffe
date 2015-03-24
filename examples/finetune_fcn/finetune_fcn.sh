#!/usr/bin/env sh
export HDF5_DISABLE_VERSION_CHECK=1
../../build/tools/caffe train --solver=fcn_solver.prototxt --weights=msra/fcn_iter_V3.caffemodel --gpu=1

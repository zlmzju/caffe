#!/usr/bin/env sh
/home/liming/project/caffe/build/tools/caffe train --solver=./solver.prototxt --weights=../../models/fcn_V4_MSRA9000.caffemodel --gpu=1

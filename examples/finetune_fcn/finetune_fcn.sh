#!/usr/bin/env sh
../../build/tools/caffe train --solver=fcn_solver.prototxt --weights=fcn/finetune_net.caffemodel

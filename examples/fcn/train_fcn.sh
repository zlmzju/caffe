#!/usr/bin/env sh
cd $HOME/project/caffe
./build/tools/caffe train --solver=examples/fcn/fcn-32s-solver.prototxt

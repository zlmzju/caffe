#!/usr/bin/env sh
cd $HOME/project/caffe
./build/tools/caffe test -model examples/fcn/fcn-32s-origin.prototxt -weights examples/fcn/fcn-32s.caffemodel -gpu 0

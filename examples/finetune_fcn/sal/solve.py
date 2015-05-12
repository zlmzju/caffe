import caffe

base_weights = '../vgg/vgg16fc.caffemodel'

# init
caffe.set_mode_gpu()
caffe.set_device(1)

solver = caffe.SGDSolver('solver.prototxt')

# copy base weights for fine-tuning
solver.net.copy_from(base_weights)

# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
solver.step(80000)
import caffe
import numpy as np
import matplotlib.pyplot as plt
# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    
## Load the original network and extract the fully connected layers' parameters.
#net = caffe.Net('../../models/bvlc_reference_caffenet/deploy.prototxt', 
#                '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', 
#                caffe.TEST)
#params = ['fc6', 'fc7', 'fc8']
## fc_params = {name: (weights, biases)}
#fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
#for fc in params:
#    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)
#
## Load the fully convolutional network to transplant the parameters.
#net_full_conv = caffe.Net('bvlc_caffenet_full_conv.prototxt', 
#                          '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
#                          caffe.TEST)
#params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
## conv_params = {name: (weights, biases)}
#conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}
#for conv in params_full_conv:
#    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)
#
#
#for pr, pr_conv in zip(params, params_full_conv):
#    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
#    conv_params[pr_conv][1][...] = fc_params[pr][1]
#net_full_conv.save('./bvlc_caffenet_full_conv.caffemodel')
net_full_conv = caffe.Net('bvlc_caffenet_full_conv.prototxt', 
                          './bvlc_caffenet_full_conv.caffemodel',
                          caffe.TEST)
print [(k, v.data.shape) for k, v in net_full_conv.blobs.items()]
# load input and configure preprocessing
im = caffe.io.load_image('../images/cat.jpg')
transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
transformer.set_mean('data', np.load('../../python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
# make classification map by forward and print prediction indices at each location
out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
print out['prob'][0].argmax(axis=0)
# show net input and confidence map (probability of the top prediction at each location)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
plt.subplot(1, 2, 2)

prob=out['prob'][0,:]
plt.imshow(prob[281,:])
plt.figure()
vis_square(prob)


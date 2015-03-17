#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import caffe
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
    
#
#def load_data(filename,W,H):
#    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#    transformer.set_transpose('data', (2,0,1))
#    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
#    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
#    net.blobs['data'].reshape(1,3,W,H)
#    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(IMAGE_FILE))

# Make sure that caffe is on the python path:
caffe_root = '/home/liming/project/caffe/'  # this file is expected to be in {caffe_root}/examples
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'fcn-32s-pascal-origin.prototxt'
PRETRAINED = 'fcn-32s-pascal-origin.caffemodel'
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,caffe.TEST)
print('\nblobs')
print [(k, v.data.shape) for k, v in net.blobs.items()]
print('\nparams')
print [(k, v[0].data.shape) for k, v in net.params.items()]

NEWMODEL_FILE = 'finetune_fcn.prototxt'
finetune_net = caffe.Classifier(NEWMODEL_FILE, PRETRAINED,
                       caffe.TEST)
print('\nparams')
print [(k, v[0].data.shape) for k, v in finetune_net.params.items()]

finetune_net.params['conv8_finetune'][0].data[0,:]=-0.05*net.params['score-fr'][0].data[0,:]
finetune_net.params['conv8_finetune'][1].data[0]=1+net.params['score-fr'][1].data[0]
finetune_net.params['deconv_finetune'][0].data[0,:]=net.params['upsample'][0].data[0,0,:]
finetune_net.params['deconv_finetune'][1].data[0]=net.params['upsample'][1].data[0]
finetune_net.save('finetune_net.caffemodel')
#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import caffe
#from time import clock

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
    plt.draw()

def load_data(filename,W=500,H=500):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    net.blobs['data'].reshape(1,3,W,H)
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(filename))
    net.forward()
    data = net.blobs['data'].data[0,:]
    map = net.blobs['outmap'].data[0,0,:]
    im=np.zeros([W,H,3])
    im=data.transpose((1,2,0))
    im-=im.min()
    im/=im.max()
    plt.figure(1)
    plt.imshow(im)
    plt.figure(2)
    plt.imshow(map)
    plt.draw()
    return map
# Make sure that caffe is on the python path:
caffe_root = '/home/liming/project/caffe/'  # this file is expected to be in {caffe_root}/examples

MODEL_FILE = 'fcn-32s-pascal-origin.prototxt'
PRETRAINED = 'fcn-32s-pascal-origin.caffemodel'

#NEWMODEL_FILE = 'finetune_fcn.prototxt'
#NEWPRETRAINED = './finetune_net.caffemodel'
NEWMODEL_FILE = '../msra/fcn_1024_sigmoid_deploy.prototxt'
NEWPRETRAINED = '../msra/fcn_V4_MSRA9000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Classifier(NEWMODEL_FILE, NEWPRETRAINED,
                       caffe.TEST)

print [(k, v[0].data.shape) for k, v in net.params.items()]
IMAGE_FILE =caffe_root+'examples/images/cat.jpg'
map=load_data(IMAGE_FILE);#'/home/liming/project/dataset/VOC/JPEGImages/000033.jpg')

#map=map-map.min()
#map=map/map.max()
#map=np.ceil(map*255)
#plt.imshow(map)
#plt.imsave('map.png',map,cmap='gray')
#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import caffe
import cv2
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
    plt.imshow(data,cmap='hot')
    plt.draw()


NEWMODEL_FILE = '../train/MSRA9000/deploy.prototxt'
NEWPRETRAINED = '../train/MSRA9000/models1/train_iter_100000.caffemodel'
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Classifier(NEWMODEL_FILE, NEWPRETRAINED,
                       caffe.TEST)

#print [(k, v[0].data.shape) for k, v in net.params.items()]
print('\nblobs')
print [(k, v.data.shape) for k, v in net.blobs.items()]

path='/home/liming/project/iccv15/ICCV_EXP/Dataset/ECSSD/Images/'
filename=path+'0007.jpg'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.array([ 104.00698793,  116.66876762,  122.67891434])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(filename))
net.forward()
data = net.blobs['data'].data[0,:]
map = net.blobs['outmap'].data[0,0,:]
im=transformer.deprocess('data',data)
plt.figure(1)
plt.imshow(im)
plt.figure(2)
plt.imshow(map)#,cmap='gray')


#map=np.ceil(map*255)
#plt.imshow(map)
#plt.imsave('map.png',map,cmap='gray')
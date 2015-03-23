#!/usr/bin/env python
import numpy as np
#import matplotlib.pyplot as plt
import caffe
import cv2
import glob
import scipy.io as sio
# Make sure that caffe is on the python path:
caffe_root = ''  # this file is expected to be in {caffe_root}/examples
W=500
H=500
batchSize=10
# save map dir
MAP_DIR='/home/liming/project/dataset/DUT/DUT_MAT/'
IMG_DIR='/home/liming/project/dataset/DUT/DUT_image/'
# All file list
fileList=glob.glob(IMG_DIR+'*.jpg')
fileNum=len(fileList)
fileSize=np.zeros((batchSize,2),dtype=np.int)
imgData=np.zeros((batchSize,3,W,H))
# init the net from model    
NEWMODEL_FILE = 'surgery_net.prototxt'
NEWPRETRAINED = './fcn_iter_39000.caffemodel'
caffe.set_device(1)
caffe.set_mode_gpu()
net = caffe.Classifier(NEWMODEL_FILE, NEWPRETRAINED,caffe.TEST)
print [(k, v[0].data.shape) for k, v in net.params.items()]
# forward to create output map
net.blobs['data'].reshape(batchSize,3,W,H)
# process each image
for fileIdx in range(fileNum):
    fileName=fileList[fileIdx]
    img=caffe.io.load_image(fileName)
    (fileSize[fileIdx%batchSize,0],fileSize[fileIdx%batchSize,1],C)=img.shape   #C=3
    transformer = caffe.io.Transformer({'data': [fileNum,3,W,H]})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load('/home/liming/project/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)
    imgData[fileIdx%batchSize,:,:]=transformer.preprocess('data',img)
    if (fileIdx%batchSize)==(batchSize-1):
        net.blobs['data'].data[...] = imgData
        net.forward()
        print fileIdx
        for mapIdx in range(batchSize):
            map=net.blobs['map'].data[mapIdx,0,:,:]
            map=cv2.resize(map,(fileSize[mapIdx,1],fileSize[mapIdx,0]))
            globalIdx=mapIdx+fileIdx-batchSize+1
            sio.savemat(MAP_DIR+'%s.mat'%fileList[globalIdx][len(IMG_DIR):-4],{'deepMap':map.astype(np.float64)})
            map2=map
            map2-=map2.min()
            map2/=map2.max()
            map2=np.ceil(map2*255)
            cv2.imwrite(IMG_DIR+'%s.png'%fileList[globalIdx][len(IMG_DIR):-4],map2)

#data = net.blobs['data'].data[1,:]
#map = net.blobs['map'].data[1,0,:]
#im=np.zeros([W,H,3])
#im=data.transpose((1,2,0))
#im-=im.min()
#im/=im.max()
#plt.figure(1)
#plt.imshow(im)
#plt.figure(2)
#plt.imshow(map)
#plt.draw()
#map=load_data(img);#'/home/liming/project/dataset/VOC/JPEGImages/000033.jpg')
#map2=cv2.resize(map,(img.shape[1],img.shape[0]))
#
#map=map-map.min()
#map=map/map.max()
#map=np.ceil(map*255)
#plt.imshow(map,cmap='gray')
##plt.imsave('map.png',map,cmap='gray')
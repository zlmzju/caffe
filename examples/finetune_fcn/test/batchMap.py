#!/usr/bin/env python
import numpy as np
#import matplotlib.pyplot as plt
import caffe
import cv2
import glob
import scipy.io as sio
import time
# Make sure that caffe is on the python path:
caffe_root = ''  # this file is expected to be in {caffe_root}/examples
W=500
H=500
batchSize=15
all_start=time.clock()
all_forward_time=0.0
all_file_numbers=0
# init the net from model    
NEWMODEL_FILE = '../train/MSRA9000/deploy.prototxt'
NEWPRETRAINED = '../train/MSRA9000/models1/train_iter_100000.caffemodel'
caffe.set_device(1)
caffe.set_mode_gpu()
net = caffe.Classifier(NEWMODEL_FILE, NEWPRETRAINED,caffe.TEST)
print [(k, v[0].data.shape) for k, v in net.params.items()]
net.blobs['data'].reshape(batchSize,3,W,H)
datasets=['DUT-OMRON','ECSSD','MSRA1000','PASCAL-S','SED2','SOD','THUR']#['DUT-OMRON','ECSSD','MSRA1000','PASCAL-S','SED2','SOD','THUR','THUS']
for DATASET in datasets:
    #DATASET='MSRA1000'
    ROOTDIR='/mnt/ftp/project/Saliency/ICCV_EXP/'
    MAP_DIR=ROOTDIR+'Result/'+DATASET+'/DeepMap/OurBR/'
    IMG_DIR=ROOTDIR+'Dataset/'+DATASET+'/Images/'
    # All file list
    fileList=glob.glob(IMG_DIR+'*.jpg')
    fileNum=len(fileList)
    fileSize=np.zeros((batchSize,2),dtype=np.int)
    imgData=np.zeros((batchSize,3,W,H))
    # process each image
    print DATASET
    for fileIdx in range(fileNum):
        fileName=fileList[fileIdx]
        img=caffe.io.load_image(fileName)
        (fileSize[fileIdx%batchSize,0],fileSize[fileIdx%batchSize,1],C)=img.shape   #C=3
        transformer = caffe.io.Transformer({'data': [fileNum,3,W,H]})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.array([ 104.00698793,  116.66876762,  122.67891434])) # mean pixel
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0)) 
        imgData[fileIdx%batchSize,:,:]=transformer.preprocess('data',img)
        if (fileIdx==(fileNum-1)) or (fileIdx%batchSize)==(batchSize-1):
            leftNum=(fileIdx%batchSize)+1
            net.blobs['data'].reshape(leftNum,3,W,H)
            net.blobs['data'].data[...] = imgData[0:leftNum,:,:]
            start_time=time.clock()
            net.forward()
            stop_time=time.clock()
            all_file_numbers+=leftNum
            all_forward_time+=(stop_time-start_time)
            print fileIdx
            for mapIdx in range(leftNum):
                map=net.blobs['outmap'].data[mapIdx,0,:,:]
                map=cv2.resize(map,(fileSize[mapIdx,1],fileSize[mapIdx,0]))
                globalIdx=mapIdx+fileIdx-leftNum+1
                matFile=MAP_DIR+'MAT/%s.mat'%fileList[globalIdx][len(IMG_DIR):-4]
                sio.savemat(matFile,{'deepMap':map.astype(np.float64)})
                #map2=map
                #map2-=map2.min()
                #map2/=map2.max()
                #map2=np.ceil(map2*255)
                #mapFile=MAP_DIR+'MAP/%s.png'%fileList[globalIdx][len(IMG_DIR):-4]
                #print mapFile
                #cv2.imwrite(mapFile,map2)
# all files run time with save the mat file
all_stop=time.clock()
print "all run time:"
print (all_stop-all_start)

print "average time of net process an image:"
print (all_forward_time/all_file_numbers)

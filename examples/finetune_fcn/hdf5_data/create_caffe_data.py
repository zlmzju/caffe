# -*- coding: utf-8 -*-
import caffe
import os
import h5py
import numpy as np
import cv2
def process_image(imageDir,filelist,M,N):
    with open(filelist) as f:
        content1 = f.readlines()
    content=content1
    imageData=np.zeros([len(content),3,M,N]);
    idx=0
    for string in content:
        filename=imageDir+string[0:-1]
        transformer = caffe.io.Transformer({'data': (1,3,M,N)})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.array([104.00698793,  116.66876762,  122.67891434])) # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        image=transformer.preprocess('data', caffe.io.load_image(filename))
        imageData[idx,:]=image
        idx=idx+1
    return imageData
    
def process_map(imageDir,filelist,M,N):
    with open(filelist) as f:
        content2 = f.readlines()
    content=content2
    inputMap=np.zeros([len(content),1,M,N]);
    idx=0
    for string in content:
        filename=imageDir+string[0:-1]
        image=cv2.imread(filename,0)
        image=cv2.resize(image,(M,N))
        image[image>0]=1
        inputMap[idx,:]=image
        idx=idx+1
    return inputMap


M=500
N=500
file_dir='/home/liming/project/dataset/VOC/'
trainData=process_image(file_dir+'JPEGImages/',file_dir+'script/traindata.txt' ,M,N)
trainMap=process_map(file_dir+'label/',file_dir+'script/trainlabel.txt',M,N)

testData=process_image(file_dir+'JPEGImages/',file_dir+'script/testdata.txt' ,M,N)
testMap=process_map(file_dir+'label/',file_dir+'script/testlabel.txt',M,N)
#file_dir='/home/liming/project/dataset/MSRA/MSRA10K_Imgs_GT/'
#trainData=process_image(file_dir,file_dir+'trainImage.txt' ,M,N)
#trainMap=process_map(file_dir,file_dir+'trainMap.txt',M,N)
#
#testData=process_image(file_dir,file_dir+'testImage.txt' ,M,N)
#testMap=process_map(file_dir,file_dir+'testMap.txt',M,N)

with h5py.File(os.path.dirname(__file__) + '/train.h5', 'w') as f:
    f['data'] = trainData
    f['inputmap'] = trainMap
with h5py.File(os.path.dirname(__file__) + '/test.h5', 'w') as f:
    f['data'] = testData
    f['inputmap'] = testMap

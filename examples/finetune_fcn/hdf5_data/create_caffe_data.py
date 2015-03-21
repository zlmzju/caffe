# -*- coding: utf-8 -*-
import caffe
import os
import h5py
import numpy as np
import cv2
batchSize=500
def process(imageDir,filelist,prefix,M,N):
    with open(filelist) as f:
        content1 = f.readlines()
    content=content1
    imageData=np.zeros([batchSize,3,M,N]);
    inputMap=np.zeros([batchSize,1,M,N]);
    idx=0
    for string in content:
        filename=imageDir+string
        if filename[-1]=='\n':
            filename=filename[0:-1]
        mapname=filename[0:-4]+'.png'
        #image
        transformer = caffe.io.Transformer({'data': (1,3,M,N)})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.array([104.00698793,  116.66876762,  122.67891434])) # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        image=transformer.preprocess('data', caffe.io.load_image(filename))
        #map
        map=cv2.imread(mapname,0)
        map=cv2.resize(map,(M,N))
        map[map>0]=1
        #imageData and inputMap
        imageData[idx%batchSize,:]=image
        inputMap[idx%batchSize,:]=map
        if (idx%batchSize)>=(batchSize-1):  #never reach >
            with h5py.File(os.path.dirname(__file__) + '/'+prefix+'%d_MSRA.h5'%np.ceil(idx/batchSize), 'w') as f:
                f['data'] = imageData
                f['inputmap'] = inputMap
        idx=idx+1
    return idx
    
M=500
N=500
file_dir='/home/liming/project/dataset/MSRA/MSRA10K_Imgs_GT/'
process(file_dir+'Imgs/',file_dir+'train.list','train',M,N)
process(file_dir+'Imgs/',file_dir+'test.list','test',M,N)

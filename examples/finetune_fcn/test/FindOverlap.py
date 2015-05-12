# -*- coding: utf-8 -*-
import glob
import Image,ImageChops
import numpy as np
ROOTDIR='/mnt/ftp/project/Saliency/ICCV_EXP/'

file_dirSOD=ROOTDIR+'Dataset/SOD/Images/'
SODlist=glob.glob(file_dirSOD+'*.jpg')

file_dirECSSD=ROOTDIR+'Dataset/ECSSD/Images/'
ECSSDlist=glob.glob(file_dirECSSD+'*.jpg')

for idx in range(len(SODlist)):
    im_SOD=Image.open(SODlist[idx])
    for idx2 in range(len(ECSSDlist)):
        im_ECSSD=Image.open(ECSSDlist[idx])
        if im_ECSSD.size!=im_SOD.size:
            continue
        im_Invert=ImageChops.invert(im_ECSSD)
        im_diff=Image.blend(im_SOD,im_ECSSD,0.5)
        x=np.matrix(im_diff)
        if x.sum()<10:
            print ECSSDlist[idx2]
        else:
            print '.'
            
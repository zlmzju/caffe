# -*- coding: utf-8 -*-
import shutil
dstDir='/mnt/ftp/project/Saliency/ICCV_EXP/Dataset/MSRA-B/'
srcDir='/mnt/ftp/project/Saliency/ICCV_EXP/Dataset/MSRA5000/'
f=open(dstDir+'/test.txt','r')
contents=f.readlines()
for filename in contents:
    srcImageName=srcDir+'Images/'+filename[0:-5]+'.jpg'
    dstImageName=dstDir+'Images/'+filename[0:-5]+'.jpg'
    shutil.copyfile(srcImageName,dstImageName)
    srcMapName=srcDir+'Groundtruth/'+filename[0:-5]+'.png'
    dstMapName=dstDir+'Groundtruth/'+filename[0:-5]+'.png'
    shutil.copyfile(srcMapName,dstMapName) 
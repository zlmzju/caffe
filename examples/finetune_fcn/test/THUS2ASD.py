import glob
import shutil
import os
ROOTDIR='/mnt/ftp/project/Saliency/ICCV_EXP/'
file_dir1=ROOTDIR+'Dataset/MSRA1000/Images/'
filelist=glob.glob(file_dir1+'*.jpg')
namelist=[]
for idx in range(len(filelist)):
      namelist.append(filelist[idx][len(file_dir1)+2:-4])

file_dir2=ROOTDIR+'Result/THUS/SaliencyMap/DRFI/'
namelist2=[]
for idx in range(len(namelist)):
    namelist2.append(namelist[idx][namelist[idx].find('_')+1:]+'_DRFI.png')

dest_dir=ROOTDIR+'Result/MSRA1000/SaliencyMap/DRFI/'
for idx in range(len(namelist2)):
    oldfilename=file_dir2+namelist2[idx]
    newfilename=dest_dir+filelist[idx][len(file_dir1):-4]+'.png'
    if not os.path.exists(oldfilename):
        print oldfilename
        continue
    if not os.path.exists(newfilename):
        shutil.copyfile(oldfilename,newfilename)
import glob
import shutil
import os
ROOTDIR='/mnt/ftp/project/Saliency/ICCV_EXP/'
file_dir0=ROOTDIR+'Dataset/MSRA1000/Images/'
filelist=glob.glob(file_dir0+'*.jpg')
namelist=[]
for idx in range(len(filelist)):
    startIdx=len(file_dir0);
    for i in range(len(filelist[idx])):
        if(filelist[idx][-1-i]=='_'):
            startIdx=len(filelist[idx])-i
            break
    namelist.append(filelist[idx][startIdx:])

file_dirSOD=ROOTDIR+'Dataset/SOD/Images/'
fileSODlist=glob.glob(file_dirSOD+'*.jpg')
SODlist=[]
for idx in range(len(fileSODlist)):
    startIdx=len(file_dirSOD);
    SODlist.append(fileSODlist[idx][startIdx:])
    
#file_dir1=ROOTDIR+'Dataset/THUS/Images/'
#filelist1=glob.glob(file_dir1+'*.jpg')
#THUSinASD=[]
#THUSinSOD=[]
#THUSnotASDnotSOD=[]
#for idx1 in range(len(filelist1)):
#    isInASD=0
#    isInSOD=0
#    for idx2 in range(len(namelist)):
#        if filelist1[idx1][len(file_dir1):]==namelist[idx2]:
#            isInASD=1
#            break
#    for idx2 in range(len(SODlist)):
#        if filelist1[idx1][len(file_dir1):]==SODlist[idx2]:
#            isInSOD=1
#            break
#    if isInASD:
#        THUSinASD.append(filelist1[idx1][len(file_dir1):]+'\n')
#    if isInSOD:
#        THUSinSOD.append(filelist1[idx1][len(file_dir1):]+'\n')
#    if isInASD and isInSOD:
#        print filelist1[idx1]
#    if (not isInASD) and (not isInSOD):
#        THUSnotASDnotSOD.append(filelist1[idx1][len(file_dir1):]+'\n')

#file_dir9='/home/liming/project/dataset/MSRA/MSRA9000/'
#for fileASD in THUSinASD:
#    fileASDname1=file_dir9+fileASD[0:-1]
#    fileASDname2=fileASDname1[0:-4]+'.png'
#    if os.path.isfile(fileASDname1):
#        os.remove(fileASDname1)
#        os.remove(fileASDname2)
#    else:
#        print 'no such file: %s'%fileASDname1
        

#f1=open('./THUSinASD.list','w')
#f1.writelines(THUSinASD)
#f1.close()
#f2=open('./THUSnotASD.list','w')
#f2.writelines(THUSnotASD)
#f2.close()
#f3=open('./THUSnotASDnotSOD.list','w')
#f3.writelines(THUSnotASDnotSOD)
#f3.close()

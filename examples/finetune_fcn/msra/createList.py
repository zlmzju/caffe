import glob
file_dir='/home/liming/project/dataset/MSRA/MSRA1000/'
filelist=glob.glob(file_dir+'*.jpg')
namelist=[]
for idx in range(len(filelist)):
      namelist.append(filelist[idx]+'\n')
               


#file_dir1='/home/liming/project/dataset/MSRA/MSRA10K_Imgs_GT/Imgs/'
#filelist1=glob.glob(file_dir1+'*.jpg')
#namelist1=[]
#for idx1 in range(len(filelist1)):
#    app=1
#    for idx2 in range(len(namelist)):
#        if namelist[idx2] == filelist1[idx1][len(file_dir1):len(filelist1[idx1])]:
#            app=0
#            break
#    if app:
#        namelist1.append(filelist1[idx1][len(file_dir1):len(filelist1[idx1])]+'\n')

f=open('/home/liming/project/dataset/MSRA/MSRA10K_Imgs_GT/MSRA1000.list','w')
f.writelines(namelist)
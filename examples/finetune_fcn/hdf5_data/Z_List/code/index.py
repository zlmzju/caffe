# USAGE
# python index.py --dataset images --shelve db.shelve

# import the necessary packages
from PIL import Image
import shelve
import glob
import sys
sys.path.append('/usr/lib/python2.7/site-packages/')
import imagehash
datasets=['THUS']#['DUT-OMRON','ECSSD','MSRA1000','PASCAL-S','SED2','SOD','THUR','THUS']
for DATASET in datasets:
    #DATASET='MSRA1000'
    ROOTDIR='/mnt/ftp/project/Saliency/ICCV_EXP/'
    IMG_DIR=ROOTDIR+'Dataset/'+DATASET+'/Images/'
    # open the shelve database
    db = shelve.open('./db/'+DATASET+'.shelve', writeback = True)

    # loop over the image dataset
    for imagePath in glob.glob(IMG_DIR + "*.jpg"):
        # load the image and compute the difference hash
        image = Image.open(imagePath)
        h = str(imagehash.dhash(image))
        # extract the filename from the path and update the database
        # using the hash as the key and the filename append to the
        # list of values
        filename = imagePath[imagePath.rfind("/") + 1:]
        db[h] = db.get(h, []) + [filename]
    
    # close the shelf database
    db.close()
    print './db/'+DATASET+'.shelve'
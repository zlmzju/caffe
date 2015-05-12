import shelve
def hamdist(str1,str2):
    diffs=0
    for ch1, ch2 in zip(str1,str2):
        if ch1 != ch2:
            diffs+=1
    return diffs
# open the shelve database
db1 = shelve.open('./db/PASCAL-S.shelve')
db2=shelve.open('./db/SED2.shelve')
# load the query image, compute the difference image hash, and
# and grab the images from the database that have the same hash
# value
namelist=[]
for query in db1:
    filenames = db1[query]
    if len(filenames)>1:
        print filenames
    if db2.has_key(query):
        for file in filenames:
            namelist.append('PASCAL-S:%11s'%file+'\t SED2:%11s'%db2[query][0]+'\n')
# close the shelve database
db1.close()
db2.close()
if len(namelist)>0:
    f=open('../overlap/SED2&PASCAL-S.txt','w')
    f.writelines(namelist)
    f.close()
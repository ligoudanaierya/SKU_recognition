import os
import shutil
import glob
import cv2

'''dirs =os.listdir('../workspace')
for i in range(62):
    #os.mkdir(str(i))
    if os.path.exists(str(i)+'/120.jpg'):
        os.remove(str(i)+'/120.jpg')
    if os.path.exists(str(i)+'/240.jpg'):
        os.remove(str(i)+'/240.jpg')
    if os.path.exists(str(i)+'/360.jpg'):
        os.remove(str(i)+'/360.jpg')
    if os.path.exists(str(i)+'/480.jpg'):
        os.remove(str(i)+'/480.jpg')
    if os.path.exists(str(i)+'/600.jpg'):
        os.remove(str(i)+'/600.jpg')
    if os.path.exists(str(i)+'/720.jpg'):
        os.remove(str(i)+'/720.jpg')
    if os.path.exists(str(i)+'/840.jpg'):
        os.remove(str(i)+'/840.jpg')'''

'''for di in dirs :
    n_di  = str(int(di)-1)
    images = glob.glob('../workspace/'+di+'/images/*.jpg')
    for im in images:
        print(im)
        shutil.copyfile(im,os.path.join(n_di,im.split('/')[-1]))'''
for i in range(0,62):

    names = glob.glob(str(i)+'/*.jpg')
    for name in names:
        if name.split('_')[-1]=='3.jpg':
            os.remove(name)

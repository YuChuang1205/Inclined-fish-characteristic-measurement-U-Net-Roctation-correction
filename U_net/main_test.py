#coding=gbk
'''
Created on 2020年3月27日

@author: 余创
'''

from model import *
from data import *
import os
import skimage.io


root_path = os.path.abspath('.')
pic_path = os.path.join(root_path,"data/test")
pic_out_path = os.path.join(root_path,"data/test_results")
model_path = os.path.join(root_path,"unet_mydata.hdf5")
model = load_model(model_path)
#stride = 512
#image_size = 512

piclist = os.listdir(pic_path)
#piclist.sort(key= lambda x:int(x[:-4])) 
for n in range(len(piclist)):
#     image = skimage.io.imread(os.path.join(pic_path,piclist[i]))
#     print(image.shape)
    new_name = piclist[n].split('.')[0]+'.png'
    image = skimage.io.imread(os.path.join(pic_path,piclist[n]),as_gray=True)
    #image = image/255
    #print(image.shape)
    h,w = image.shape
    image = np.expand_dims(image, axis=2)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image,verbose=1)
    pred = pred.reshape((h,w))
    pred = np.where(pred>0.5,255,0)
    
    skimage.io.imsave(os.path.join(pic_out_path,new_name), pred)
    print(np.max(pred),np.min(pred))
   































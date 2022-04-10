# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:49:01 2021

@author: TF
"""

'''import os
import cv2
#import keras


from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow
import tensorflow.keras as keras


from tensorflow.keras.models import load_model
#from keras import layers
#from keras import models


import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import math
import csv
#import h5py
#from keras.models import load_model
import sys
import os
from os import listdir

import resnet
from PIL import Image



from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.layers import Activation'''

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import resnet
import os
import cv2
import keras
import matplotlib.pyplot as plt
from PIL import Image

from keras.models import load_model


imagelabel_list = []
image_list = []
shape_list = []
IMG_list = []

i = 0
count_1 = 0
count_0 = 0
#将path的图片导入

def getFileList(dir,Filelist, ext=None):
    """
#    获取文件夹及其子文件夹中文件列表
#    输入 dir：文件夹根目录
#    输入 ext: 扩展名
#    返回： 文件路径列表
#    """
    newDir = dir
    
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)
    
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            getFileList(newDir, Filelist, ext)

    return Filelist


org_img_folder="./resize_all\\no_generator-筛选-2-文件\\all_generator+original"

# 检索文件

imglist = getFileList(org_img_folder, [], 'jpg')

print('本次执行检索到 '+str(len(imglist))+' 张图像\n')
 
for imgpath in imglist:
    
    imgname= os.path.splitext(os.path.basename(imgpath))[0]
    
    #将单斜杠换为双斜杠, \ 变为\\
            
    imgpath = eval(repr(imgpath).replace('\\', '\\\\'))
    #Label_01= imgpath[9:11]
    

     #定位最后一个\\的位置
    Last_index = imgpath.rfind('\\')
        
    A = Last_index+1
    B = Last_index+4   #留患者名字

    NAME = imgpath[A:B]
    
    print(NAME)
    
    '''if imgpath[A:B] == str("正常"):    
        
        Label_01 = 0
        
    else:
        
        Label_01 = 1'''
    
    #这是从文件夹提取图片打标签 
    if imgpath[14:16] == str("正常"):    
        Label_01 = 0
    else:
        Label_01 = 1
    
    print(Label_01)
    
    #img2 = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    #img2 = cv2.imdecode(imgpath, cv2.IMREAD_COLOR)
    
    
    # 对每幅图像执行相关操作  
    
    
    #img2 = load_img(imgpath, grayscale=False, color_mode='rgb', target_size=None,interpolation='nearest')
    
    
    img2 = load_img(imgpath, grayscale=False, color_mode='grayscale', target_size=None,interpolation='nearest')
    
    
    img2 = img2.resize((440, 400), Image.ANTIALIAS) #已经resize了的话就不用再做了 
    #img2 = img2.resize((400, 400), Image.ANTIALIAS) #已经resize了的话就不用再做了 

    
    #img2 = img2[::0]
    
    
    #img2.show()
    
    
    #img2.save('H:\\guangzhoufuyou\\其他基因病2021.11.9\\其他基因疾病_process//'+str(Label_01)+'_'+str(i)+'resize_2'+str(Label_01)+str(NAME)+'.jpg')
    #img2.save('resize_all//all_jpg4-正常//'+str(Label_01)+'_'+str(i)+'resize_2'+str(Label_01)+'.jpg')
    #img2.save('H:\\guangzhoufuyou\\其他异常（单染色体疾病） -更改后//'+str(Label_01)+'_'+str(i)+'resize_2'+str(Label_01)+'.jpg')
    #img2.save('H:\\guangzhoufuyou\\resize_ok\\forresize//'+str(Label_01)+'_'+str(i)+'resize_2'+str(Label_01)+'.jpg')
    #image_data.save((Destination+ 'Test/'  + myFile[:-7] + '-'+str(x)+'.jpg'))
    
    #img2 = img2[::0]
    
    #img2 = cv2.resize(img2, (400,400,3))
    
    img2= np.array(img2)#将Image格式转换为ndarry格式
    img2 =img2[:,:,np.newaxis] #[400,440]变为[400,440,1]
    
    img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
    
    
    '''IMG = img2[:,:,0] #这是灰度图选择RBG的一个通道就行了
    
    IMG_list.append(IMG)'''
    #print(img2)
    print(img2.shape)
    shape_list.append(img2.shape)
    
    '''Label = imgpath[9:11]
    
    print(imgpath[9:11])
    
    if imgpath[9:11] == str("正常"):    
        imagelabel = [0,1]
    else:
        imagelabel = [1,0]'''
        
    #定位最后一个\\的位置
    last_index = imgpath.rfind('\\')
        
    a = last_index+1
    b = last_index+2

    Label = imgpath[a:b]
    
    print(imgpath[a:b])
    
    if imgpath[a:b] == str("0"):    
        
        imagelabel = [0,1]
        count_0 = count_0+1
    else:
        
        imagelabel = [1,0]
        count_1 = count_1+1
    print(imagelabel)


        
    '''Label = imgpath[-5:-4]
    
    print(imgpath[-5:-4])
    
    if imgpath[-5:-4] == str("0"):    
        imagelabel = [0,1]
    else:
        imagelabel = [1,0]'''
        

    
    image_list.append(img2)
    imagelabel_list.append(imagelabel)
    i = i+1
#构建深度学习模型 

image_list = np.array(image_list)
imagelabel_list = np.array(imagelabel_list)


#打乱样本
np.random.seed(700)
np.random.shuffle(image_list) 
np.random.seed(700)
np.random.shuffle(imagelabel_list)

print(imagelabel_list)
print(count_0)
print(count_1)
#归一化
#image_list = (image_list - np.min(image_list)) / (np.max(image_list) - np.min(image_list))

img_rows, img_cols = 400, 440
#img_rows, img_cols = 400, 400
# The CIFAR10 images are RGB.
img_channels = 1
nb_classes = 2 

#model构建
#第二部：建立模型 
'''img_inputs = Input(shape=(400, 440, 1, ),name='imgs')
img_labels = Input((2,),name='labels')
#input_shape=(150, 150, 3)
#img_input = Input(shape=input_shape)

Conv2D_1=Conv2D(32, (3, 3), activation='relu',input_shape=(400, 440, 1),name='Conv2D_1')(img_inputs)
MaxPooling2D_1=MaxPooling2D((2, 2),name='MaxPooling2D_1')(Conv2D_1)
Conv2D_2=Conv2D(64, (3, 3), activation='relu',name='Conv2D_2')(MaxPooling2D_1)
MaxPooling2D_2=MaxPooling2D((2, 2),name='MaxPooling2D_2')(Conv2D_2)
Conv2D_3=Conv2D(128, (3, 3), activation='relu',name='Conv2D_3')(MaxPooling2D_2)
MaxPooling2D_3=MaxPooling2D((2, 2),name='MaxPooling2D_3')(Conv2D_3)
Conv2D_4=Conv2D(128, (3, 3), activation='relu',name='Conv2D_4')(MaxPooling2D_3)
MaxPooling2D_4=MaxPooling2D((2, 2),name='MaxPooling2D_4')(Conv2D_4)
Conv2D_5=Conv2D(128, (3, 3), activation='relu',name='Conv2D_5')(MaxPooling2D_4)
MaxPooling2D_5=MaxPooling2D((2, 2),name='MaxPooling2D_4')(Conv2D_4)
flatten=Flatten()(MaxPooling2D_5)
#flatten=Flatten()(MaxPooling2D_4)
dense_1=Dense(256, activation='relu',name='dense_1')(flatten)
#model.add(layers.Dense(2, activation='sigmoid'))  #512个神经元以前
dropout=Dropout(0.5)(dense_1)
prediction_dense_2=Dense(2, activation='softmax',name='prediction_dense_2')(dropout)


model_1=Model(inputs=[img_labels, img_inputs], outputs=[prediction_dense_2])'''

model =resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)          

print(model.summary())

model.compile(loss='binary_crossentropy',#这个是MNIST自带的优化器 binary_crossentropy这个 一般用于二分类 
              optimizer=keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])

'''history = model.fit([imagelabel_list, image_list],imagelabel_list,validation_split=0.05
                    ,shuffle=True,batch_size=8,epochs=50)'''
history = model.fit(image_list,imagelabel_list,validation_split=0.2
                     ,shuffle=True,batch_size=32,epochs=23)

#model.save("resize_all\\guangfuyou_yingerchaosheng-kmeans_generator.h5")
model.save("resize_all\\筛选第一次\\guangfuyou_yingerchaosheng_generator_筛选_23_resnet18_xin(xiyou)+其他基因23.h5")

print(history.history)
print(history.history.keys())
plt.figure(figsize=(5,5))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#rs = history.history['r_square']

#val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(loss))

#plt.plot(epochs, rs, 'bo', label='Training rs')

plt.figure(dpi=100)  #设置分辨率 

plt.plot(epochs, loss, 'r', label='loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')



plt.title('Training and validation accuracy')

plt.legend()





'''model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    b_size = 1
    max_epochs =100
    print("Starting training ")
    #history = model.fit(train_x, train_y,batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
    history = model.fit(train_x, train_y, validation_split=0.10, nb_epoch=100, batch_size=1, verbose=1)
    print("Training finished \n")
    
    print(history.history.keys())
    
    
    plt.figure(figsize=(5,5))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()'''



#predict = pd.DataFrame({'true_list':true_y_list,'predict_list':predict_list})
#predict.to_csv('test4_150ephch.csv',index = None,encoding = 'utf8')      
        


                     
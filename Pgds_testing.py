# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 13:29:40 2021

@author: TF
"""
import os
import cv2
import keras
'''from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras import models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import models
from keras import layers
from keras import optimizers'''


'''from keras.layers import Dense,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
from keras.optimizers import adam
from keras.models import load_model
from random import randint'''

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf
import tensorflow.keras as keras

'''from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.layers import Activation
#from keras import optimizers #optimizers优化器
from sklearn.model_selection import train_test_split'''

from tensorflow.keras.models import load_model
#from keras import layers
#from keras import models


import matplotlib.pyplot as plt


from sklearn.metrics import roc_curve, auc  ###计算roc和auc

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score



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
from tensorflow.keras.layers import Activation



from keras import layers
from keras import models

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
 
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

model = load_model("guangfuyou_yingerchaosheng_generator_筛选_25_resnet18_xin(xiyou)+其他基因25.h5") 
#VGG、resnet34 52结果
model2 = load_model("guangfuyou_yingerchaosheng_generator_筛选_10_resnet34.h5")  
model3 = load_model("guangfuyou_yingerchaosheng_generator_筛选_80_1.h5")   
model4 = load_model("guangfuyou_yingerchaosheng_generator_筛选_10_vgg6.h5")   



print(model.summary())



imagelabel_list = []
image_list = []
shape_list = []
IMG_list = []
predict_list = []
predict_labellist = []
matrix_pre = []
matrix_true = []
roc_label_list = []




imagelabel_list2 = []
image_list2 = []
predict_list2 = []
predict_labellist2 = []
matrix_pre2 = []
matrix_true2 = []
roc_label_list2 = []


imagelabel_list3 = []
image_list3 = []
predict_list3 = []
predict_labellist3 = []
matrix_pre3 = []
matrix_true3 = []
roc_label_list3 = []

imagelabel_list4 = []
image_list4 = []
predict_list4 = []
predict_labellist4 = []
matrix_pre4 = []
matrix_true4 = []
roc_label_list4 = []
csv_predict = []
csv_predict2 = []
csv_predict3 = []
csv_predict4 = []




i = 0
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
 

#org_img_folder="./resize_all\\test4"  #test data
org_img_folder="I:\\guangzhoufuyou\\resize_all\\test2" #AI-humen data

#org_img_folder="./验证用-1"
# 检索文件
imglist = getFileList(org_img_folder, [], 'jpg')
print('本次执行检索到 '+str(len(imglist))+' 张图像\n')
 
for imgpath in imglist:
    imgname= os.path.splitext(os.path.basename(imgpath))[0]
    
    #将单斜杠换为双斜杠, \ 变为\\
            
    imgpath = eval(repr(imgpath).replace('\\', '\\\\'))
    
    #img2 = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    #img2 = cv2.imdecode(imgpath, cv2.IMREAD_COLOR)
    
    # 对每幅图像执行相关操作  
    #img2 = load_img(imgpath, grayscale=False, color_mode='rgb', target_size=None,interpolation='nearest')
    img2 = load_img(imgpath, grayscale=False, color_mode='grayscale', target_size=None,interpolation='nearest')
    
    #vgg用的224，224的数据 
    img3 = img2.resize((224, 224), Image.ANTIALIAS)  #ANTIALIAS 代表高质量保存 
    img3= np.array(img3)#将Image格式转换为ndarry格式
    img3 =img3[:,:,np.newaxis] #[400,440]变为[400,440,1]
    img3 =img3[np.newaxis,:,:]
    
    #img2 = img2[::0]
    #img2.show()
    #img2.save('testout'+str(i)+'.jpg')
    
    #image_data.save((Destination+ 'Test/'  + myFile[:-7] + '-'+str(x)+'.jpg'))
    
    #img2 = img2[::0]
    
    #img2 = cv2.resize(img2, (400,400,3))
    
    img2= np.array(img2)#将Image格式转换为ndarry格式
    img =img2[:,:,np.newaxis] #[400,440]变为[400,440,1]
    img =img[np.newaxis,:,:]
    
    '''IMG = img2[:,:,0] #这是灰度图选择RBG的一个通道就行了
    
    IMG_list.append(IMG)'''
    #print(img2)
    print(img2.shape)
    shape_list.append(img2.shape)
    
    '''Label = imgpath[9:11]
    
    print(imgpath[9:11])
    
    if imgpath[9:11] == str("正常"):    
        #imagelabel = [0,1]
        imagelabel = "正常"
    else:
        #imagelabel = [1,0]
        imagelabel = "异常"     '''
        
        
     #定位最后一个\\的位置
    last_index = imgpath.rfind('\\')
        
    a = last_index+1
    b = last_index+2

    Label = imgpath[a:b]
    if imgpath[a:b] == str("0"):    
        #imagelabel = [0,1]
        imagelabel = "正常"
        true_value = 0
        roc_label = [0,1]
    else:
        #imagelabel = [1,0]
        imagelabel = "异常"
        true_value = 1
        roc_label = [1,0]
    #归一化
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img3 = (img3 - np.min(img3)) / (np.max(img3) - np.min(img3))
    
    
    #model 1
    predict = model.predict(img)
    print(predict)
    print(predict[0][0])
    if predict[0][0] <= predict[0][1]:    
        
       predict_label1 = "正常"
       pre_value1 = 0
       
    else:

       predict_label1 = "异常"
       pre_value1 = 1
    matrix_pre.append(pre_value1)
    matrix_true.append(true_value) 
    roc_label_list.append(roc_label)
    #predict_list.append(predict[0][0])
    
    predict_list.append(predict[0][0])
    
    csv_predict.append(predict)
    
    
    predict_labellist.append(predict_label1)
    #image_list.append(img2)
    imagelabel_list.append(imagelabel)
    print(i)
    
    
    #model 2
    predict2 = model2.predict(img)
    print(predict2)
    print(predict2[0][0])
    if predict2[0][0] <= predict2[0][1]:    
        
       predict_label2 = "正常"
       pre_value2 = 0
       
    else:

       predict_label2 = "异常"
       pre_value2 = 1
       
    matrix_pre2.append(pre_value2)
    matrix_true2.append(true_value) 
    roc_label_list2.append(roc_label)
    predict_list2.append(predict2[0][0])
    predict_labellist2.append(predict_label2)
    #image_list2.append(img2)
    imagelabel_list2.append(imagelabel)
    
    csv_predict2.append(predict2)
    
    print(i)
    


    #model 3
    predict3 = model3.predict(img)
    print(predict3)
    print(predict3[0][0])
    if predict3[0][0] <= predict3[0][1]:    
        
       predict_label3 = "正常"
       pre_value3 = 0
       
    else:

       predict_label3 = "异常"
       pre_value3 = 1
       
    matrix_pre3.append(pre_value3)
    matrix_true3.append(true_value) 
    roc_label_list3.append(roc_label)
    predict_list3.append(predict3[0][0])
    predict_labellist3.append(predict_label3)
    #image_list3.append(img3)
    imagelabel_list3.append(imagelabel)
    
    csv_predict3.append(predict2)
    
    print(i)
    
    
    
    #model 4
    predict4 = model4.predict(img3)
    print(predict4)
    print(predict4[0][0])
    if predict4[0][0] <= predict4[0][1]:    
        
       predict_label4 = "正常"
       pre_value4 = 0
       
    else:

       predict_label4 = "异常"
       pre_value4 = 1
       
    matrix_pre4.append(pre_value4)
    matrix_true4.append(true_value) 
    roc_label_list4.append(roc_label)
    predict_list4.append(predict4[0][0])
    predict_labellist4.append(predict_label4)
    #image_list3.append(img3)
    imagelabel_list4.append(imagelabel)
    
    csv_predict4.append(predict2)
    
    print(i)
    
    i = i+1
    
#model 1
predict_array = np.array(predict_list)
predict_array = np.array(csv_predict)
predict_array = np.squeeze(predict_array)   #(88,1,2)变为(88,2)  np.squeeze做ndarry的降维
#Y_pred = [np.max(y) for y in predict_array]  # 取出y中元素最大值所对应的索引 np.max()是取出最大值 np.argmax()取出最大值的索引
roc_label_array = np.array(roc_label_list)
y_binarize = label_binarize(matrix_true, classes=[0, 1])
print(roc_label_array.shape)    
print(predict_array.shape)
print(predict_array)
#print(Y_pred)
print(y_binarize)
fpr, tpr, thresholds = roc_curve(y_binarize, predict_list)  ###计算真正率和假正率
roc_auc = auc(fpr, tpr)  ###计算auc的值
#最佳阈值

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)


#model 2
predict_array2 = np.array(predict_list2)
predict_array2 = np.array(csv_predict2)
predict_array2 = np.squeeze(predict_array2)   #(88,1,2)变为(88,2)  np.squeeze做ndarry的降维
#Y_pred = [np.max(y) for y in predict_array]  # 取出y中元素最大值所对应的索引 np.max()是取出最大值 np.argmax()取出最大值的索引
roc_label_array2 = np.array(roc_label_list2)
y_binarize2 = label_binarize(matrix_true2, classes=[0, 1])
fpr2, tpr2, threshold2 = roc_curve(y_binarize2, predict_list2)  ###计算真正率和假正率
roc_auc2 = auc(fpr2, tpr2)  ###计算auc的值

#model 3
predict_array3 = np.array(predict_list3)
predict_array3 = np.array(csv_predict3)
predict_array3 = np.squeeze(predict_array3)   #(88,1,2)变为(88,2)  np.squeeze做ndarry的降维
#Y_pred = [np.max(y) for y in predict_array]  # 取出y中元素最大值所对应的索引 np.max()是取出最大值 np.argmax()取出最大值的索引
roc_label_array3 = np.array(roc_label_list3)
y_binarize3 = label_binarize(matrix_true3, classes=[0, 1])
fpr3, tpr3, threshold3 = roc_curve(y_binarize3, predict_list3)  ###计算真正率和假正率
roc_auc3 = auc(fpr3, tpr3)  ###计算auc的值

#model 4
predict_array4 = np.array(predict_list4)
predict_array4 = np.array(csv_predict4)
predict_array4 = np.squeeze(predict_array4)   #(88,1,2)变为(88,2)  np.squeeze做ndarry的降维
#Y_pred = [np.max(y) for y in predict_array]  # 取出y中元素最大值所对应的索引 np.max()是取出最大值 np.argmax()取出最大值的索引
roc_label_array4 = np.array(roc_label_list4)
y_binarize4 = label_binarize(matrix_true4, classes=[0, 1])
fpr4, tpr4, threshold4 = roc_curve(y_binarize4, predict_list4)  ###计算真正率和假正率
roc_auc4 = auc(fpr4, tpr4)  ###计算auc的值

#计算F1
'''precision = precision_score(y_binarize, predict_list, average='weighted')
recall = recall_score(y_binarize, predict_list, average='weighted')
f1_score = f1_score(y_binarize, predict_list, average='weighted')
accuracy_score = accuracy_score(y_binarize, predict_list)'''


print(f1_score)


#plt.figure()
lw = 2
#plt.figure(figsize=(10,10),dpi = 600,facecolor='snow',edgecolor='white')
'''plt.figure(figsize=(10,10),linewidth=2.5,dpi = 600,facecolor='white',edgecolor='black') #这儿的facecolor是图表区的 edgecolor为图表区的边框颜色
plt.axes(facecolor = "white") #这儿的faccolor是图工作区的背景色

ax = plt.axes()

ax.spines['left'].set_visible(True)

plt.plot(fpr, tpr, color='c',
         lw=lw, label='Pgds-ResNet (AUC = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fpr2, tpr2, color='red',
         lw=lw, label='resnet34 (AUC = %0.2f)' % roc_auc2) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fpr3, tpr3, color='black',
         lw=lw, label='vgg16 (AUC = %0.2f)' % roc_auc3) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fpr4, tpr4, color='blue',
         lw=lw, label='vgg19 (AUC = %0.2f)' % roc_auc4) ###假正率为横坐标，真正率为纵坐标做曲线

#plt.plot([0.3125],[0.92],'*',color = 'black',markersize='20')
plt.plot(optimal_point[0], optimal_point[1], marker='o', color='black')
plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')

#设置坐标刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity',fontsize=20)
plt.ylabel('Sensitivity',fontsize=20)
plt.title('Receiver operating characteristic',fontsize=20)
plt.legend(loc="lower right")
plt.grid(False)
plt.show()'''








#绘制人机大战roc












#构建深度学习模型 

image_list = np.array(image_list)

#归一化
#image_list = (image_list - np.min(image_list)) / (np.max(image_list) - np.min(image_list))


predict = pd.DataFrame({'true_list':imagelabel_list,'predict_list':predict_labellist,'predict_value':csv_predict})
predict.to_csv('true_predict_yanzheng_3.csv',index = None,encoding = 'utf8')
    
#做混淆矩阵
sns.set()
f,ax=plt.subplots(figsize = (5,4),dpi = 600)
C2= confusion_matrix(matrix_true, matrix_pre, labels=[0, 1])
print(C2) #打印出来看看
p2 = sns.heatmap(C2,annot=True,ax=ax) #画热力图

ax.set_title('Pgds-ResNet') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴

C3= confusion_matrix(matrix_true, matrix_pre2, labels=[0, 1])

C4= confusion_matrix(matrix_true, matrix_pre3, labels=[0, 1])

C5= confusion_matrix(matrix_true, matrix_pre4, labels=[0, 1])


sns.set()
f,ax=plt.subplots(figsize = (5, 4),dpi = 600)
print(C3) #打印出来看看
p3 = sns.heatmap(C3,annot=True,ax=ax) #画热力图

ax.set_title('resnet34') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴

sns.set()
f,ax=plt.subplots(figsize = (5,4),dpi = 600)
print(C4) #打印出来看看
p4 = sns.heatmap(C4,annot=True,ax=ax) #画热力图

ax.set_title('vgg16') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴

sns.set()
f,ax=plt.subplots(figsize = (5,4),dpi = 600)
#C5 = float(C5)
print(C5) #打印出来看看
C5_data = np.array([[100.0,0],[47,39]])
#sns.heatmap(data,annot=True)

p5 = sns.heatmap(C5_data,annot=True,ax=ax) #画热力图

ax.set_title('vgg19') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴




'''Heatmap_path = "I:\\guangzhoufuyou\\resize_all\\test4_pic"

s2 = p2.get_figure()
s2.savefig(Heatmap_path+'\\'+'Pgds-ResNet.jpg',dpi=300,bbox_inches='tight')

s3 = p3.get_figure()
s3.savefig(Heatmap_path+'\\'+'resnet34.jpg',dpi=300,bbox_inches='tight')

s4 = p4.get_figure()
s4.savefig(Heatmap_path+'\\'+'vgg16.jpg',dpi=300,bbox_inches='tight')

s5 = p5.get_figure()
s5.savefig(Heatmap_path+'\\'+'vgg19.jpg',dpi=300,bbox_inches='tight')'''




#做人机大战的ROC


#plt.figure(figsize=(10,10),dpi = 600,facecolor='snow',edgecolor='white')
plt.figure(figsize=(10,10),dpi = 600,linewidth=2.5,facecolor = "white",edgecolor= "black") #这儿的facecolor是图表区的 edgecolor为图表区的边框颜色

plt.axes(facecolor = "white") #facecolor设置画布的背景颜色



plt.plot(fpr, tpr, color='c',
         lw=lw, label='Pgds-ResNet (AUC = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线

'''plt.plot(fpr2, tpr2, color='red',
         lw=lw, label='resnet34 (AUC = %0.2f)' % roc_auc2) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fpr3, tpr3, color='black',
         lw=lw, label='vgg16 (AUC = %0.2f)' % roc_auc3) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fpr4, tpr4, color='blue',
         lw=lw, label='vgg19 (AUC = %0.2f)' % roc_auc4) ###假正率为横坐标，真正率为纵坐标做曲线'''

#plt.plot([0.3125],[0.92],'*',color = 'black',markersize='20')
plt.plot(optimal_point[0], optimal_point[1], marker='o', color='black')
plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')

plt.plot(0.06, 0.85, marker='v', color='r',lw=4,label='Chief physician')
#plt.text(0.06, 0.85, f'Chief physician')

plt.plot(0.13, 0.58, marker='v', color='b',lw=4,label='Attending doctor')
#plt.text(0.13, 0.58, f'Attending doctor')

plt.plot(0.28, 0.44, marker='v', color='m',lw=4,label='Assistant physician')
#plt.text(0.28, 0.44, f'Assistant physician')
#设置坐标刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity',fontsize=20)
plt.ylabel('Sensitivity',fontsize=20)
#plt.title('AI compared with humen doctor',fontsize=20)
plt.legend(loc="lower right")
plt.grid(False)


ax = plt.axes()
ax.spines['left'].set_visible(True)

plt.show()


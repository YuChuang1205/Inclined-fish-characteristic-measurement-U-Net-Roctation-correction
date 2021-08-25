#coding=gbk
'''
Created on 2021年3月20日

@author: 余创
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np 
import math 
from math import *
from PIL import Image
import os
import skimage.io 
 
 

#定义计算欧式距离函数
def distEur(a,b):
    sum1 = 0
    for i in range(len(a)):
        val = np.power(a[i]-b[i],2)
        sum1 = sum1 + val
    dist = np.sqrt(sum1)
    return dist



def smooth_mean(a,m):
    for i in range(m,len(a)-m):
        a[i]=np.mean(a[i-m:i+m])
    return a

#对数据做平滑操作
def smooth_med(a,m):
    for i in range(m,len(a)-m):  
        a[i]=np.median(a[i-m:i+m])
    return a
    #print(tmp1)



def find_angel(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    #print(gray)
     
     
    ###找到二值图的重心
    r,c = np.shape(gray)
    #print(r,c)
    
    
    r_num = 0
    c_num = 0
    count  = 0
    
    for i in range(r):
        for j in range(c):
            if gray[i][j] == 255:
                r_num = r_num + i + 1
                c_num = c_num + j + 1
                count = count + 1
            else:
                continue
                
    center_x =   int(c_num/count)
    center_y =   int(r_num/count)  
    
    print([center_x,center_y]) 
    
    #gray[center_x - 1][center_y - 1]=0
    #gray[center_x - 3:center_x + 1,center_y - 3:center_y + 1]=0
    
    
    #ret, binary = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
     
    
    contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) 
    
    max_area = 0
    k= 0 
    
    #取出轮廓面积最大的
    #print(len(contours))
    for i in range(len(contours)):
        #print(cv2.contourArea(contours[i]))
        if (cv2.contourArea(contours[i])>max_area):
            max_area =cv2.contourArea(contours[i])
            k = i
        else:
            continue
            
    
    contours_temp = contours[k]
    
    #contours_temp = contours[0]
    #print(contours_temp)
    #print(np.shape(contours_temp))
    
    contours_squeeze = contours_temp.squeeze(axis = 1)
    #print(contours_squeeze)
    #print(np.shape(contours_squeeze))
    
    ###找到中心点在轮廓上的点（注意相同x有多个对应y）然后逆时针排序
    
    c_r,c_c = np.shape(contours_squeeze)
    
    temp = 0
    temp2 = 0
    
    #定上点
    for i in range(c_r):
        if(contours_squeeze[i,0] == center_x ):
            temp = i 
            break
        elif(contours_squeeze[c_r-1-i,0] == center_x ):
            temp = c_r-1-i
            break
        else:
            continue
    
    #print("temp=",temp)
    
    
    #定下点
    for i in range(c_r):
        if(contours_squeeze[i,0] == center_x and abs(contours_squeeze[i,1] - contours_squeeze[temp,1])>100  ):
            temp2 = i 
            break
        elif(contours_squeeze[c_r-1-i,0] == center_x and abs(contours_squeeze[c_r-1-i,1] - contours_squeeze[temp,1])>100 ):
            temp2 = c_r-1-i
            break
        else:
            continue
    
    #print("temp2=",temp2)
    
    
    
    
    
    
    contours_news = [0 for i in range(c_r)]
    contours_news2 = [0 for i in range(c_r)]
    
    #对于一个数组组成循环，并改变初始值进行输出
    for j in range(c_r):
        if(j+temp <c_r):
            contours_news[c_r-1-j]=contours_squeeze[j+temp]
        else:
            contours_news[c_r-1-j]=contours_squeeze[j+temp-c_r]
        
     
    for j in range(c_r):
        if j == 0 :
            contours_news2[j] = contours_news[c_r-1]
        else:
            contours_news2[j] = contours_news[j-1] 
     
     
    #print(contours_news2)
    #print(np.shape(contours_news2))  
    
    
    #提取鱼嘴方向平角部分
    
    if temp > temp2:
        contours_news_half = contours_news2[0:temp-temp2+1]
    else:
        contours_news_half = contours_news2[0:temp+c_r-temp2+1]
    
    
    r_half,c_half = np.shape(contours_news_half)
    
    #print("r_half",r_half)
         
     
    
    
    
    angle = []
    for i in range(r_half):
        
        temp_num = (contours_news_half[i][0] - center_x)/(contours_news_half[i][1] - center_y)
        if temp_num == inf:
            temp_num = 1000000
            #print('i:',i)
        if str(temp_num)==str(-0.0):
            
            temp_num = -0.0000001
        if str(temp_num)==str(0.0):
            temp_num = 0.0000001   
    #     print(temp_num)
    #     angle_temp = math.tan(temp_num)
        angle.append(temp_num)
    
    
    angle_true = []     
    for i in range(1800):
        angle_temp = math.tan(pi-(pi/1800)*i)
        angle_true.append(angle_temp)
      
     
     
    index_list = []
    point_list = []   
    for i in range(1800):
        temp_list = [] 
        for j in range(r_half):
            temp = abs(angle[j] - angle_true[i])
            temp_list.append(temp)
        temp1 = np.argmin(temp_list)
        index_list.append(temp1)
        point_list.append(contours_news_half[temp1])
    
        
    
        
    dist = []
    for i in range(1800):
        dist_temp = distEur(point_list[i], [center_x,center_y])
        dist.append(dist_temp)
      
      

     
    #dist = smooth_mean(dist,3)
    #dist = smooth_med(dist,5) 
     
     
    key_point = 0
    # for i in range(1,c_r//2):
    #     if dist[i]< dist[i-1] and dist[i]< dist[i+1] and dist[i]>200:
    #         key_point = i
    #         break
    #     else:
    #         continue
    
    
    m = 25
    for i in range(m,1800-m):
        if dist[i] == min(dist[i-m:i+m])  and dist[i]>200:
            key_point = i
            break
        else:
            continue
    
    
    # for i in range(1,c_r//2):
    #     if dist[i]< dist[i-5] and dist[i]< dist[i+5] and dist[i]>200:
    #         key_point = i
    #         break
    #     else:
    #         continue
    
    
    
    angel = ((i+1)/1800)*180
    #print("拐点：",i)
    #print("角度：",angel-98) 
    
        
            
    #   
    # plt.plot([i for i in range(1800)],dist,color="red",linewidth=1 )
    # plt.show()
    return  angel-98

root_path = os.path.abspath('.')
input_img_path = os.path.join(root_path,'input\\img')
input_mask_path = os.path.join(root_path,'input\\mask')
output_img_path = os.path.join(root_path,'output\\img')
output_mask_path = os.path.join(root_path,'output\\mask')

pic_list = os.listdir(input_mask_path)
pic_list.sort(key= lambda x:int(x[:-4]))
for i in range(len(pic_list)):
    print("--------------------------------------------")
    print("处理的图像为：%s" %(pic_list[i]))
    img = cv2.imread(os.path.join(input_mask_path,pic_list[i]))
    new_name = pic_list[i].split('.')[0]+'.jpg'
    img_ro = Image.open(os.path.join(input_mask_path,pic_list[i]))
    img1_ro = Image.open(os.path.join(input_img_path,new_name))
    angel = find_angel(img)
    print("倾斜角度为：%f" %(angel))
    img_ro = img_ro.rotate(angel)
    img1_ro = img1_ro.rotate(angel)
    img_ro.save(os.path.join(output_mask_path,pic_list[i]))
    img1_ro.save(os.path.join(output_img_path,new_name))
    print("---------------------------------------------")
    
    
#矩形框检测    
pic_list2 = os.listdir(output_mask_path)
pic_list2.sort(key= lambda x:int(x[:-4]))
for i in range(len(pic_list2)):
    print("--------------------------------------------")
    print("处理的图像为：%s" %(pic_list2[i]))
    img = cv2.imread(os.path.join(output_mask_path,pic_list2[i]))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    w = max(w,h)
    h = min(w,h)
    print("--------------------------------------------------------------")
    print("图片：%s的正接矩形长为：%f" %(pic_list2[i],w))
    print("图片：%s的正接矩形宽为：%f" %(pic_list2[i],h))


print("DONE!!!!")
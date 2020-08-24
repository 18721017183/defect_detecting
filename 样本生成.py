import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from time import time
import argparse

#cv2.imshow()
def show_image(name,image):
    cv2.namedWindow(name,flags=1)
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def deal_contours_image(image_path):
    # image = cv2.imread(image_path, 1)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if opt.show_image:
        cv2.imshow('crc',image)
    # cv2.imshow("ori", image)

    #保留原图，用于显示图像
    img = image.copy()

    # #保存原图，用于裁剪图片
    img1 = image.copy()

    image = image[:,:,2]
    image_name = os.path.split(image_path)[-1]
    # 阈值处理后的图片
    # thresh_image = threshTwoPeaks1(image)
    # cv2.imshow('thresh_image',thresh_image)
    print(image_name)
    # 高斯平滑
    image = cv2.GaussianBlur(image,(3,3),0.5)
    #中值滤波,消除椒盐噪声。速度慢
    # image = mediaBlur(image,(3,3))
    # 膨胀处理，过滤细小竖直边线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    image = cv2.dilate(image, kernel)
    # cv2.imshow("dilate", image)
    # 边缘检测，找出明显轮廓。
    image = cv2.Canny(image, 60, 80)
    if opt.show_image:
        cv2.namedWindow("Canny",flags=1)
        cv2.imshow("Canny", image)
    # 膨胀处理，将紧挨着的边线合并，用于后续寻找外轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    image = cv2.dilate(image, kernel)
    # cv2.imshow('canny', image)
    #寻找外轮廓
    hc = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = hc[1]
    print('轮廓数：', len(contours))
    for i in range(len(contours)):
        # 绘制轮廓
        # cv2.drawContours(img, contours, i, 255, 2)
        # 获取最小旋转矩形
        # points = cv2.minAreaRect(contours[i])
        # # 计算旋转矩形4个顶点坐标
        # rect = cv2.boxPoints(points)

        #计算最小外包直立矩形
        rect = cv2.boundingRect(contours[i])

        print(rect)
        # 数据类型转换
        rect = np.uint0(rect)
        # 绘制矩形
        #旋转矩形4个顶点
        # cv2.drawContours(img, [rect], 0, (0, 255, 0), 2)
        # 直立矩形对角点
        x1 = rect[0]
        x2 = rect[0] + rect[2]
        y1 = rect[1]
        y2 = rect[1] + rect[3]
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
        global num
        num = num + 1
        if rect[3] > 600:
            continue
        cv2.imwrite('./datas/{}_A.jpg'.format(num),img1[y1:y2,x1:x2])  ######修改图片保存路径
    if opt.show_image:
        show_image(image_name, img)

"""检测轮廓"""
def contours_defect():
    # image_path = r'F:\datas\原始数据\20200630_拍摄（OCR-辣条缺陷）\缺陷检测\test_缺陷检测 - 右' #50--120
    # image_path = r'C:\Users\pc\Desktop\img\ori'
    image_path = r'C:\Users\pc\Desktop\img\20200812cut'
    if os.path.isfile(image_path):
        deal_contours_image(image_path)
    elif os.path.isdir(image_path):
        images = os.listdir(image_path)
        #给图片排序用
        # images = [image.split('.')[0] for image in images]
        # images = np.sort([int(i) for i in images])
        # images = [str(i) + '.jpg' for i in images]
        for img_ in images:
            start_time = time()
            image_full_name = os.path.join(image_path, img_)
            deal_contours_image(image_full_name)
            print('时间',time()-start_time)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--show_image",default=True,help="choose to show image")
    opt = parse.parse_args()

    num = 0

    contours_defect()
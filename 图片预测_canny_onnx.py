import numpy as np
from time import time
import argparse
import onnxruntime as rt
import core.config as cfg
import cv2
np.set_printoptions(threshold=np.inf)
import os

def do_onnx_predict(image):
    img = cv2.resize(image,(cfg.image_height, cfg.image_width))
    cv2.imshow("img",img)
    cv2.waitKey(0)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # b35. 将图像转换为numpy数组的形式
    img_arr = np.array(img).reshape(cfg.image_height, cfg.image_width, 3)
    img_arr = img_arr.transpose(2, 0, 1)
    # 防止图像的数据像素值太大，导致模型过拟合，将像素值从0~255缩减到0~1之间（这个计算规则就是MinMaxScaler）
    img_arr = img_arr / 255.0
    img_arr = np.expand_dims(img_arr,0).astype(np.float32)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: img_arr})[0]
    print("prec_onx",pred_onx)
    Max = np.argmax(pred_onx, 1)
    print('预测值:',Max[0])
    if Max == 1:
        # cv2.imwrite("./data_ct/1.jpg",image)
        # cv2.imshow("pre:",image)
        # cv2.waitKey(0)
        return 1
    else:
        return 0

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
    #print('轮廓数：', len(contours))
    error_image_list = []
    total_num = 0
    for i in range(len(contours)):
        # 绘制轮廓
        # cv2.drawContours(img, contours, i, 255, 2)
        # 获取最小旋转矩形
        # points = cv2.minAreaRect(contours[i])
        # # 计算旋转矩形4个顶点坐标
        # rect = cv2.boxPoints(points)

        #计算最小外包直立矩形
        rect = cv2.boundingRect(contours[i])

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
        flag = do_onnx_predict(img1[y1:y2,x1:x2])
        if flag == 1:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            error_image_list.append(img1[y1:y2,x1:x2])
        total_num += 1

    if opt.show_image:
        show_image(image_name, img)
    error_num = len(error_image_list)
    print("出错区域数量：{}/{}".format(error_num,total_num))
    if error_num > 0:
        cv2.imshow("原图",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


"""检测轮廓"""
def contours_defect():
    image_path = r'C:\Users\pc\Desktop\img\20200630cut\2020_6_30_11_15_48_64.bmp_left.jpg'
    # image_path = r'C:\Users\pc\Desktop\img\img\image_total\expand\5'
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
    parse.add_argument("--show_image",default=False,help="choose to show image")
    opt = parse.parse_args()

    num = 0

    sess = rt.InferenceSession("torch_model_train_Acc98.036_test_Acc0.959_ct.onnx")
    contours_defect()
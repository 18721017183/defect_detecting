'''
利用onnx进行预测
'''
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import numpy as np
import time
import onnxruntime as rt
import cv2
from utils import utils


#图像处理
image_width = 224   #图像宽的
image_height = 224  #图像高度

#导入onnx模型，生成sess
onnx_model = "./torch_model_train_Acc0.9385_test_Acc0.928.onnx"  #保存的onnx模型
sess = rt.InferenceSession(onnx_model)
#读取输入输出的名字
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

#定义摄像头
vid = cv2.VideoCapture(0)
print(vid.get(cv2.CAP_PROP_FPS))
while True:
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
    else:
        raise ValueError("No image!")

    frame_size = frame.shape[:2]
    image_data = utils.image_preporcess(np.copy(frame), [image_width, image_height])
    image_data = image_data[np.newaxis, ...]

    start_time = time.time()
    #图片转换为pytorch的NCHW形式
    image_data = image_data.transpose((0,3,1,2))
    #模型运行
    pred_onx = sess.run([label_name], {input_name: image_data.astype(np.float32)})[0]
    print(np.argmax(pred_onx,1))
    end_time = time.time()
    print(end_time-start_time)
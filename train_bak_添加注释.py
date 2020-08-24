#coding:utf-8
#!/usr/bin/env python3.7

'''ubuntu中跑,反正找不到cv2包'''
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')   #解决cv2没法导入的问题

import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.onnx as torch_onnx
from tensorboardX import SummaryWriter

from core.networks import Model
import core.config as cfg

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

writer = SummaryWriter('./runs')   #定义日志输出

# if hasattr(torch.cuda, 'empty_cache'):
#     torch.cuda.empty_cache()

'''
模型效果评估
'''
def val(net,loss_func,max_iter = 50):
    #加载数据
    transform = transforms.Compose([
        transforms.Resize((cfg.image_height, cfg.image_width)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    val_data = datasets.ImageFolder(root=cfg.val_data_path, transform=transform)
    val_loader = DataLoader(dataset=val_data, batch_size=cfg.val_batch_size, shuffle=True)

    #确定不做训练操作
    net.eval()

    #定义损失，用于评估
    total_loss = 0.0
    total_correct_num = 0

    #定义数据迭代器
    val_iter = iter(val_loader)
    max_iter = min(max_iter, len(val_loader))

    #开始测试
    for i in range(max_iter):
        #获取当前批次数据
        (data, label) = val_iter.next()

        #使用cuda加速
        if torch.cuda.is_available:
            data,label = data.cuda(),label.cuda()

        #前向预测
        output = net(data)
        #计算损失
        loss = loss_func(output,label)
        #计算准确率
        correct_num = torch.sum(torch.argmax(output, 1) == label.data)
        total_correct_num += correct_num
        #累加损失
        total_loss += float(loss)
        print('loss is:{:.4f},accuracy is:{:.4f}'.format(loss/cfg.val_batch_size,float(correct_num)/cfg.val_batch_size))
    print('Loss is:{:.4f},Correct is:{:.4f}'.format(total_loss/len(val_data),float(total_correct_num)/len(val_data)))


'''
模型训练
'''
def train():
    #加载训练数据
    transform = transforms.Compose([
        transforms.Resize((cfg.image_height,cfg.image_width)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_data = datasets.ImageFolder(root=cfg.train_data_path, transform=transform)
    data_loader = DataLoader(dataset=train_data, batch_size=cfg.train_batch_size, shuffle=True)

    #定义模型
    model = Model()
    model = model.train()
    if torch.cuda.is_available():
        model = model.cuda()

    #使用交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()
    #定义优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    if os.path.exists(cfg.save_torch_model):
        print('进行模型恢复操作。。。')
        model = torch.load(cfg.save_torch_model)

    #开始训练
    for epoch in range(cfg.train_max_epoch):
        running_loss = 0
        running_accuracy = 0
        # scheduler.step()
        for idx, data in enumerate(data_loader):
            #读取数据
            x_train, y_train = data
            x_train, y_train = Variable(x_train), Variable(y_train)
            if torch.cuda.is_available:
                x_train, y_train = x_train.cuda(),y_train.cuda()

            #前向计算
            y_pred = model(x_train)
            #计算损失
            loss = criterion(y_pred, y_train)
            #梯度重置
            optimizer.zero_grad()
            #反向计算
            loss.backward()
            #参数更新
            optimizer.step()

            #效果评估
            running_loss += loss.data.data
            running_accuracy += torch.sum(torch.argmax(y_pred, 1) == y_train.data)
            print('{}:{}/{}'.format(epoch, idx, len(data_loader)))
            print('loss:{},accuracy:{}'.format(loss.data.data, float(torch.sum(torch.argmax(y_pred, 1) == y_train.data))/cfg.train_batch_size))

            #记录日志
            writer.add_scalar('loss', loss, epoch * cfg.train_batch_size + idx)
            writer.add_scalar('running_accuracy', running_accuracy, epoch * cfg.train_batch_size + idx)
            writer.add_scalar('running_loss', running_loss, epoch + cfg.train_batch_size + idx)

            #模型保存
            if (epoch*cfg.train_batch_size+idx)//100==0:
                torch.save(model,cfg.save_torch_model)

        #当前批次的模型效果
        print('Loss is:{:.4f},accuracy is:{:.4f}'.format(running_loss / len(train_data),
                                                         100 * float(running_accuracy) / len(train_data)))
        torch.save(model, './model/model_loss{}_acc{}.pkl'.format(running_loss / len(train_data),
                                                                  100 * float(running_accuracy) / len(train_data)))
        # torch.save(model.state_dict(), './model/model_loss{}_acc{}.pkl'.format(running_loss / len(train_data),
        #                                                                        100 * float(running_accuracy) / len(train_data)))
        torch.save(model,cfg.save_torch_model)
        print('模型保存成功')

        #测试集上效果评估
        val(model,criterion)
    # writer.export_scalars_to_json("C:/Users/pc/Desktop/tensorboardX/all_scalars.json")
    writer.close()

    #模型转换为onnx
    input_shape = (cfg.imgge_channels, cfg.image_height, cfg.image_width)
    model_onnx_path = "torch_model.onnx"
    # model = Model()
    model.train(False)

    # print(list(model.parameters))
    # Export the model to an ONNX file
    dummy_input = Variable(torch.randn(1, *input_shape))
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    output = torch_onnx.export(model,
                               dummy_input,
                               model_onnx_path,
                               verbose=False)
    print('onnx转出成功')

if __name__ == '__main__':
    train()
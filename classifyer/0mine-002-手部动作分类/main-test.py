# coding=utf-8

import os
# 图像读取库
from PIL import Image
# 矩阵运算库
import numpy as np
import tensorflow as tf
from PIL import Image

# 数据文件夹
data_dir = "H:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/train-data"
# 训练还是测试True False
train = False
# 模型文件路径
model_path = "H:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/train-model/image_model"


# 从文件夹读取图片和标签到numpy数组中
# 标签信息在文件名中，例如1_40.jpg表示该图片的标签为1
def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        # data = np.array(image)
        label = int(fname.split("_")[0])
        datas.append(data)
        labels.append(label)

    datas = np.array(datas)
    labels = np.array(labels)

    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels
    # return fpaths, datas


fpaths, datas, labels = read_data(data_dir)
# fpaths, datas = read_data(data_dir)

# 计算有多少类图片
# num_classes = len(set(labels))
num_classes = 10

# 定义Placeholder，存放输入和标签
datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
labels_placeholder = tf.placeholder(tf.int32, [None])

# 存放DropOut参数的容器，训练时为0.25，测试时为0
dropout_placeholdr = tf.placeholder(tf.float32)
#
# 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

# 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

# 定义卷积层, 60个卷积核, 卷积核大小为3，用Relu激活
conv2 = tf.layers.conv2d(pool0, 60, 3, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2])

# 将3维特征转换为1维向量
flatten = tf.layers.flatten(pool2)

# 全连接层，转换为长度为100的特征向量
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

# 加上DropOut，防止过拟合
dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

# 未激活的输出层
logits = tf.layers.dense(dropout_fc, num_classes)

predicted_labels = tf.arg_max(logits, 1)

# 利用交叉熵定义损失
losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, num_classes),
    logits=logits
)
# 平均损失
mean_loss = tf.reduce_mean(losses)

# 定义优化器，指定要优化的损失函数
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)

# 用于保存和载入模型
saver = tf.train.Saver()

#用于保存结果
res=open('result.txt','w')
resu='H:/桌面/笔记/大三上/TensorFlow/0mine-002-数字识别/result/'

with tf.Session() as sess:
    # if train:
    #     print("训练模式")
    #     # 如果是训练，初始化参数
    #     sess.run(tf.global_variables_initializer())
    #     # 定义输入和Label以填充容器，训练时dropout为0.25
    #     train_feed_dict = {
    #         datas_placeholder: datas,
    #         labels_placeholder: labels,
    #         dropout_placeholdr: 0.5
    #     }
    #     for step in range(150):
    #         _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)
    #
    #         if step % 10 == 0:
    #             print("step = {}\tmean loss = {}".format(step, mean_loss_val))
    #     saver.save(sess, model_path)
    #     print("训练结束，保存模型到{}".format(model_path))
    # else:
        print("测试模式")
        # 如果是测试，载入参数
        saver.restore(sess, model_path)
        print("从{}载入模型".format(model_path))
        # label和名称的对照关系
        label_name_dict = {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9"
        }
        # 定义输入和Label以填充容器，测试时dropout为0
        test_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0
        }
        # test_feed_dict = {
        #     datas_placeholder: datas,
        #     # labels_placeholder: labels,
        #     dropout_placeholdr: 0
        # }
        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        # 真实label与模型预测label
        num=1
        for fpath, real_label, predicted_label in zip(fpaths, labels, predicted_labels_val):
        # for fpath, predicted_label in zip(fpaths ,predicted_labels_val):
            # 将label id转换为label名
            real_label_name = label_name_dict[real_label]
            predicted_label_name = label_name_dict[predicted_label]
            # print(predicted_label_name)
            # print(predicted_label)
            print("{}\t{} => {}".format(fpath, real_label_name, predicted_label_name))
            # print("{}\t=> {}".format(fpath, predicted_label))
            # input()
            res.write(fpath[39:]+'=>'+str(predicted_label)+'\n')
            im = Image.open(fpath)
            im.save(resu+str(num)+'-'+str(predicted_label)+'.jpg', 'JPEG')
            num=num+1

        # 正确次数
        correct_number = 0
        sum_number=0
        # 计算正确率
        for fpath, real_label, predicted_label in zip(fpaths, labels, predicted_labels_val):
            sum_number +=1
            if real_label == predicted_label:
                correct_number += 1

        correct_rate = correct_number / sum_number
        print('正确率: {:.2%}'.format(correct_rate))

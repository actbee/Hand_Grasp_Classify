#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# File  : autofenbianlv.py
# Author: DaShenHan&道长-----先苦后甜，任凭晚风拂柳颜------
# Date  : 2019/5/14
from glob import glob
from PIL import Image
import os

img_path = glob("H:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/test2/*.jpg")
path_save = "H:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/test2/"
a = range(0, len(img_path))
i = 1
for file in img_path:
    # print(file[39:])
    print(file[45:])
    # input()
    # name = os.path.join(path_save, "%d.jpg" % a[i])
    name = os.path.join(path_save, str(num) + '_' + str(i) + '.jpg')
    im = Image.open(file).convert('RGB')
    # im.thumbnail((32, 32))
    om = im.resize((32, 32))
    print(im.format, im.size, im.mode)
    print(om.format, om.size, om.mode)
    # im.save(name, 'JPEG')
    om.save(name)
    i = i + 1

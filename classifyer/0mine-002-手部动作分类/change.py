# coding=utf-8
from PIL import Image
import os
import os.path
import glob


def rename(rename_path, outer_path, folderlist):
    # 列举文件夹
    for folder in folderlist:
        if os.path.basename(folder) == 'L-大拇指夹持':
            foldnum = 0
        elif os.path.basename(folder) == 'L-单指':
            foldnum = 1
        elif os.path.basename(folder) == 'L-兰花指':
            foldnum = 2
        elif os.path.basename(folder) == 'L-捏':
            foldnum = 3
        elif os.path.basename(folder) == 'L-手臂主体':
            foldnum = 4
        elif os.path.basename(folder) == 'L-握(C字型)':
            foldnum = 5
        elif os.path.basename(folder) == 'L-握拳':
            foldnum = 6
        elif os.path.basename(folder) == 'L-抓取':
            foldnum = 7
        elif os.path.basename(folder) == 'R-good手势':
            foldnum = 8
        elif os.path.basename(folder) == 'R-大拇指夹持':
            foldnum = 9
        elif os.path.basename(folder) == 'R-捏':
            foldnum = 10
        elif os.path.basename(folder) == 'R-食指甚至':
            foldnum = 11
        elif os.path.basename(folder) == 'R-手臂主体':
            foldnum = 12
        elif os.path.basename(folder) == 'R-手掌端着':
            foldnum = 13
        elif os.path.basename(folder) == 'R-握(C字型)':
            foldnum = 14
        elif os.path.basename(folder) == 'R-握笔':
            foldnum = 15
        elif os.path.basename(folder) == 'R-握拳':
            foldnum = 16
        elif os.path.basename(folder) == 'R-单指':
            foldnum = 17
        elif os.path.basename(folder) == 'R-抓取':
            foldnum = 18
        elif os.path.basename(folder) == 'two':
            foldnum = 19
        elif os.path.basename(folder) == 'other':
            foldnum = 20
        inner_path = os.path.join(outer_path, folder)
        total_num_folder = len(folderlist)  # 文件夹的总数
        print('total have %d folders' % (total_num_folder))  # 打印文件夹的总数
        filelist = os.listdir(inner_path)  # 列举图片
        i = 0
        for item in filelist:
            total_num_file = len(filelist)  # 单个文件夹内图片的总数
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(inner_path), item)  # 原图的地址
                dst = os.path.join(os.path.abspath(rename_path), str(foldnum) + '_' + str(
                    i) + '.jpg')  # 新图的地址（这里可以把str(folder) + '_' + str(i) + '.jpg'改成你想改的名称）
            try:
                os.rename(src, dst)
                # print 'converting %s to %s ...' % (src, dst)
                i += 1
            except:
                continue


# 训练集
rename_path1 = 'H:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/renametrain'
outer_path1 = 'H:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/train'
folderlist1 = os.listdir(r"H:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/train")
rename(rename_path1, outer_path1, folderlist1)
print("train totally rename ! ! !")
# 测试集
rename_path2 = 'H:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/renametest'
outer_path2 = 'H:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/val'
folderlist2 = os.listdir(r"HH:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/val")
rename(rename_path2, outer_path2, folderlist2)
print("test totally rename ! ! !")


# 修改图片尺寸
def convertjpg(jpgfile, outdir, width=32, height=32):
    img = Image.open(jpgfile)
    img = img.convert('RGB')
    img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    new_img = img.resize((width, height), Image.BILINEAR)
    new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))


# 训练集
for jpgfile in glob.glob("H:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/renametrain/*.jpg"):
    convertjpg(jpgfile, "H:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/data")
print("train totally resize ! ! !")
# 测试集
for jpgfile in glob.glob("H:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/*.jpg"):
    convertjpg(jpgfile, "H:/桌面/笔记/计算机视觉/TensorFlow/0mine-002-手势识别/test")

print("test totally resize ! ! !")

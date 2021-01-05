import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

mask_path='/content/gdrive/Colab Notebooks/UMA-master/result/egtea2yale_round00'
images_path='/content/gdrive/Colab Notebooks/UMA-master/data/Yale_Human_Grasp/test'
cut_path = '/content/gdrive/Colab Notebooks/UMA-master/data/cut'
    
# def getMaskedIMG(fileName,handFlag,mask):
    
#         img_path = os.path.join(images_path, fileName)
#         img = cv2.imread(img_path)
#         masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)  #将image的相素值和mask像素值相加得到结果
#         cv2.imwrite(os.path.join(cut_path, fileName[:-4]+handFlag+'.png'),masked)
        
def main():


    for img_item in os.listdir(mask_path):
        print(img_item)
        img_path = os.path.join(mask_path, img_item)
       
        # 1.导入图片
        img_src = cv2.imread(img_path)
        # 2.灰度化与二值化
        img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
        # 3.连通域分析
        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
        # 4.轮廓面积打印
        img_contours = []
        max_idx=-1
        max_area=0
        second_idx=-1
        second_area=0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            print("轮廓 %d 的面积是:%d" % (i, area))
            #更新最大面积 并记录index
            if area>max_area:
                #之前的最大变成第二大
                second_area=max_area
                second_idx=max_idx
                max_area=area
                max_idx=i
            #更第二最大面积 并记录index
            elif area>second_area:
                second_area=area
                second_idx=i
            img_temp = np.zeros(img_src.shape, np.uint8)
            img_contours.append(img_temp)
    
            cv2.drawContours(img_contours[i], contours, i, (255, 255, 255), -1)
            # cv2.imshow("%d" % i, img_contours[i])
        #把至多两张图片存储下来
        size_threshold=200
        if max_area>size_threshold:
            #找到两张有效区域  最大的可能是右手
            print(second_area,' ',second_idx,' ',max_area,' ',max_idx)
            if second_area>size_threshold and second_idx<max_idx:
                #之前输出的mask好死不死加了个_p.png的后缀 跟原图名字不一样这里要把_p去掉
                # getMaskedIMG(img_item[:-6]+'.png','L',img_contours[max_idx])
                # getMaskedIMG(img_item[:-6]+'.png','R',img_contours[second_idx])
                cv2.imwrite(os.path.join(cut_path, img_item[:-6]+'L.png'), img_contours[max_idx])
                cv2.imwrite(os.path.join(cut_path, img_item[:-6]+'R.png'), img_contours[second_idx])
            #找到两张有效区域  最大的可能是左手
            elif second_area>size_threshold and second_idx>=max_idx:
                # getMaskedIMG(img_item[:-6]+'.png','R',img_contours[max_idx])
                # getMaskedIMG(img_item[:-6]+'.png','L',img_contours[second_idx])
                cv2.imwrite(os.path.join(cut_path, img_item[:-6]+'R.png'), img_contours[max_idx])
                cv2.imwrite(os.path.join(cut_path, img_item[:-6]+'L.png'), img_contours[second_idx])
            elif second_area<size_threshold:
            #只有一个有效区域 默认是右手
                # getMaskedIMG(img_item[:-6]+'.png','R',img_contours[max_idx])
                 cv2.imwrite(os.path.join(cut_path, img_item[:-6]+'R.png'), img_contours[max_idx])


if __name__ == '__main__':
    main()
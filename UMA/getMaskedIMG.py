
import os
import cv2
import numpy as np

def add_mask2image_binary(images_path, masks_path, masked_path):
    # Add binary masks to images
    # for img_item in os.listdir(images_path):
    #     print(img_item)
    #     img_path = os.path.join(images_path, img_item)
    #     img = cv2.imread(img_path)
    #     mask_path = os.path.join(masks_path, img_item[:-4]+'_p.png')  # mask是.png格式的，image是.jpg格式的
    #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
    #     masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)  #将image的相素值和mask像素值相加得到结果
    #     cv2.imwrite(os.path.join(masked_path, img_item), masked)
    for mask_item in os.listdir(masks_path):
        print(mask_item)
        mask_path = os.path.join(masks_path, mask_item)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)           # 将彩色mask以二值图像形式读取
        img_path = os.path.join(images_path, mask_item[:-5]+'.png')  # mask是.png格式的，image是.jpg格式的
        img = cv2.imread(img_path) 
        masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)  #将image的相素值和mask像素值相加得到结果
        cv2.imwrite(os.path.join(masked_path, mask_item), masked)
images_path = '/content/gdrive/Colab Notebooks/UMA-master/data/Yale_Human_Grasp/test'
masks_path = '/content/gdrive/Colab Notebooks/UMA-master/data/cut'
masked_path = '/content/gdrive/Colab Notebooks/UMA-master/data/cut_img'
add_mask2image_binary(images_path, masks_path, masked_path)
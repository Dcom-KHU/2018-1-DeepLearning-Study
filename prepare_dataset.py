import numpy as np
import glob
import scipy as scp
import scipy.misc
import random
import scipy
from PIL import Image

def prepare_img(class_number) :
    img_c0 = []
    img_c1 = []
    img_c2 = []
    img_c3 = []
    img_c4 = []
    img_c5 = []
    img_c6 = []
    img_c7 = []
    img_c8 = []
    img_c9 = []

    for i in glob.glob('C:\\new_img\\0\\*.jpg'):
        img = scp.misc.imread(i)
        img = scp.misc.imresize(img,(192, 192))
        img_c0.append(img)

    for i in glob.glob('C:\\new_img\\1\\*.jpg'):
        img = scp.misc.imread(i)
        img = scp.misc.imresize(img,(192, 192))
        img_c1.append(img)

    for i in glob.glob('C:\\new_img\\2\\*.jpg'):
        img = scp.misc.imread(i)
        img = scp.misc.imresize(img,(192, 192))
        img_c2.append(img)

    for i in glob.glob('C:\\new_img\\3\\*.jpg'):
        img = scp.misc.imread(i)
        img = scp.misc.imresize(img,(192, 192))
        img_c3.append(img)

    for i in glob.glob('C:\\new_img\\4\\*.jpg'):
        img = scp.misc.imread(i)
        img = scp.misc.imresize(img,(192, 192))
        img_c4.append(img)

    for i in glob.glob('C:\\new_img\\5\\*.jpg'):
        img = scp.misc.imread(i)
        img = scp.misc.imresize(img,(192, 192))
        img_c5.append(img)

    for i in glob.glob('C:\\new_img\\6\\*.jpg'):
        img = scp.misc.imread(i)
        img = scp.misc.imresize(img,(192, 192))
        img_c6.append(img)

    for i in glob.glob('C:\\new_img\\7\\*.jpg'):
        img = scp.misc.imread(i)
        img = scp.misc.imresize(img,(192, 192))
        img_c7.append(img)

    for i in glob.glob('C:\\new_img\\8\\*.jpg'):
        img = scp.misc.imread(i)
        img = scp.misc.imresize(img,(192, 192))
        img_c8.append(img)

    for i in glob.glob('C:\\new_img\\9\\*.jpg'):
        img = scp.misc.imread(i)
        img = scp.misc.imresize(img,(192, 192))
        img_c9.append(img)


    img_c0 = np.array(img_c0)
    img_c1 = np.array(img_c1)
    img_c2 = np.array(img_c2)
    img_c3 = np.array(img_c3)
    img_c4 = np.array(img_c4)
    img_c5 = np.array(img_c5)
    img_c6 = np.array(img_c6)
    img_c7 = np.array(img_c7)
    img_c8 = np.array(img_c8)
    img_c9 = np.array(img_c9)

    img_c0 = img_c0.reshape(-1, 3, 192, 192).astype(np.float32)
    img_c0 = img_c0.transpose(0,2,3,1)
    img_c1 = img_c1.reshape(-1, 3, 192, 192).astype(np.float32)
    img_c1 = img_c1.transpose(0, 2, 3, 1)
    img_c2 = img_c2.reshape(-1, 3, 192, 192).astype(np.float32)
    img_c2 = img_c2.transpose(0, 2, 3, 1)
    img_c3 = img_c3.reshape(-1, 3, 192, 192).astype(np.float32)
    img_c3 = img_c3.transpose(0, 2, 3, 1)
    img_c4 = img_c4.reshape(-1, 3, 192, 192).astype(np.float32)
    img_c4 = img_c4.transpose(0, 2, 3, 1)
    img_c5 = img_c5.reshape(-1, 3, 192, 192).astype(np.float32)
    img_c5 = img_c5.transpose(0, 2, 3, 1)
    img_c6 = img_c6.reshape(-1, 3, 192, 192).astype(np.float32)
    img_c6 = img_c6.transpose(0, 2, 3, 1)
    img_c7 = img_c7.reshape(-1, 3, 192, 192).astype(np.float32)
    img_c7 = img_c7.transpose(0, 2, 3, 1)
    img_c8 = img_c8.reshape(-1, 3, 192, 192).astype(np.float32)
    img_c8 = img_c8.transpose(0, 2, 3, 1)
    img_c9 = img_c9.reshape(-1, 3, 192, 192).astype(np.float32)
    img_c9 = img_c9.transpose(0, 2, 3, 1)

    # label
    c0_label = np.zeros((np.size(img_c0, 0), class_number))
    c0_label[:, 0] = 1
    c1_label = np.zeros((np.size(img_c1, 0), class_number))
    c1_label[:, 1] = 1
    c2_label = np.zeros((np.size(img_c2, 0), class_number))
    c2_label[:, 2] = 1
    c3_label = np.zeros((np.size(img_c3, 0), class_number))
    c3_label[:, 3] = 1
    c4_label = np.zeros((np.size(img_c4, 0), class_number))
    c4_label[:, 4] = 1
    c5_label = np.zeros((np.size(img_c5, 0), class_number))
    c5_label[:, 5] = 1
    c6_label = np.zeros((np.size(img_c6, 0), class_number))
    c6_label[:, 6] = 1
    c7_label = np.zeros((np.size(img_c7, 0), class_number))
    c7_label[:, 7] = 1
    c8_label = np.zeros((np.size(img_c8, 0), class_number))
    c8_label[:, 8] = 1
    c9_label = np.zeros((np.size(img_c9, 0), class_number))
    c9_label[:, 9] = 1


    img = np.concatenate((img_c0,img_c1,img_c2,img_c3,img_c4,img_c5,img_c6,img_c7,img_c8,img_c9), 0)
    label = np.concatenate((c0_label,c1_label,c2_label,c3_label,c4_label,c5_label,c6_label,c7_label,c8_label,c9_label), 0)

    data = list(zip(img, label))

    print('이미지 로드드 완료')

    return data


def get_batch_images(batch_size,count,data):
    total_length = len(data)
    repeat = total_length / batch_size
    remain = total_length % batch_size
    batch_start = batch_size * count

    result_img = []
    result_label = []

    if batch_size == total_length:
        for i in range(total_length):
            temp = data[i]
            result_img.append(temp[0])
            result_label.append(temp[1])
        return result_img,result_label

    if (batch_start + remain) == total_length:
        for batch_start in range(batch_start,total_length):
            temp = data[batch_start]
            result_img.append(temp[0])
            result_label.append(temp[1])

        result_img = np.array(result_img)
        result_label = np.array(result_label)
        return result_img,result_label

    else:
        for batch_start in range(batch_start,batch_start + batch_size):
            temp = data[batch_start]
            result_img.append(temp[0])
            result_label.append(temp[1])

        result_img = np.array(result_img)
        result_label = np.array(result_label)
        return result_img, result_label
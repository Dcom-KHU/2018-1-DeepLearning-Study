import numpy as np
import glob
import scipy as scp
import scipy.misc


def prepare_data(class_number):
    source_img_dog_list = []
    source_img_cat_list = []

    for i in glob.glob('C:\\train\\dog\\temp\\*.jpg'):
        img = scp.misc.imread(i)
        img = scp.misc.imresize(img, (128, 128))
        source_img_dog_list.append(img)
    for i in glob.glob('C:\\train\\cat\\temp\\*.jpg'):
        img = scp.misc.imread(i)
        img = scp.misc.imresize(img, (128, 128))
        source_img_cat_list.append(img)
    source_img_dog_list = np.array(source_img_dog_list)
    source_img_cat_list = np.array(source_img_cat_list)

    source_img_dog_list = source_img_dog_list.reshape(-1, 3, 128, 128).astype(np.float32)
    source_img_dog_list = source_img_dog_list.transpose(0, 2, 3, 1)
    source_img_cat_list = source_img_cat_list.reshape(-1, 3, 128, 128).astype(np.float32)
    source_img_cat_list = source_img_cat_list.transpose(0, 2, 3, 1)

    label_dog = np.zeros((len(source_img_dog_list), class_number))
    label_cat = np.zeros((len(source_img_cat_list), class_number))
    label_dog[:, 0] = 1
    label_cat[:, 1] = 1

    source_img = np.concatenate((source_img_dog_list, source_img_cat_list), 0)
    label = np.concatenate((label_dog, label_cat), 0)

    data = list(zip(source_img, label))

    print("Image load complete!")

    return data


def get_batch_data(batch_size, count, data):
    total_length = len(data)
    try:
        repeat = total_length / batch_size
        remain = total_length % batch_size
    except ZeroDivisionError:
        print("ZeroDivision")
    batch_start = batch_size * count

    result_source_img = []
    result_label = []

    if batch_size == total_length:
        for i in range(total_length):
            temp = data[i]
            result_source_img.append(temp[0])
            result_label.append(temp[1])

        result_source_img = np.array(result_source_img)
        result_label = np.array(result_label)

        return result_source_img, result_label

    if (batch_start + remain) == total_length:
        for i in range(batch_start, total_length):
            temp = data[i]
            result_source_img.append(temp[0])
            result_label.append(temp[1])

        result_source_img = np.array(result_source_img)
        result_label = np.array(result_label)

        return result_source_img, result_label

    else:
        for i in range(batch_start, batch_start + batch_size):
            temp = data[i]
            result_source_img.append(temp[0])
            result_label.append(temp[1])

        result_source_img = np.array(result_source_img)
        result_label = np.array(result_label)

        return result_source_img, result_label

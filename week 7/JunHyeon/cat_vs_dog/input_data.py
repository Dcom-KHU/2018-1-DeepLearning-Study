import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import math

# def get_files(file_dir):
# 	cats = []
# 	dogs = []
# 	label_dogs = []
# 	label_cats = []
# 	for file in os.listdir(file_dir):
# 		name = file.split(sep = ".")
# 		if name[0]=="dog":
# 			dogs.append(file_dir + file)
# 			label_dogs.append(1)
# 		else:
# 			cats.append(file_dir + file)
# 			label_cats.append(0)
# 	print("There are %d cats. \nThere are %d dogs. " %(len(cats), len(dogs)))


# 	image_list = np.hstack((cats,dogs))
# 	label_list = np.hstack((label_cats,label_dogs))

# 	temp = np.array([image_list,label_list])
# 	# transpose temp from [2,25000] to [25000,2]
# 	temp = temp.transpose()
# 	# shuffle temp
# 	np.random.shuffle(temp)

# 	image_list = list(temp[:,0])
# 	label_list = list(temp[:,1])
# 	# translate label_list class from numpy.str to int
# 	label_list = [int(i) for i in label_list]

# 	return image_list, label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):
	image = tf.cast(image, tf.string) # image 자체가 현재는 파일 이름을 나타냄. -> tf.string을 할 필요 있음
	label = tf.cast(label, tf.int32) # label은 int형 
	input_queue = tf.train.slice_input_producer([image,label]) #앞에서 배열을 transpose 한 이유
	label = input_queue[1] 
	image_contents = tf.read_file(input_queue[0])
	image = tf.image.decode_jpeg(image_contents, channels = 3) 

	image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
	image = tf.image.per_image_standardization(image)

	image_batch, label_batch = tf.train.batch([image, label],
						batch_size = batch_size,
						num_threads = 512,
						capacity = capacity)
	label_batch = tf.reshape(label_batch,[batch_size])
	image_batch = tf.cast(image_batch, tf.float32)
	return image_batch, label_batch

if __name__ == '__main__':
    get_files(train_dir)

def get_files(file_dir, ratio):
    """
    Args:
        file_dir: file directory
        ratio:ratio of validation datasets
    Returns:
        list of images and labels
    """
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir): #경로안의 파일들 반환
        name = file.split(sep='.')
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0) #리스트에 추가
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))

    image_list = np.hstack((cats, dogs)) #옆으로 더한다 = 열 증가
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]

    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample*ratio) # number of validation samples
    n_train = n_sample - n_val # number of trainning samples

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels

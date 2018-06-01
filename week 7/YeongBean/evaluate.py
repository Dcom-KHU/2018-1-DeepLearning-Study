import os
import numpy as np
import tensorflow as tf
import input_data
import model

# you need to change the directories to yours.
train_dir = "C://Users//Bean//Documents//Python Scripts//MyCNN//train/"
test_dir = "C://Users//Bean//Documents//Python Scripts//MyCNN//logs//trainer/"
train_logs_dir = './logs/train/'
val_logs_dir = './logs/val'



#Test one image

def get_one_image(file_dir):
    """
    Randomly pick one image from test data
    Return: ndarray
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    test =[]
    for file in os.listdir(file_dir):
        test.append(file_dir + file)
    #print('There are %d test pictures' %(len(test)))

    n = len(test)
    ind = np.random.randint(0, n)
    print('%dth picture is...' %ind)
    img_test = test[ind]
    print(test[ind])

    image = Image.open(img_test)
    plt.imshow(image)
    image = image.resize([208, 208])
    image = np.array(image)
    return image

def test_one_image():
    """
    Test one image with the saved models and parameters
    """

    test_image = get_one_image(test_dir)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(test_image, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        saver = tf.train.Saver()

        with tf.Session() as sess:

            #print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(train_logs_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: test_image})
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a cat with possibility %.6f\n' %prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f\n' %prediction[:, 1])


if __name__ == '__main__':
    for i in range(0, 1):
        test_one_image()








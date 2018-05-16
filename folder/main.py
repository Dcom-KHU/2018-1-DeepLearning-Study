import tensorflow as tf
import random
from prepare import prepare_data, get_batch_data


class_num = 2
learning_rate = 0.01
training_epochs = 10
batch_size = 40
width = 128
height = 128
data = prepare_data(class_num)

keep_prob = tf.placeholder(tf.float32)
input_image = tf.placeholder(tf.float32, [None, width, height, 3])
label = tf.placeholder(tf.float32, [None, class_num])

filters = {
    'cf1': tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.01)),
    'cf2': tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01)),
    'cf3': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01)),
    'cf4': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
}

cl1 = tf.nn.conv2d(input_image, filters['cf1'], strides=[1, 1, 1, 1], padding='SAME')
cl1 = tf.nn.max_pool(cl1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#cl1 : [-1, width / 2, height / 2, 32]

cl2 = tf.nn.conv2d(cl1, filters['cf2'], strides=[1, 1, 1, 1], padding='SAME')
cl2 = tf.nn.relu(cl2)
cl2 = tf.nn.max_pool(cl2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
cl2 = tf.nn.dropout(cl2, keep_prob=keep_prob)
#cl2 : [-1, width / 4, height / 4, 64]

cl3 = tf.nn.conv2d(cl2, filters['cf3'], strides=[1, 1, 1, 1],  padding='SAME')
cl3 = tf.nn.max_pool(cl3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#cl3 : [-1, width / 8, height / 8, 64]

cl4 = tf.nn.conv2d(cl3, filters['cf4'], strides=[1, 1, 1, 1],  padding='SAME')
cl4 = tf.nn.relu(cl4)
cl4 = tf.nn.max_pool(cl4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
cl4 = tf.nn.dropout(cl4, keep_prob=keep_prob)
#cl4 : [-1, width / 16, height / 16, 128]

flat_cl4 = tf.reshape(cl4, [-1, int((width / 16) * (height / 16) * 128)])

fc1_w = tf.get_variable("fc1_w", shape=[(width / 16) * (height / 16) * 128, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
fc1 = tf.nn.relu(tf.matmul(flat_cl4, fc1_w) + b1)
fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

fc2_w = tf.get_variable("fc2_w", shape=[256, 2], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([2]))
classify_layer = tf.matmul(fc1, fc2_w) + b2

classify_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=classify_layer, labels=label), name='Classification_Cost')
classify_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(classify_cost)
classify_prediction = tf.equal(tf.argmax(classify_layer, 1), tf.argmax(label, 1))
classify_accuracy = tf.reduce_mean(tf.cast(classify_prediction, tf.float32), name='Classification_Accuracy')

with tf.Session() as sess:
    for i in range(training_epochs):
        print("Training Start!!")
        sess.run(tf.global_variables_initializer())

        random.shuffle(data)
        training_data = data[:int(len(data) * 0.9)]
        test_data = data[int(len(data) * 0.9) + 1:]
        test_source_img, test_label = get_batch_data(int(len(test_data)), 0, test_data)

        random.shuffle(training_data)
        eval_data = training_data[int(len(training_data)*0.9)+1:]
        eval_source_img, eval_label = get_batch_data(int(len(eval_data)), 0, eval_data)

        training_steps = int(len(training_data) / batch_size)

        for j in range(training_steps):
            batch_s_img, batch_l = get_batch_data(batch_size, j, training_data)
            class_cost, _, = sess.run([classify_cost, classify_optimizer], feed_dict={input_image: batch_s_img, label: batch_l, keep_prob: 0.8})
            if (j % (batch_size / 2)) == 0:
                eval_c_cost, eval_c_acc = sess.run([classify_cost, classify_accuracy], feed_dict={input_image: eval_source_img, label: eval_label, keep_prob: 1.0})
                print("Evaluation Classification Cost : ", '{:5f}'.format(eval_c_cost), "\nEvaluation Classification Accuracy : ", '{:5f}'.format(eval_c_acc))

        test_c_cost, test_c_acc = sess.run([classify_cost, classify_accuracy], feed_dict={input_image: test_source_img, label: test_label, keep_prob: 1.0})
        print("Training ", (i + 1), "results : ")
        print("Classification Cost : ", '{:5f}'.format(test_c_cost), "\nClassification Accuracy : ", '{:5f}'.format(test_c_acc))
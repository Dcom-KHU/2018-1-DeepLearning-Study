import tensorflow as tf
import numpy as np
import random
from prepare_dataset import prepare_img, get_batch_images

img_data = prepare_img(10)

learning_rate = 0.01
training_epochs = 10
batch_size = 100

init_width = 192
init_height = 192
class_num = 10


keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32,[None,init_width,init_height,3])
y = tf.placeholder(tf.float32,[None,class_num])

w1 = tf.Variable(tf.random_normal([3,3,3,8],stddev=0.001))
l1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME')
l1 = tf.nn.relu(l1)
l1 = tf.nn.max_pool(l1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
l1 = tf.nn.dropout(l1,keep_prob=keep_prob)
# l1 = (-1,96,96,8)

w2 = tf.Variable(tf.random_normal([3,3,8,16],stddev=0.001))
l2 = tf.nn.conv2d(l1,w2,strides=[1,1,1,1],padding='SAME')
l2 = tf.nn.relu(l2)
l2 = tf.nn.max_pool(l2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
l2 = tf.nn.dropout(l2,keep_prob=keep_prob)
# l2 = (-1,48,48,16)

w3 = tf.Variable(tf.random_normal([3,3,16,32],stddev=0.001))
l3 = tf.nn.conv2d(l2,w3,strides=[1,1,1,1],padding='SAME')
l3 = tf.nn.relu(l3)
l3 = tf.nn.max_pool(l3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
l3 = tf.nn.dropout(l3,keep_prob=keep_prob)
# l3 = (-1,24,24,32)

w4 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.001))
l4 = tf.nn.conv2d(l3,w4,strides=[1,1,1,1],padding='SAME')
l4 = tf.nn.relu(l4)
l4 = tf.nn.max_pool(l4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
l4 = tf.nn.dropout(l4,keep_prob=keep_prob)
# l4 = (-1,12,12,64)

w5 = tf.Variable(tf.random_normal([3,3,64,128],stddev=0.001))
l5 = tf.nn.conv2d(l4,w5,strides=[1,1,1,1],padding='SAME')
l5 = tf.nn.relu(l5)
l5 = tf.nn.max_pool(l5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
l5 = tf.nn.dropout(l5,keep_prob=keep_prob)
# l5 = (-1,6,6,128)
l5_flat = tf.reshape(l5,[-1,128 * 6 * 6])

fc1 = tf.get_variable("fc1",shape=[128 * 6 * 6, 512],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
l6 = tf.nn.relu(tf.matmul(l5_flat,fc1)+b1)
l6 = tf.nn.dropout(l6,keep_prob=keep_prob)

fc2 = tf.get_variable("fc2",shape=[512, 10],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([10]))
result = tf.matmul(l6,fc2)+b2


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=y), name='Cost')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='Accuracy')

with tf.Session() as sess:
    for i in range(training_epochs):
        print("Training Start!!")
        sess.run(tf.global_variables_initializer())

        random.shuffle(img_data)
        training_data = img_data[:int(len(img_data) * 0.9)]
        test_data = img_data[int(len(img_data) * 0.9) + 1:]
        test_x, test_y = get_batch_images(len(test_data), 0, test_data)

        random.shuffle(training_data)
        eval_data = training_data[int(len(training_data)*0.9)+1:]
        eval_x, eval_y = get_batch_images(len(eval_data),0,eval_data)
        repeat_time = int(len(training_data) / batch_size)

        for k in range(repeat_time):
            batch_x, batch_y = get_batch_images(batch_size,k,training_data)
            temp_cost, _ = sess.run([cost,optimizer],feed_dict={x : batch_x, y : batch_y, keep_prob : 0.8})

            if k % 100 == 0:
                eval_cost, eval_acc = sess.run([cost, accuracy], feed_dict={x: eval_x, y: eval_y, keep_prob: 1})
                print("  Evaluation_Accuracy : ",'{:9f}'.format(eval_acc),
                      "  Evaluation_Cost : ",'{:9f}'.format(eval_cost))

        step_cost, step_acc = sess.run([cost,accuracy],feed_dict={x : test_x, y : test_y, keep_prob : 1})
        print("Complete Training : ",'%04d' %  (i + 1),"  Test Cost : ",'{:8f}'
              .format(step_cost),"  Test Accuracy : ",'{:8f}'.format(step_acc))
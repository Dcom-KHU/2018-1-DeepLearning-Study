import tensorflow as tf


def inference(images, batch_size, n_classes):

    with tf.variable_scope('block1') as scope:
        with tf.variable_scope('conv1') as scope:
            conv1 = tf.layers.conv2d(
                                inputs=images,
                                filters=64,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu)

        with tf.variable_scope('conv2') as scope:
            conv2 = tf.layers.conv2d(
                                inputs=conv1,
                                filters=64,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu)


        with tf.variable_scope('pool1') as scope:
            pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


    with tf.variable_scope('block2') as scope:
        with tf.variable_scope('conv3') as scope:
            conv3 = tf.layers.conv2d(
                                inputs=pool1,
                                filters=128,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu)

        with tf.variable_scope('conv4') as scope:
            conv4 = tf.layers.conv2d(
                                inputs=conv3,
                                filters=128,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu)

        with tf.variable_scope('pool2') as scope:
            pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)


    with tf.variable_scope('block3') as scope:
        with tf.variable_scope('conv5') as scope:
            conv5 = tf.layers.conv2d(
                                inputs=pool2,
                                filters=256,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu)

        with tf.variable_scope('conv6') as scope:
            conv6 = tf.layers.conv2d(
                                inputs=conv5,
                                filters=256,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu)

        with tf.variable_scope('conv7') as scope:
            conv7 = tf.layers.conv2d(
                                inputs=conv6,
                                filters=256,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu)

        with tf.variable_scope('pool3') as scope:
            pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)


    with tf.variable_scope('block4') as scope:
        with tf.variable_scope('conv8') as scope:
            conv8 = tf.layers.conv2d(
                                inputs=pool3,
                                filters=512,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu)

        with tf.variable_scope('conv9') as scope:
            conv9 = tf.layers.conv2d(
                                inputs=conv8,
                                filters=512,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu)
        
        with tf.variable_scope('conv10') as scope:
            conv10 = tf.layers.conv2d(
                                inputs=conv9,
                                filters=512,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu)

        with tf.variable_scope('pool3') as scope:
            pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)



    with tf.variable_scope('block5') as scope:
        with tf.variable_scope('conv11') as scope:
            conv11 = tf.layers.conv2d(
                                inputs=pool4,
                                filters=512,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu)

        with tf.variable_scope('conv12') as scope:
            conv12 = tf.layers.conv2d(
                                inputs=conv11,
                                filters=512,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu)
        
        with tf.variable_scope('conv13') as scope:
            conv13 = tf.layers.conv2d(
                                inputs=conv12,
                                filters=512,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu)

        with tf.variable_scope('pool3') as scope:
            pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)

    with tf.variable_scope('block6') as scope:
        with tf.variable_scope('dense1') as scope:
            dense1 = tf.contrib.layers.flatten(pool5)
            dense1 = tf.layers.dense(dense1, 4096, activation = tf.nn.relu)

    
        with tf.variable_scope('dense2') as scope:
            dense2 = tf.contrib.layers.flatten(dense1)
            dense2 = tf.layers.dense(dense2, 4096, activation = tf.nn.relu)


        with tf.variable_scope('dense3') as scope:
            dense3 = tf.contrib.layers.flatten(dense2)
            dense3 = tf.layers.dense(dense3, 1000, activation = tf.nn.relu)

    logits = tf.layers.dense(inputs=dense3, units=2)
    logits = tf.nn.softmax(logits, name="logits")

    return logits

                            
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels, name='xentropy_per_example')
        #loss = tf.reduce_mean(cross_entropy, name='loss')
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss


def trainning(loss, learning_rate):
    
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy
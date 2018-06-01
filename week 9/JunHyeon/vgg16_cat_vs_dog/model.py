import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def make_network(images, labels, keep_prob):

    L1 = images

    with tf.variable_scope('conv_Block1'):
        L1 = tf.layers.conv2d(L1, 64, [3,3], activation=tf.nn.relu, padding='SAME', name='block1_conv1')
        L1 = tf.layers.conv2d(L1, 64, [3,3], activation=tf.nn.relu, padding='SAME', name='block1_conv2')
        L1 = tf.layers.max_pooling2d(L1, [2,2], [2,2], name='block1_pool')

    print(L1)

    with tf.variable_scope('conv_Block2'):
        L2 = tf.layers.conv2d(L1, 128, [3,3], activation=tf.nn.relu, padding='SAME', name='block2_conv1')
        L2 = tf.layers.conv2d(L2, 128, [3,3], activation=tf.nn.relu, padding='SAME', name='block2_conv2')
        L2 = tf.layers.max_pooling2d(L2, [2,2], [2,2], name='block2_pool')

    print(L2)

    with tf.variable_scope('conv_Block3'):
        L3 = tf.layers.conv2d(L2, 256, [3,3], activation=tf.nn.relu, padding='SAME', name='block3_conv1')
        L3 = tf.layers.conv2d(L3, 256, [3,3], activation=tf.nn.relu, padding='SAME', name='block3_conv2')
        L3 = tf.layers.conv2d(L3, 256, [3,3], activation=tf.nn.relu, padding='SAME', name='block3_conv3')
        L3 = tf.layers.max_pooling2d(L3, [2,2], [2,2], name='block3_pool')        

    print(L3)

    with tf.variable_scope('conv_Block4'):
        L4 = tf.layers.conv2d(L3, 512, [3,3], activation=tf.nn.relu, padding='SAME', name='block4_conv1')
        L4 = tf.layers.conv2d(L4, 512, [3,3], activation=tf.nn.relu, padding='SAME', name='block4_conv2')
        L4 = tf.layers.conv2d(L4, 512, [3,3], activation=tf.nn.relu, padding='SAME', name='block4_conv3')
        L4 = tf.layers.max_pooling2d(L4, [2,2], [2,2], name='block4_pool')
            
    print(L4)

    with tf.variable_scope('conv_Block5'):
        L5 = tf.layers.conv2d(L4, 512, [3,3], activation=tf.nn.relu, padding='SAME', name='block5_conv1')
        L5 = tf.layers.conv2d(L5, 512, [3,3], activation=tf.nn.relu, padding='SAME', name='block5_conv2')
        L5 = tf.layers.conv2d(L5, 512, [3,3], activation=tf.nn.relu, padding='SAME', name='block5_conv3')
        L5 = tf.layers.max_pooling2d(L5, [2,2], [2,2], name='block5_pool')

    print(L5)
    
    with tf.variable_scope('FullyConnected'):
        fc_layer = tf.contrib.layers.flatten(L5)
        print(fc_layer)
        fc_layer = tf.layers.dense(fc_layer, 4096, activation=tf.nn.relu, name='fc_layer1')
        fc_layer = tf.layers.dropout(fc_layer, keep_prob)
        fc_layer = tf.layers.dense(fc_layer, 4096, activation=tf.nn.relu, name='fc_layer2')
        fc_layer = tf.layers.dropout(fc_layer, keep_prob)
        print(fc_layer)

    with tf.variable_scope('Softmax'):
        hypothesis = tf.layers.dense(fc_layer, 2, activation=None, name='softmax')
        print(hypothesis)
    return hypothesis

def losses(logits, labels):
    """
    Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
    Returns:
        loss tensor of float type
    """
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss


def trainning(loss, learning_rate):
    """
    Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.
    Args:
        loss: loss tensor, from losses()
    Returns:
        train_op: The op for trainning
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op


def evaluation(logits, labels):
    """
    Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy
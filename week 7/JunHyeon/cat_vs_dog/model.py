import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def make_network(images, labels, keep_prob):
    num_conv = FLAGS.num_conv
    kernel_size = FLAGS.kernel_size
    pool_size = FLAGS.pool_size
    num_map = FLAGS.num_map
    num_fc_layer = FLAGS.num_fc_layer
    num_fc_input = FLAGS.num_fc_input

    height = FLAGS.height
    width = FLAGS.width
    prev_num_map = FLAGS.depth
    h_pool = images
    with tf.variable_scope('conv_layer'):
        for i in range(num_conv):
            weight = tf.get_variable('weight'+ str(i+1), 
                                        shape = [kernel_size, kernel_size, prev_num_map, num_map],
                                        dtype = tf.float32,
                                        initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            bias = tf.get_variable('bias' + str(i+1), 
                                        shape = [num_map],
                                        dtype = tf.float32, 
                                        initializer = tf.constant_initializer(0.1))

            print(h_pool)
            
            h_conv = tf.nn.conv2d(h_pool, weight, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(h_conv, bias)
            h_relu = tf.nn.relu(pre_activation, name='relu'+str(i+1))
            h_pool = tf.nn.max_pool(h_relu, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME', name='pool'+str(i+1))

            prev_num_map = num_map
            num_map *= 2
            height /= 2
            width /= 2


    h_pool = tf.nn.lrn(h_pool, depth_radius=4, bias=1.0, alpha=0.001112, beta=0.75,name='norm'+str(i+1))
    num_map /= 2
    h_fc_input = tf.reshape(h_pool, [-1, int(height) *int(width) * int(num_map)])
    prev_num_fc_input = height * width * num_map
    print(h_pool)

    with tf.variable_scope('fc_layer'):
        for i in range(num_fc_layer):
            W_fc = tf.get_variable('W_fc' + str(i+1),
                                        shape = [prev_num_fc_input, num_fc_input],
                                        dtype = tf.float32,
                                        initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            b_fc = tf.get_variable('b_fc' + str(i+1),
                                        shape = [num_fc_input],
                                        dtype = tf.float32,
                                        initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            h_fc = tf.nn.relu(tf.matmul(h_fc_input, W_fc) + b_fc)
            h_fc_input = tf.nn.dropout(h_fc, keep_prob)
            prev_num_fc_input = num_fc_input
            num_fc_input /= 2

    num_fc_input *= 2
    W_fc = tf.get_variable('W_fc' + str(i+2),
									shape = [num_fc_input, FLAGS.num_class],
									dtype = tf.float32,
									initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b_fc = tf.get_variable('b_fc' + str(i+2),  
									shape = [FLAGS.num_class],
									dtype = tf.float32,
									initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
                                                                    
    hypothesis = tf.matmul(h_fc_input, W_fc) + b_fc
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=hypothesis, labels=labels))
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    return hypothesis, cross_entropy, train_step

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
	    correct = tf.nn.in_top_k(logits, labels, 1)
	    correct = tf.cast(correct, tf.float16)
	    accuracy = tf.reduce_mean(correct)
    return accuracy

def val_make_network(images, labels, keep_prob):
    num_conv = FLAGS.num_conv
    kernel_size = FLAGS.kernel_size
    pool_size = FLAGS.pool_size
    num_map = FLAGS.num_map
    num_fc_layer = FLAGS.num_fc_layer
    num_fc_input = FLAGS.num_fc_input

    height = FLAGS.height
    width = FLAGS.width
    prev_num_map = FLAGS.depth
    h_pool = images
    with tf.variable_scope('val_conv_layer'):
        for i in range(num_conv):
            weight = tf.get_variable('val_weight'+ str(i+1), 
                                        shape = [kernel_size, kernel_size, prev_num_map, num_map],
                                        dtype = tf.float32,
                                        initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            bias = tf.get_variable('val_bias' + str(i+1), 
                                        shape = [num_map],
                                        dtype = tf.float32, 
                                        initializer = tf.constant_initializer(0.1))

            print(h_pool)
            
            h_conv = tf.nn.conv2d(h_pool, weight, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(h_conv, bias)
            h_relu = tf.nn.relu(pre_activation)
            h_pool = tf.nn.max_pool(h_relu, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME')

            prev_num_map = num_map
            num_map *= 2
            height /= 2
            width /= 2

    num_map /= 2
    h_pool = tf.nn.lrn(h_pool, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75)    
    h_fc_input = tf.reshape(h_pool, [-1, int(height) *int(width) * int(num_map)])
    prev_num_fc_input = height * width * num_map
    print(h_pool)

    with tf.variable_scope('val_fc_layer'):
        for i in range(num_fc_layer):
            W_fc = tf.get_variable('val_W_fc' + str(i+1),
                                        shape = [prev_num_fc_input, num_fc_input],
                                        dtype = tf.float32,
                                        initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            b_fc = tf.get_variable('val_b_fc' + str(i+1),
                                        shape = [num_fc_input],
                                        dtype = tf.float32,
                                        initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            h_fc = tf.nn.relu(tf.matmul(h_fc_input, W_fc) + b_fc)
            h_fc_input = tf.nn.dropout(h_fc, keep_prob)
            prev_num_fc_input = num_fc_input
            num_fc_input /= 2

    num_fc_input *= 2
    W_fc = tf.get_variable('val_W_fc' + str(i+2),
									shape = [num_fc_input, FLAGS.num_class],
									dtype = tf.float32,
									initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b_fc = tf.get_variable('val_b_fc' + str(i+2),  
									shape = [FLAGS.num_class],
									dtype = tf.float32,
									initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
                                                                    
    hypothesis = tf.matmul(h_fc_input, W_fc) + b_fc
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=hypothesis, labels=labels))
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    return hypothesis, cross_entropy, train_step
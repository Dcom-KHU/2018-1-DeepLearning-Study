import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import input_data
import config
import model

FLAGS = tf.app.flags.FLAGS



def run_training():
    train, train_label = input_data.get_files(FLAGS.train_dir)
    train_batch, train_label_batch = input_data.get_batch(train, train_label, FLAGS.height, FLAGS.width, FLAGS.batch_size, FLAGS.capacity)
    

    
    keep_prob = tf.placeholder(tf.float32)

    hypothesis, cross_entropy, train_step = model.make_network(train_batch, train_label_batch, keep_prob)

    cost_sum = tf.summary.scalar("cost", cross_entropy)
    saver = tf.train.Saver()

    train_acc = model.evaluation(hypothesis, train_label_batch)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(FLAGS.log_dir)
        writer.add_graph(sess.graph)  # Show the graph
 
        merge_sum = tf.summary.merge_all()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for step in np.arange(FLAGS.max_steps+1):
            _, summary, tra_loss, tra_acc = sess.run([train_step, merge_sum, cross_entropy, train_acc], feed_dict={keep_prob: 0.7})
            writer.add_summary(summary, global_step=step)

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
            if step % 2000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
       
        coord.request_stop()
        coord.join(threads)
        
        sess.close()

if __name__ == '__main__':
    run_training()

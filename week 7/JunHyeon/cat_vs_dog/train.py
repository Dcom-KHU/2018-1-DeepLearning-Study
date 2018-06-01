import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import input_data
import config
import model

FLAGS = tf.app.flags.FLAGS


def run_trainuating():
    
    train_data, train_label, val_data, val_label = input_data.get_files(FLAGS.train_dir, FLAGS.ratio)
    #train_data, train_label = input_data.get_files(FLAGS.train_dir)

    train_batch, train_label_batch = input_data.get_batch(train_data, train_label, FLAGS.height, FLAGS.width, FLAGS.batch_size, FLAGS.capacity)
    val_batch, val_label_batch = input_data.get_batch(val_data, val_label, FLAGS.height, FLAGS.width, FLAGS.batch_size, FLAGS.capacity)

    keep_prob = tf.placeholder(tf.float32)

    train_hypothesis, train_cross_entropy, train_step = model.make_network(train_batch, train_label_batch, keep_prob)
    val_hypothesis, val_cross_entropy, val_step = model.val_make_network(val_batch, val_label_batch, keep_prob)

    cost_sum_train = tf.summary.scalar("cost_train", train_cross_entropy)
    cost_sum_val = tf.summary.scalar("cost_val", val_cross_entropy)

    train_acc = model.evaluation(train_hypothesis, train_label_batch)
    val_acc = model.evaluation(val_hypothesis, val_label_batch)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        val_writer = tf.summary.FileWriter(FLAGS.val_dir, sess.graph)
        train_writer.add_graph(sess.graph)  # Show the graph
        val_writer.add_graph(sess.graph)  # Show the graph

        merge_sum = tf.summary.merge_all()      
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        for step in np.arange(FLAGS.max_steps+1):
            _, train_summary, train_loss, train_accuracy = sess.run([train_step, merge_sum, train_cross_entropy, train_acc], feed_dict={keep_prob: 0.7})
            train_writer.add_summary(train_summary, global_step=step)

            if step % 50 == 0:
                print('step : %d, loss : %f, train_accuracy : %f' % (step, train_loss, train_accuracy*100))
            if step % 200 == 0:
                _, val_summary, val_loss, val_accuracy = sess.run([val_step, merge_sum, val_cross_entropy, val_acc], feed_dict={keep_prob: 0.7})
                print('   **  step : %d, val_loss : %f, val_accuracy : %f  **' % (step, val_loss, val_accuracy*100))
                val_writer.add_summary(val_summary, global_step=step)


            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
        
        sess.close()

if __name__ == '__main__':
    run_trainuating()

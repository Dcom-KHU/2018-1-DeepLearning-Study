import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import input_data
import config
import model

FLAGS = tf.app.flags.FLAGS


def run_evaluating():
    
    eval_data, eval_label = input_data.get_files(FLAGS.eval_dir)
    eval_batch, eval_label_batch = input_data.get_batch(eval_data, eval_label, FLAGS.height, FLAGS.width, 512, FLAGS.capacity)

    
    keep_prob = tf.placeholder(tf.float32)

    hypothesis, cross_entropy, eval_step = model.make_network(eval_batch, eval_label_batch, keep_prob)

    cost_sum = tf.summary.scalar("cost_eval", cross_entropy)

    eval_accuracy = tf.nn.in_top_k(hypothesis, eval_label_batch, 1)
    eval_acc = model.evaluation(hypothesis, eval_label_batch)

    saver = tf.train.Saver()
    
    print ('Start Evaluation......')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        total_sample_count = FLAGS.eval_steps * 512
        true_count = 0

        writer = tf.summary.FileWriter(FLAGS.log_dir)
        writer.add_graph(sess.graph)  # Show the graph

        merge_sum = tf.summary.merge_all()      

        saver.restore(sess, './CNN_Homework/logs/model.ckpt-30000')  
        

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        for step in np.arange(FLAGS.eval_steps+1):
            _, summary, eval_loss = sess.run([eval_step, merge_sum, cross_entropy], feed_dict={keep_prob: 1.0})
            predictions, accuracy = sess.run([eval_accuracy, eval_acc], feed_dict={keep_prob: 1.0})
            writer.add_summary(summary, global_step=step)

            true_count = true_count + np.sum(predictions)

            if step % 10 == 0:
                print('step : %d, loss : %f, eval_accuracy : %f' % (step, eval_loss, accuracy*100))

        coord.request_stop()
        coord.join(threads)
        
        print ('precision : %f' % (true_count / total_sample_count))

        sess.close()

if __name__ == '__main__':
    run_evaluating()

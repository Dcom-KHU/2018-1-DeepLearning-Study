import os
import numpy as np
import tensorflow as tf
import input_data
import model
import config

FLAGS = tf.app.flags.FLAGS

# you need to change the directories to yours.
eval_dir = FLAGS.eval_dir
train_logs_dir = FLAGS.log_dir
eval_log_dir = FLAGS.eval_log_dir

def evaluating():

    keep_prob = tf.placeholder(tf.float32)

    eval_image, eval_label = input_data.get_eval_files(eval_dir)
    eval_batch, eval_label_batch = input_data.get_batch(
                                                        eval_image, 
                                                        eval_label, 
                                                        FLAGS.height, 
                                                        FLAGS.width, 
                                                        FLAGS.eval_batch_size, 
                                                        FLAGS.capacity)

    logits = model.make_network(eval_batch, eval_label_batch, keep_prob)
    loss = model.losses(logits, eval_label_batch)
    eval_op = model.trainning(loss, FLAGS.learning_rate)
    acc = model.evaluation(logits, eval_label_batch)

    x = tf.placeholder(tf.float32, shape=[FLAGS.eval_batch_size, FLAGS.height, FLAGS.width, 3])
    y_ = tf.placeholder(tf.int16, shape=[FLAGS.eval_batch_size])


    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        ckpt = tf.train.get_checkpoint_state(train_logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            #saver.restore(sess, './DeepLearning/vgg16/logs/model.ckpt')
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        
        tf.summary.image("ver1", eval_batch, 32);
        im = tf.summary.image("ver2", eval_batch, 32);
        
        summary_op = tf.summary.merge_all()
        eval_writer = tf.summary.FileWriter(eval_log_dir, sess.graph)

        try:
            for step in np.arange(FLAGS.eval_step):
                if coord.should_stop():
                        break
                e_images,e_labels = sess.run([eval_batch, eval_label_batch])
                _, e_loss, e_acc = sess.run([eval_op, loss, acc],
                                                feed_dict={x:e_images, y_:e_labels})

                summary_image = sess.run(tf.summary.image("ver3", e_images, 32));

                if step % 10 == 0:
                    print('Step %d, eval loss = %f, eval accuracy = %f%%' %(step, e_loss, e_acc*100.0))
                    summary_str = sess.run(summary_op)
                    im = sess.run(im)

                    eval_writer.add_summary(summary_str, step)
                    eval_writer.add_summary(summary_image, step)
                    eval_writer.add_summary(im, step)

        except tf.errors.OutOfRangeError:
            print('Done evaling -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    evaluating()








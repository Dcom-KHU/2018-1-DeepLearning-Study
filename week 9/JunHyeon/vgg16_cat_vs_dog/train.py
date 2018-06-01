import os
import numpy as np
import tensorflow as tf
import input_data
import model
import config

FLAGS = tf.app.flags.FLAGS

# you need to change the directories to yours.
train_dir = FLAGS.train_dir
train_logs_dir = FLAGS.log_dir
val_dir = FLAGS.val_dir

def training():

    keep_prob = tf.placeholder(tf.float32)

    train, train_label, val, val_label = input_data.get_files(train_dir, FLAGS.ratio)
    train_batch, train_label_batch = input_data.get_batch(
                                                        train, 
                                                        train_label, 
                                                        FLAGS.height, 
                                                        FLAGS.width, 
                                                        FLAGS.batch_size, 
                                                        FLAGS.capacity)
    val_batch, val_label_batch = input_data.get_batch(
                                                    val,
                                                    val_label,
                                                    FLAGS.height, 
                                                    FLAGS.width, 
                                                    FLAGS.batch_size, 
                                                    FLAGS.capacity)

    logits = model.make_network(train_batch, train_label_batch, keep_prob)
    loss = model.losses(logits, train_label_batch)
    train_op = model.trainning(loss, FLAGS.learning_rate)
    acc = model.evaluation(logits, train_label_batch)

    x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.height, FLAGS.width, 3])
    y_ = tf.placeholder(tf.int16, shape=[FLAGS.batch_size])


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

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(train_logs_dir, sess.graph)
        val_writer = tf.summary.FileWriter(val_dir, sess.graph)

        try:
            for step in np.arange(FLAGS.max_steps):
                if coord.should_stop():
                        break
                tra_images,tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x:tra_images, y_:tra_labels})
                if step % 10 == 0:
                    print('Step %d, train loss = %f, train accuracy = %f%%' %(step, tra_loss, tra_acc*100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)

                if step % 200 == 0 or (step + 1) == FLAGS.max_steps:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, acc],
                                                 feed_dict={x:val_images, y_:val_labels})
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc*100.0))
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str, step)

                if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(train_logs_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    training()








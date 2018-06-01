
import os
import numpy as np
import tensorflow as tf
import input_data
import model

# you need to change the directories to yours.
train_dir = "C://Users//Bean//Documents//Python Scripts//layerCNN//train/"
test_dir = "C://Users//Bean//Documents//Python Scripts//layerCNN//test/"
train_logs_dir = './logs3//train/'
val_logs_dir = './logs3//val'

N_CLASSES = 2
IMG_W = 160     # resize the image, if the input image is too large, training will be very slow.
IMG_H = 160
RATIO = 0.2     # take 20% of dataset as validation data
BATCH_SIZE = 32
CAPACITY = 800
MAX_STEP = 20000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0005 # with current parameters, it is suggested to use learning rate<0.0001


def training():

    train, train_label, val, val_label = input_data.get_files(train_dir, RATIO)
    train_batch, train_label_batch = input_data.get_batch(train,
                                                  train_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE,
                                                  CAPACITY)
    val_batch, val_label_batch = input_data.get_batch(val,
                                                  val_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE,
                                                  CAPACITY)

    logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    loss = model.losses(logits, train_label_batch)
    train_op = model.trainning(loss, learning_rate)
    acc = model.evaluation(logits, train_label_batch)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE])


    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        #ckpt = tf.train.get_checkpoint_state(train_logs_dir)
        #if ckpt and ckpt.model_checkpoint_path:
        #    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        #    saver.restore(sess, './logs3/train/LayerModel.ckpt-8000')
        #    #saver.restore(sess, ckpt.model_checkpoint_path)
        #    print('Loading success, global_step is %s' % global_step)
        #else:
        #    print('No checkpoint file found')
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(train_logs_dir, sess.graph)
        val_writer = tf.summary.FileWriter(val_logs_dir, sess.graph)

        try:
            for step in np.arange(1,MAX_STEP):
                if coord.should_stop():
                        break
                tra_images,tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x:tra_images, y_:tra_labels})

                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)

                if step % 10000 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, acc],
                                                 feed_dict={x:val_images, y_:val_labels})
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc*100.0))
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str, step)

                if step % 1000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(train_logs_dir, 'LayerModel.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
    writer = tf.summary.FileWriter('./logs3/train/', sess.graph)
    

    
if __name__ == '__main__':
    training()








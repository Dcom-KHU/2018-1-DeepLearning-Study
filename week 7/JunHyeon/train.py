import tensorflow as tf
import model
import input

FLAGS = tf.app.flags.FLAGS

def train():
    keep_prob = tf.placeholder(tf.float32)
    images, labels = input.get_data('train', FLAGS.batch_size)
    hypothesis, cross_entropy, train_step = model.make_network(images, labels, keep_prob)

    cost_sum = tf.summary.scalar("cost", cross_entropy)

    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)
        writer = tf.summary.FileWriter("./logs/cost_log")
        writer.add_graph(sess.graph)  # Show the graph

        merge_sum = tf.summary.merge_all()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        for step in range(FLAGS.max_steps):
            summary, _ = sess.run([merge_sum,train_step], feed_dict={keep_prob: 0.7})
            writer.add_summary(summary, global_step=step)
            print(step, sess.run(cross_entropy, feed_dict={keep_prob: 1.0}))

        coord.request_stop()
        coord.join(threads)
g
def main(argv = None):
    train()

if __name__ == '__main__':
    tf.app.run()

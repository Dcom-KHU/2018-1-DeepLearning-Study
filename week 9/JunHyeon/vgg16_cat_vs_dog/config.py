import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('height', 224, '')
tf.app.flags.DEFINE_integer('width', 224, '')
tf.app.flags.DEFINE_integer('depth', 3, '')
tf.app.flags.DEFINE_integer('num_class', 2, '')
tf.app.flags.DEFINE_integer('num_conv', 5, '')
tf.app.flags.DEFINE_integer('kernel_size', 5, '')
tf.app.flags.DEFINE_integer('pool_size', 2, '')
tf.app.flags.DEFINE_integer('num_map', 32, '')
tf.app.flags.DEFINE_integer('num_fc_layer', 4, '')
tf.app.flags.DEFINE_integer('num_fc_input', 768, '')
tf.app.flags.DEFINE_integer('max_steps', 15000, '')
tf.app.flags.DEFINE_integer('eval_step', 1, '')
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_integer('eval_batch_size', 32, '')
tf.app.flags.DEFINE_integer('capacity', 2000, '')
tf.app.flags.DEFINE_float('learning_rate', 1e-7, '')
tf.app.flags.DEFINE_float('ratio', 0.2, '')

tf.app.flags.DEFINE_string('train_dir', './Cat_Vs_Dog_Data/Data/train/', '')
tf.app.flags.DEFINE_string('eval_dir', './Cat_Vs_Dog_Data/Data/test/', '')
tf.app.flags.DEFINE_string('log_dir', './DeepLearning/vgg16/logs/', '')
tf.app.flags.DEFINE_string('eval_log_dir', './DeepLearning/vgg16/logs/eval/', '')
tf.app.flags.DEFINE_string('val_dir', './DeepLearning/vgg16/logs/val/', '')

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('height', 192, '')
tf.app.flags.DEFINE_integer('width', 192, '')
tf.app.flags.DEFINE_integer('depth', 3, '')
tf.app.flags.DEFINE_integer('num_class', 2, '')
tf.app.flags.DEFINE_integer('num_conv', 5, '')
tf.app.flags.DEFINE_integer('kernel_size', 5, '')
tf.app.flags.DEFINE_integer('pool_size', 2, '')
tf.app.flags.DEFINE_integer('num_map', 32, '')
tf.app.flags.DEFINE_integer('num_fc_layer', 4, '')
tf.app.flags.DEFINE_integer('num_fc_input', 768, '')
tf.app.flags.DEFINE_integer('max_steps', 20000, '')
tf.app.flags.DEFINE_integer('eval_steps', 1, '')
tf.app.flags.DEFINE_integer('batch_size', 128, '')
tf.app.flags.DEFINE_integer('capacity', 8000, '')
tf.app.flags.DEFINE_float('learning_rate', 0.00005, '')
tf.app.flags.DEFINE_float('ratio', 0.2, '')

tf.app.flags.DEFINE_string('train_dir', './Cat_Vs_Dog_Data/Data/train/', '')
tf.app.flags.DEFINE_string('eval_dir', './Cat_Vs_Dog_Data/Data/test/', '')
tf.app.flags.DEFINE_string('log_dir', './Cat_Vs_Dog_Data/logs/', '')
tf.app.flags.DEFINE_string('val_dir', './Cat_Vs_Dog_Data/logs/val/', '')

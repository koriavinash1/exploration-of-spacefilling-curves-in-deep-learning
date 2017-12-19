from tf_ops import *
import tensorflow as tf
from keras.datasets import cifar10

num_classes = 10

(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
# Use this for testing network
(train_features, train_labels), (test_features, test_labels) = (train_features[:200], train_labels[:200]), (test_features[:100], test_labels[:100])

train_labels = OneHot(train_labels, num_classes)
test_labels = OneHot(test_labels, num_classes)

inputph = tf.placeholder(shape=(None, 32*32,), name='mainInput', dtype=tf.float32)
targets = tf.placeholder(shape=(None, num_classes), name="segmentation_map", dtype=tf.float32)

class networkGraph(object):
	def __init__(self, inputs, targets):
		self._inputs = inputs
		self._targets = targets

	def denseBlock(self, depth, name, ):
		for i in xrange(layers):
			BN = BatchNorm(name='bn_'+name+str(i),
			            data=volumes_,
			            training=FLAGS.train_mode)
			conv = Conv3D(filters=32, 
			        name='convolution_'+name+str(i),
			        kernel_size=[3,3,3],
			        activation=None,
			        data=BN)
			volumes_ = Concatenate([volumes_, conv], 
			        name='concatenate_'+name+str(i))
		return volumes_

	
	def network(input):
		inputph = Input()		
		
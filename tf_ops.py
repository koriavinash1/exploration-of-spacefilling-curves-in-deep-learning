import tensorflow as tf
import numpy as np
import sys, os, shutil

def Weight_Bias(W_shape, b_shape,collection_name = 'no_name'):
	with tf.device('/cpu:0'):
		W = tf.get_variable(name = 'weights', shape = W_shape, 
				initializer = tf.truncated_normal_initializer(stddev=0.1/np.prod(W_shape),dtype = tf.float32))
		tf.add_to_collection('l2_norm_vars',W)
		b = tf.get_variable(name = 'biases', shape = b_shape, initializer = tf.constant_initializer(0.1))
		tf.add_to_collection('l2_norm_vars',b)

		tf.add_to_collection(collection_name,W)
		tf.add_to_collection(collection_name,b)
	return W, b

def Inputs(*args):
	return tf.placeholder(tf.float32,args,name = 'Inputs')
		
def Targets(*args):
	return tf.placeholder(tf.float32,args,name = 'Targets')

def OneHot(targets,num_class):
	return tf.one_hot(targets,num_class,1,0)

def Softmax(logits):
	return tf.nn.softmax(logits,name = 'softmax')

def Dropout(x, is_training, keep_prob = 0.8):
	keep_prob_pl = tf.cond(is_training, lambda : tf.constant(keep_prob), lambda : tf.constant(1.0))
	return tf.nn.dropout(x,keep_prob_pl)

def flexiSession():
	# Helps in softplacement of the device
	config = tf.ConfigProto()
	config.allow_soft_placement = True
	config.gpu_options.allow_growth = True
	return tf.Session(config = config)

def Conv1D(x, filter_shape,collection_name, stride = 1, padding = 'VALID'):
	# filter_shape: 3D tensor, [None, features, channels]
	# x: 3D tensor, [None, features, Channels]
	strides = None
	if isinstance(stride,int):
		strides = stride
	
	W_shape = filter_shape
	b_shape = [filter_shape[3]]
	W, b = __Weight_Bias(W_shape, b_shape,collection_name = collection_name)
	conv_out = tf.nn.conv1d(x, W, strides, padding)
	ret_val = conv_out + b
	return ret_val

def Elu(x):
	# elu activation helps in faster convergance
	return tf.nn.elu(x)


def MaxPool1D(x):
	# To maintain hilbert property always size should be reduced by 4
	ret_val = tf.contrib.keras.layers.MaxPool1D(x, pool_size=4, strides = 4, padding = 'VALID')
	return ret_val

def Adam(lr):
	return tf.train.AdamOptimizer(learning_rate = lr)

def BatchNorm(inputs, is_training, decay = 0.9, epsilon=1e-3):
	with tf.device('/cpu:0'):
		scale = tf.get_variable(name = 'scale', shape = inputs.get_shape()[-1],\
								initializer = tf.constant_initializer(1.0),dtype = tf.float32)
		tf.add_to_collection('l2_norm_vars', scale)
		beta = tf.get_variable(name = 'beta', shape = inputs.get_shape()[-1],\
								initializer = tf.constant_initializer(0.0),dtype = tf.float32)
		tf.add_to_collection('l2_norm_vars',beta)

		pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
		pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
		axis = list(range(len(inputs.get_shape())-1))

	def Train(inputs, pop_mean, pop_var, scale, beta):
		batch_mean, batch_var = tf.nn.moments(inputs,axis)
		train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
		train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))

		mean_distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pop_mean, batch_mean))))
		var_distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pop_var, batch_var))))

		with tf.control_dependencies([train_mean, train_var]):
			return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
	def Eval(inputs, pop_mean, pop_var, scale, beta):
		return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

	return tf.cond(is_training, lambda: Train(inputs, pop_mean, pop_var, scale, beta),\
									lambda: Eval(inputs, pop_mean, pop_var, scale, beta))


def BN_eLU_Conv(inputs, n_filters,collection_name, filter_size=3, keep_prob=0.8, is_training=tf.constant(False,dtype=tf.bool)):
	l = Elu(BatchNorm(inputs,is_training=is_training))
	l = Conv2D(l, [filter_size, filter_size, l.get_shape()[-1].value, n_filters],collection_name = collection_name, padding='SAME')
	l = Dropout(l, is_training=is_training,keep_prob=keep_prob)
	return l
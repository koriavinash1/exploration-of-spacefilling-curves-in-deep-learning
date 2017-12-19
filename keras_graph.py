import keras.backend as k
from hilbert_ops import image2signal
from keras.layers import Input, Flatten, Dense, BatchNormalization, Activation, Dropout, concatenate, Conv1D, MaxPool1D
from keras.layers.merge import Multiply
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras import regularizers
import numpy as np
import cv2, os
from tqdm import tqdm
from keras.utils import np_utils
from keras.datasets import cifar10
import pandas as pd
from tqdm import tqdm
from datetime import date
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()

# base_path = "/media/koriavinash/New Volume/Research/Datasets/CIFAR10/newCifar10/"
# train_features = np.array([cv2.imread(base_path +"train/"+ image) for image in tqdm(os.listdir(base_path+"train")) if image.endswith('.jpg')])
# train_lab= pd.read_csv(base_path+"train.csv")['Label'].as_matrix()
# train_labels = np.array([np.array([int(d[1])]) for d in train_lab])
# print train_features.shape, train_labels.shape

# (train_features, train_labels), (test_features, test_labels) = (train_features[:200], train_labels[:200]), (test_features, test_labels)
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("./dataset", one_hot=True)

image_width = 32
epochs = 300
learning_rate = 0.001
batch_size = 128
n_classes = 10
stages = 5
num_blocks = 4

nx, ny = (image_width, image_width)
xt = np.linspace(0, 1, nx) 
yt = np.linspace(0, 1, ny)
xpos, ypos = np.meshgrid(xt, yt)

# xpos = np.array(xpos).flatten()
# ypos = np.array(ypos).flatten()

def pre_processing(images):
	print "Processing images: \n"
	processed_images_32, processed_images_16, processed_images_8 = [], [], []
	for image32 in tqdm(images):
		# image16 = cv2.resize(image32, (16, 16), interpolation=cv2.INTER_AREA)
		# image8 = cv2.resize(image32, (8, 8), interpolation=cv2.INTER_AREA)
		# print(np.concatenate([[image32.T[0]], [image32.T[1]], [image32.T[2]], [xpos], [ypos]]).T.shape)
		image32 = np.concatenate([np.divide([image32.T[0]+image32.T[1]+image32.T[2]], 3), [xpos], [ypos]]).T
		himage32 = image2signal(image32)
		himage32 = np.diff(himage32.T).T
		# himage32 = np.divide(himage32, 255.0)

		# himage16 = image2signal(image16)
		# himage16 = np.divide(himage16, 255.0)

		# himage8 = image2signal(image8)
		# himage8 = np.divide(himage8, 255.0)
		
		# asd = np.swapaxes(np.swapaxes(np.array(np.concatenate([[image], [xpos], [ypos]])), 0, 1), 1, 2)
		# print himage.T[0].T.shape
		processed_images_32.append(himage32)
		# processed_images_16.append(himage16)
		# processed_images_8.append(himage8)
	return np.array(processed_images_32), np.array(processed_images_16), np.array(processed_images_8)

def denseBlock(datain):
	for i in range(stages):
		concat_layers = []
            	concat_layers.append(datain)
		data = datain if i == 0 else dense
		conv = Conv1D(32, 4, activation='elu', padding='SAME', 
			kernel_regularizer=regularizers.l2(5e-6),
                		activity_regularizer=regularizers.l1(5e-6))(data)
		conv = Dropout(0.25)(conv)
		for j in range(len(concat_layers)-1):
			data = conacat_layers[j]
			data = concatenate([data, concat_layers[i]])
		dense = BatchNormalization()(data)
		dense = concatenate([dense, conv])
	return dense

def graph():
	model_input_32 = Input(shape=(32*32-1, 3), name="orig_32")
	# model_input_16 = Input(shape=(16*16, 3), name="orig_16")
	# model_input_8 = Input(shape=(8*8, 3), name="orig_8")
	
	dense = Dense(1024, activation='elu')(model_input_32)
	dense = Dropout(0.3)(dense)
	
	dense = Dense(1024, activation='elu')(dense)
	dense = Dropout(0.3)(dense)

	dense = Dense(512, activation='elu')(dense)
	dense = Dropout(0.3)(dense)

	dense = Dense(128, activation='elu')(dense)
	dense = Dropout(0.3)(dense)

	# output = Dense(10, activation='softmax')(dense)

	#########
	#    orig_32  #
	#########
	# conv = Conv1D(32, 4, activation='elu', padding='SAME', 
	# 		kernel_regularizer=regularizers.l2(5e-6),
	# 		activity_regularizer=regularizers.l1(5e-6))(model_input_32)
	# conv = MaxPool1D(pool_size=4, strides=4, padding='valid')(conv)
	# # conv = BatchNormalization()(conv)
	# conv = Dropout(0.25)(conv)

	# conv = Conv1D(64, 4, activation='elu', padding='SAME', 
	# 		kernel_regularizer=regularizers.l2(5e-6),
	# 		activity_regularizer=regularizers.l1(5e-6))(conv)
	# conv = MaxPool1D(pool_size=4, strides=4, padding='valid')(conv)
	# # conv = BatchNormalization()(conv)
	# conv = Dropout(0.25)(conv)

	# conv = Conv1D(128, 4, activation='elu', padding='SAME', 
	# 		kernel_regularizer=regularizers.l2(5e-6),
	# 		activity_regularizer=regularizers.l1(5e-6))(conv)
	# conv = MaxPool1D(pool_size=4, strides=4, padding='valid')(conv)
	# conv = BatchNormalization()(conv)
	# conv = Dropout(0.25)(conv)

	#########
	#    orig_16  #
	#########
	# conv1 = Conv1D(64, 4, activation='elu', padding='SAME', 
	# 		kernel_regularizer=regularizers.l2(5e-6),
	# 		activity_regularizer=regularizers.l1(5e-6))(model_input_16)
	# conv1 = MaxPool1D(pool_size=4, strides=4, padding='valid')(conv1)
	# conv1 = BatchNormalization()(conv1)
	# conv1 = Dropout(0.25)(conv1)

	# conv1 = Conv1D(128, 4, activation='elu', padding='SAME', 
	# 		kernel_regularizer=regularizers.l2(5e-6),
	# 		activity_regularizer=regularizers.l1(5e-6))(conv1)
	# conv1 = MaxPool1D(pool_size=4, strides=4, padding='valid')(conv1)
	# conv1 = BatchNormalization()(conv1)
	# conv1 = Dropout(0.25)(conv1)

	# conv1 = Multiply()([conv1, conv])

	#########
	#    orig_8  #
	#########
	# conv2 = Conv1D(128, 4, activation='elu', padding='SAME', 
	# 		kernel_regularizer=regularizers.l2(5e-6),
	# 		activity_regularizer=regularizers.l1(5e-6))(model_input_8)
	# conv2 = MaxPool1D(pool_size=4, strides=4, padding='valid')(conv2)
	# conv2 = BatchNormalization()(conv2)
	# conv2 = Dropout(0.25)(conv2)
	# conv2 = concatenate([conv1, conv2])

	# conv = Conv1D(128, 4, activation='elu', padding='SAME', 
	# 		kernel_regularizer=regularizers.l2(5e-6),
	# 		activity_regularizer=regularizers.l1(5e-6))(conv2)
	# conv = MaxPool1D(pool_size=4, strides=4, padding='valid')(conv)
	# conv = BatchNormalization()(conv)
	# conv = Dropout(0.25)(conv)

	#########

	dense = Flatten()(dense)
	# dense3 = Dense(64, activation='elu')(dense)
	# dense3 = BatchNormalization()(dense3)
	output = Dense(10, activation='softmax')(dense)

	model = Model(input=model_input_32, output=output)
	model.summary()
	return model






model = graph()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# date = date.today()
checkpointer = ModelCheckpoint(filepath="./results/bestmodels/fn_model.{epoch:02d}-{val_acc:.2f}.hdf5", verbose=1, monitor='val_acc', save_best_only=True, save_weights_only=False, mode='max', period=1)
tf_board = TensorBoard(log_dir='./results/logs', histogram_freq=0, write_graph=True, write_images=True)
csv_logger = CSVLogger('./results/training.log')
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.01)


batch32, batch16, batch8 = pre_processing(train_features)
batch_y = np_utils.to_categorical(train_labels, n_classes)




model.fit(batch32, batch_y, batch_size=batch_size, nb_epoch=epochs, validation_split=0.08, callbacks=[checkpointer, tf_board, csv_logger, reduce_lr, early_stopping])
model.save("./results/final_hilbert.hdf5")






tbatch32, tbatch16, tbatch8 = pre_processing(test_features)
testLabels = np_utils.to_categorical(test_labels, n_classes)
print model.evaluate(	tbatch32, testLabels, batch_size=batch_size, verbose=1)

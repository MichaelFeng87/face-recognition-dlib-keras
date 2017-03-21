"""
This script is used to train the model to recognize person's ID based on it's facial feature
using Keras

The output of this script is stored in nn_model directory which will contain .hdf5, .json and .pkl files

The .hdf5 file is the keras weight file
The .json file is the keras architecture file
The .pkl file is the file containing dictionary information about person's label corresponding with neural network output data
"""
import dlib
import cv2
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.externals import joblib
from keras.models import Sequential, model_from_json
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD,RMSprop

# Enable continue_training to train existing model if set to 'True'
continue_training = False

# Enable to use pre-extracted negative face data from LFW dataset if set to 'True'
use_preprocessed_neg_data = True

# Ratio of Train and Test data
test_data_ratio=0.33


# The output directory of the trained neural net model
nn_model_dir='nn_model/'
# hdf5_filename = 'face_recog_weights.hdf5'
# json_filename = 'face_recog_arch.json'
# labeldict_filename = 'label_dict.pkl'

hdf5_filename = 'face_recog_special_weights.hdf5'
json_filename = 'face_recog_special_arch.json'
labeldict_filename = 'label_dict_special.pkl'

# DLIB's model path for face pose predictor and deep neural network model
predictor_path='dlib_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path='dlib_model/dlib_face_recognition_resnet_model_v1.dat'

# The input directory of positive and negative person's facial data
pos_image_dir='target_person/'
neg_image_dir='unknown_person/'


# Keras neural network model learning parameter
batch_size = 150
nb_epoch = 50
loss = 'categorical_crossentropy'
# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
optimizer=rmsprop


##
## @brief      Function to make new nn architecture
##
## @param      nb_class  The number of class (n authorized person + 1 unknown person)
##
## @return     keras model
##
def make_model(nb_class):
	print('Building CNN architecture model..')
	model = Sequential()

	model.add(Dense(4096,input_dim=128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_class))
	model.add(Activation('softmax'))

	
	model.compile(loss=loss,
				  optimizer=optimizer,
				  metrics=['accuracy'])
	
	print('Finished building CNN architecture..')
	return model

##
## @brief      Function to train new keras architecture
##
## @param      model        The model
## @param      train_data   The train data
## @param      train_label  The train label
## @param      test_data    The test data
## @param      test_label   The test label
## @param      nb_epoch     The number of epoch
##
## @return     keras model
##
def train_model(model,
				train_data,
				train_label,
				test_data,
				test_label,
				nb_epoch=100):
	
	checkpointer = ModelCheckpoint(filepath=nn_model_dir+hdf5_filename,
								   verbose=1,
								   save_best_only=True)

	cnn_json_model = model.to_json()

	with open(nn_model_dir+json_filename, "w") as json_file:
		json_file.write(cnn_json_model)
	
	print("Saved CNN architecture to disk..")
		
	print('Start optimizing CNN model..')
	model.fit(train_data,
			  train_label,
			  batch_size=batch_size,
			  nb_epoch=nb_epoch,
			  validation_data=(test_data, test_label),
			  callbacks=[checkpointer],
			  shuffle=True,
			  verbose=1)
	
	print('Optimization finished..')
	return model

x_train=[]
x_test=[]
y_train=[]
y_test=[]

authorized_person_list=os.listdir(pos_image_dir)
nb_class=len(authorized_person_list)+1


print 'Building neural network architecture...'
if not continue_training:
	cnn_model = make_model(nb_class)
else:
	json_model_file=open(nn_model_dir+json_filename, 'r')
	json_model = json_model_file.read()
	json_model_file.close()

	cnn_model = model_from_json(json_model)

	cnn_model.load_weights(nn_model_dir+hdf5_filename)

	cnn_model.compile(loss=loss,
				  optimizer=optimizer,
				  metrics=['accuracy'])

print 'Processing AUTHORIZED person data...'

label_dict=dict()
class_counter=0
for person in authorized_person_list:
	print 'Processing %s data.....'%(person)
	label_dict[class_counter]=person
	temp_data=joblib.load(pos_image_dir+person+'/face_descriptor.pkl')
	temp_label=np.repeat(class_counter,len(temp_data))

	temp_x_train, temp_x_test, temp_y_train, temp_y_test = train_test_split(
		temp_data, temp_label, test_size=test_data_ratio, random_state=42)
	print 'Obtained %i train and %i test data'%(len(temp_x_train),len(temp_x_test))
	if len(x_train)==0:
		x_train=temp_x_train
		x_test=temp_x_test
		y_train=np.append(y_train,temp_y_train)
		y_test=np.append(y_test,temp_y_test)
	else:
		x_train=np.append(x_train,temp_x_train,axis=0)
		x_test=np.append(x_test,temp_x_test,axis=0)
		y_train=np.append(y_train,temp_y_train)
		y_test=np.append(y_test,temp_y_test)
	class_counter+=1

print 'Finished...'


print 'Processing UNKNOWN person data'
label_dict[class_counter]='UNKNOWN'
joblib.dump(label_dict,nn_model_dir+labeldict_filename)
if not use_preprocessed_neg_data:
	stranger_list = os.listdir(neg_image_dir+'raw_data/')

	detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor(predictor_path)
	facerec = dlib.face_recognition_model_v1(face_rec_model_path)

	total_processed=0
	temp_data=[]
	temp_label=[]
	for names in stranger_list:
		percentage=(float(total_processed)/float(len(stranger_list)))*100
		print '%i%% finished'%(percentage)
		path=neg_image_dir+'raw_data/'+names+'/'
		image_list=os.listdir(path)
		for image in image_list:
			data=cv2.imread(path+image)
			dets,scores,idx = detector.run(data, 0,-0.5)
			for i, d in enumerate(dets):
				shape = sp(data, d)
				face_descriptor = np.array([facerec.compute_face_descriptor(data, shape,10)])
				if len(temp_data)==0:
					temp_data=face_descriptor
				else:
					temp_data=np.append(temp_data,face_descriptor,axis=0)
				temp_label.append(class_counter)
		total_processed+=1

	neg_data=temp_data
	joblib.dump(neg_data,neg_image_dir+'preprocessed_data/face_recog_neg_data_gray.pkl')
else:
	neg_data=joblib.load(neg_image_dir+'preprocessed_data/face_recog_neg_data_gray.pkl')
	temp_data=neg_data

temp_label=np.repeat(class_counter,len(temp_data))

temp_x_train, temp_x_test, temp_y_train, temp_y_test = train_test_split(
	temp_data, temp_label, test_size=test_data_ratio, random_state=42)
	
x_train=np.append(x_train,temp_x_train,axis=0)
x_test=np.append(x_test,temp_x_test,axis=0)
y_train=np.append(y_train,temp_y_train)
y_test=np.append(y_test,temp_y_test)

print 'Finished extracting data.....'

y_train_cat = y_train.astype('int')
y_train_cat = np_utils.to_categorical(y_train_cat, nb_class)

y_test_cat = y_test.astype('int')
y_test_cat = np_utils.to_categorical(y_test_cat, nb_class)

trained_cnn_model = train_model(cnn_model,
								x_train,
								y_train_cat,
								x_test,
								y_test_cat,
								nb_epoch)




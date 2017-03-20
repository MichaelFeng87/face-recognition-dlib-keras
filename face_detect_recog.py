"""
This script is used to demonstrate face recognition using webcam
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
from keras.optimizers import SGD
import time

# Select between 'video' and 'image'. 
input_data='video'

# If input_data set to 'image', define the image file path
image_file='/home/alvin/Downloads/IMG_6529.JPG'


# The directory of the trained neural net model
nn_model_dir='nn_model/'

# DLIB's model path for face pose predictor and deep neural network model
predictor_path='dlib_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path='dlib_model/dlib_face_recognition_resnet_model_v1.dat'

# .pkl file containing dictionary information about person's label corresponding with neural network output data
label_dict=joblib.load(nn_model_dir+'label_dict.pkl')

# Keras neural network model learning parameter
batch_size = 10
nb_epoch = 200
loss = 'categorical_crossentropy'
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


json_model_file=open(nn_model_dir+'face_recog_arch.json', 'r')
json_model = json_model_file.read()
json_model_file.close()

cnn_model = model_from_json(json_model)
cnn_model.load_weights(nn_model_dir+'face_recog_weights.hdf5')

cnn_model.compile(loss=loss,
	optimizer=sgd,
	metrics=['accuracy'])


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


if input_data == 'video':
	cap = cv2.VideoCapture(0)

	while (True):
		ret, frame = cap.read()
		start=time.time()
		# frame=cv2.resize(frame,(90,60))
		# frame=cv2.resize(frame,(320,240))

		dets,scores,idx = detector.run(frame, 0,-0.5)

		for i, d in enumerate(dets):

			# if idx[i]==0:

			if idx[i]==0 or idx[i]==1 or idx[i]==2 or idx[i]==3 or idx[i]==4:

				cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),2)
				shape = sp(frame, d)
				face_descriptor = np.array([facerec.compute_face_descriptor(frame, shape)])
				prediction = cnn_model.predict_proba(face_descriptor)

				highest_proba=0
				counter=0
				# print prediction
				for prob in prediction[0]:
					if prob > highest_proba and prob >=0.3:
						highest_proba=prob
						label=counter
						label_prob=prob
						identity = label_dict[label]
					if counter==(len(label_dict)-1) and highest_proba==0:
						label= label_dict[counter]
						label_prob=prob
						identity=label
					counter+=1

				# min_dist=99999
				# for key in face_dict:

				# 	dist = euclidean_distances(face_dict[key],face_descriptor)
				# 	# dist = np.linalg.norm(face_dict[key]-face_descriptor)
				# 	if dist <0.5:
				# 		if dist < min_dist:
				# 			min_dist=dist
				# 			identity=key
				# 	print key,dist


				cv2.putText(frame,identity+'='+str(round((label_prob*100),2))+'%',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

			# print scores[i]
			# print idx[i]

		# print x1,y1,x2,y2

		cv2.namedWindow('face detect', flags=cv2.WINDOW_NORMAL)
		cv2.imshow('face detect',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		delta=time.time()-start
		fps=float(1)/float(delta)
		print(fps)

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

else:
	image=cv2.imread(image_file)

	dets,scores,idx = detector.run(image, 0,1)

	for i, d in enumerate(dets):

		if idx[i]==0 or idx[i]==1 or idx[i]==2 or idx[i]==3 or idx[i]==4:

			cv2.rectangle(image,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),10)
			shape = sp(image, d)
			face_descriptor = np.array([facerec.compute_face_descriptor(image, shape)])
			prediction = cnn_model.predict_proba(face_descriptor)

			highest_proba=0
			counter=0
			# print prediction
			for prob in prediction[0]:
				if prob > highest_proba and prob >=0.3:
					highest_proba=prob
					label=counter
					label_prob=prob
					identity = label_dict[label]
				if counter==(len(label_dict)-1) and highest_proba==0:
					label= label_dict[counter]
					label_prob=prob
					identity=label
				counter+=1

			cv2.putText(image,identity+'='+str(round((label_prob*100),2))+'%',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),3)

	cv2.namedWindow('preview', flags=cv2.WINDOW_NORMAL)
	cv2.imshow('preview',image)
	cv2.imwrite('/home/alvin/Downloads/test.jpg',image)
	cv2.waitKey(0)

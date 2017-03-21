"""
This script is used to demonstrate face recognition using webcam
"""
from keras.models import Sequential, model_from_json
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD
import dlib
import cv2
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.externals import joblib
import time

# ================================PARAMETER============================================
# Select between 'cam','video' and 'image'. 
input_data='video'

# If input_data set to 'image', define the image file path
image_file='/home/alvin/Downloads/IMG_6529.JPG'

# If input_data set to 'video', define the video file path
video_file='/home/alvin/Downloads/jokowi_mbek.mp4'
# video_file='/home/alvin/Downloads/wiranto.mp4'

# Set tolerance for face detection smaller means more tolerance for example -0.5 compared with 0
tolerance=-0.1

# gather data to reinforced model !!! CAN BE ACTIVATED WHILE SCRIPT ACTIVE BY PRESSING 'R'
is_gathering_data=False
person_name_list=['wiranto']
target_dir='target_person/'

# The directory of the trained neural net model
nn_model_dir='nn_model/'

hdf5_filename = 'face_recog_special_weights.hdf5'
json_filename = 'face_recog_special_arch.json'
labeldict_filename = 'label_dict_special.pkl'

# DLIB's model path for face pose predictor and deep neural network model
predictor_path='dlib_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path='dlib_model/dlib_face_recognition_resnet_model_v1.dat'

# .pkl file containing dictionary information about person's label corresponding with neural network output data
label_dict=joblib.load(nn_model_dir+labeldict_filename)

# Keras neural network model learning parameter
batch_size = 10
nb_epoch = 200
loss = 'categorical_crossentropy'
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# ====================================================================================



temp_data=dict()

for person in person_name_list:
	temp_data[person]=dict()
	temp_data[person]['data']=joblib.load(target_dir+person+'/face_descriptor.pkl')
	temp_data[person]['count']=0


json_model_file=open(nn_model_dir+json_filename, 'r')
json_model = json_model_file.read()
json_model_file.close()

cnn_model = model_from_json(json_model)
cnn_model.load_weights(nn_model_dir+hdf5_filename)

cnn_model.compile(loss=loss,
	optimizer=sgd,
	metrics=['accuracy'])


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


if input_data == 'cam':
	cap = cv2.VideoCapture(0)

	while (True):
		ret, frame = cap.read()
		start=time.time()

		dets,scores,idx = detector.run(frame, 0,tolerance)

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

						if identity in person_name_list and is_gathering_data:
							print identity
							temp_data[identity]['data']=np.append(temp_data[identity]['data'],face_descriptor,axis=0)
							temp_data[identity]['count']+=1
							if temp_data[identity]['count']==5:
								joblib.dump(temp_data[identity]['data'],target_dir+identity+'/face_descriptor.pkl')
								temp_data[identity]['count']=0
							cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,0),2)
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

				if label!='UNKNOWN':
					cv2.putText(frame,identity+'='+str(round((label_prob*100),2))+'%',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
				else:
					cv2.putText(frame,'???',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

			# print scores[i]
			# print idx[i]

		# print x1,y1,x2,y2

		cv2.namedWindow('face detect', flags=cv2.WINDOW_NORMAL)
		cv2.imshow('face detect',frame)
		if cv2.waitKey(1) & 0xFF == ord('r'):
			if is_gathering_data==True:
				is_gathering_data=False
			else:
				is_gathering_data=True
			
		delta=time.time()-start
		fps=float(1)/float(delta)
		print(fps)

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

elif input_data == 'image':
	image=cv2.imread(image_file)

	dets,scores,idx = detector.run(image, 0,tolerance)

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

			if label!='UNKNOWN':
				cv2.putText(frame,identity+'='+str(round((label_prob*100),2))+'%',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
			else:
				cv2.putText(frame,'???',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

	cv2.namedWindow('preview', flags=cv2.WINDOW_NORMAL)
	cv2.imshow('preview',image)
	cv2.imwrite('/home/alvin/Downloads/test.jpg',image)
	cv2.waitKey(0)

elif input_data == 'video':
	cap = cv2.VideoCapture(video_file)

	while (True):
		ret, frame = cap.read()
		start=time.time()

		dets,scores,idx = detector.run(frame, 0,tolerance)

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

						if identity in person_name_list and is_gathering_data:
							
							temp_data[identity]['data']=np.append(temp_data[identity]['data'],face_descriptor,axis=0)
							temp_data[identity]['count']+=1
							if temp_data[identity]['count']==5:
								joblib.dump(temp_data[identity]['data'],target_dir+identity+'/face_descriptor.pkl')
								temp_data[identity]['count']=0
							cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,0),2)

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


				if identity!='UNKNOWN':
					cv2.putText(frame,identity+'='+str(round((label_prob*100),2))+'%',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
				else:
					cv2.putText(frame,'???',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

			# print scores[i]
			# print idx[i]

		# print x1,y1,x2,y2

		cv2.namedWindow('face detect', flags=cv2.WINDOW_NORMAL)
		cv2.imshow('face detect',frame)
		if cv2.waitKey(1) & 0xFF == ord('r'):
			if is_gathering_data==True:
				is_gathering_data=False
			else:
				is_gathering_data=True
			
		delta=time.time()-start
		fps=float(1)/float(delta)
		print(fps)

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
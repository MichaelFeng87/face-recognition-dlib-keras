"""
This script is used to record person's facial feature data using dlib face shape model
and deep neural network model to extract face feature in 128-D vector
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
import shutil

# Set the output directory of the user's data
target_dir = 'authorized_person/'

if not os.path.exists(target_dir):
	os.makedirs(target_dir)

# Set the maximum user's data to be stored
max_user=10

# DLIB's model path for face pose predictor and deep neural network model
predictor_path='dlib_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path='dlib_model/dlib_face_recognition_resnet_model_v1.dat'



detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)\

cap = cv2.VideoCapture(0)

is_recording=False

temp_data=[]

while (True):

	name = str(raw_input("What's your name? "))
	directory=target_dir+name+'/'

	if not os.path.exists(directory):
		dir_list=os.listdir(target_dir)
		
		if len(dir_list)>=max_user:
			print "Warning! Too many registered user (%i users limitation exceeded)! Choose user's data to delete: "%(max_user)
			for user_count in range(0,len(dir_list)):
				print '%i : %s'%(user_count+1,dir_list[user_count])
			
			try:
				user_to_delete=int(raw_input('Choose user (number) to delete: '))
				chosen_user_dir=dir_list[user_to_delete-1]
				shutil.rmtree(target_dir+chosen_user_dir)
			except:
				print "WARNING! Wrong input! Try again!"
				continue
		os.makedirs(directory)
		break
	else:
		print 'Name already exist! Try again!'

while (True):
	ret, frame = cap.read()
	vis=frame.copy()
	start=time.time()
	is_eligible_frame=False

	dets,scores,idx = detector.run(frame, 0,0)

	for i, d in enumerate(dets):
			
		if len(idx)==1 and is_recording:
			shape = sp(frame, d)
			face_descriptor = np.array([facerec.compute_face_descriptor(frame, shape)])
			if len(temp_data)==0:
				temp_data=face_descriptor
			else:
				temp_data=np.append(temp_data,face_descriptor,axis=0)
			is_eligible_frame=True
			color=(0,255,0)
		elif len(idx)!=1 and is_recording:
			color=(0,0,255)
		else:
			color=(255,0,0)
		cv2.rectangle(vis,(d.left(),d.top()),(d.right(),d.bottom()),color,2)

	if is_eligible_frame:
		scaled_frame=cv2.resize(frame,(int(frame.shape[1]/3),int(frame.shape[0]/3)))

		dets,scores,idx = detector.run(scaled_frame, 0,-0.5)

		for i, d in enumerate(dets):
			if len(idx)==1:
				shape = sp(scaled_frame, d)
				face_descriptor = np.array([facerec.compute_face_descriptor(scaled_frame, shape)])
				temp_data=np.append(temp_data,face_descriptor,axis=0)




	cv2.imshow('face detect',vis)
	if cv2.waitKey(1) & 0xFF == ord('r'):
		if is_recording==True:
			is_recording=False
			print temp_data
			print len(temp_data)
			# Save the user's training data output to .pkl 
			joblib.dump(temp_data,directory+'/face_descriptor.pkl')
			break
		else:
			is_recording=True
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	delta=time.time()-start
	fps=float(1)/float(delta)
	print(fps)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


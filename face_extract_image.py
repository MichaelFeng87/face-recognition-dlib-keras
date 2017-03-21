"""
This script is used to extract person's facial feature data from image using dlib face shape model and deep neural network model to extract face feature in 128-D vector
"""

import dlib
import cv2
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.externals import joblib
import time
import shutil

# Set the input directory of the user's data
input_dir = '/home/alvin/Downloads/wiranto/'

sub_dirs = os.listdir(input_dir)

# Set the output directory of the user's data
target_dir = 'target_person/wiranto/'

# Enable preview for each train data
enable_preview = False



if not os.path.exists(target_dir):
	os.makedirs(target_dir)

# DLIB's model path for face pose predictor and deep neural network model
predictor_path='dlib_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path='dlib_model/dlib_face_recognition_resnet_model_v1.dat'


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

temp_data=[]

for directory in sub_dirs:
	image_list = os.listdir(input_dir+directory)
	for image in image_list:
		image =  cv2.imread(input_dir+directory+'/'+image)

		dets,scores,idx = detector.run(image, 0,0)

		for i, d in enumerate(dets):
				
			if len(idx)==1:
				shape = sp(image, d)
				face_descriptor = np.array([facerec.compute_face_descriptor(image, shape)])
				if len(temp_data)==0:
					temp_data=face_descriptor
				else:
					temp_data=np.append(temp_data,face_descriptor,axis=0)
				is_eligible_frame=True

			cv2.rectangle(image,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),2)

		if is_eligible_frame:
			scaled_image=cv2.resize(image,(int(image.shape[1]/2),int(image.shape[0]/2)))

			dets,scores,idx = detector.run(scaled_image, 0,-0.5)

			for i, d in enumerate(dets):
				if len(idx)==1:
					shape = sp(scaled_image, d)
					face_descriptor = np.array([facerec.compute_face_descriptor(scaled_image, shape)])
					temp_data=np.append(temp_data,face_descriptor,axis=0)

		if enable_preview:
			cv2.namedWindow('preview', flags=cv2.WINDOW_NORMAL)
			cv2.imshow('preview',image)
			cv2.waitKey(0)

print 'Obtained %i data'%(len(temp_data))
joblib.dump(temp_data,target_dir+'/face_descriptor.pkl')






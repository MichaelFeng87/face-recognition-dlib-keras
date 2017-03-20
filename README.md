# face-recognition-dlib-keras

Face recognition script using the combination of dlib and keras library to detect authorized person and unknown person using webcam

# Requirements
Reminder : You need CUDA-supported GPU to run this, otherwise enjoy the very low FPS~

1. dlib
2. keras
3. tensorflow-gpu
4. h5py

How to use it :

1. Download dlib model and put in dlib_model/ directory:
- http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
- http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

2. Create authorized_person/ directory and unknown_person/ directory
3. Download preprocessed unknown_person data from this link 
- https://app.box.com/s/0r365azbdz0swmx3cx15jkypduvwq79g
  
4. Extract the face_recog_neg_data_gray.pkl and put it in unknown_person/preprocessed_data/ directory
5. Run the face_record.py from the command line, input the name of the person, press 'r' to begin capturing frame contained the detected face. This script will capture face data if there is only 1 face detected in the webcam which is indicated by the green box, otherwise it will give red box which indicated that the script did not save the face data

6. Repeat the step in number 5 for each person you want to recognize. The maximum number person data that can be stored can be edited in the face_record.py

7. Train the neural network model by running the face_train_model.py

8. Run the face_detect_record.py

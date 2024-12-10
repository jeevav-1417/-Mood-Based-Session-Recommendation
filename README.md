Emotion-Based Yoga Recommender System
Description
This system detects emotions from a user’s face and hands using Mediapipe and classifies the emotions using a deep learning model. Based on the detected emotion, it recommends personalized yoga sessions through a Streamlit interface.

Features
Emotion detection using facial and hand landmarks.
Deep learning model to classify emotions.
Personalized yoga recommendations based on emotion and user preferences.
Real-time webcam integration using Streamlit.
Requirements
Python 3.x
TensorFlow for the model
Mediapipe for emotion detection
Streamlit for the web interface
OpenCV for image and video processing
NumPy for data handling

Documentation for Your Project: Emotion-Based Yoga Recommender System
1. Project Overview
The Emotion-Based Yoga Recommender system is designed to recommend personalized yoga sessions based on a user's emotional state. The system uses computer vision and machine learning techniques to recognize emotions through facial expressions and hand gestures, and then provides customized yoga recommendations based on these inputs.

2. Approach
2.1 Data Collection
The data collection phase involves capturing real-time data from the webcam using mediapipe to extract landmarks related to the face and hands. This data is preprocessed and stored in a NumPy file for later use. The key components of the data collected include:
Face Landmarks: These capture the position of various facial points such as eyes, nose, and mouth.
Left and Right Hand Landmarks: These capture hand positions to help detect gestures related to emotions.

2.2 Model Training
After collecting the data, we proceed to train a machine learning model to recognize different emotions based on the facial and hand gestures. The training process includes:
Data Preprocessing: This involves encoding the labels and shuffling the dataset.
Model Architecture: A simple neural network is built using Keras with two hidden layers.
Training: The model is trained using categorical cross-entropy loss and RMSProp optimizer.

2.3 Inference
For emotion recognition, the trained model is used to predict emotions from webcam input. The emotion is extracted using mediapipe, and the result is passed to the trained model to predict the corresponding emotion.

2.4 Streamlit Web Application
To make the application interactive, a Streamlit interface is used. It captures video input, processes the emotion, and recommends yoga sessions based on the recognized emotion.


Documentation for Your Project: Emotion-Based Yoga Recommender System
1. Project Overview
The Emotion-Based Yoga Recommender system is designed to recommend personalized yoga sessions based on a user's emotional state. The system uses computer vision and machine learning techniques to recognize emotions through facial expressions and hand gestures, and then provides customized yoga recommendations based on these inputs.

2. Approach
2.1 Data Collection
The data collection phase involves capturing real-time data from the webcam using mediapipe to extract landmarks related to the face and hands. This data is preprocessed and stored in a NumPy file for later use. The key components of the data collected include:
Face Landmarks: These capture the position of various facial points such as eyes, nose, and mouth.
Left and Right Hand Landmarks: These capture hand positions to help detect gestures related to emotions.

2.2 Model Training
After collecting the data, we proceed to train a machine learning model to recognize different emotions based on the facial and hand gestures. The training process includes:
Data Preprocessing: This involves encoding the labels and shuffling the dataset.
Model Architecture: A simple neural network is built using Keras with two hidden layers.
Training: The model is trained using categorical cross-entropy loss and RMSProp optimizer.

2.3 Inference
For emotion recognition, the trained model is used to predict emotions from webcam input. The emotion is extracted using mediapipe, and the result is passed to the trained model to predict the corresponding emotion.

2.4 Streamlit Web Application
To make the application interactive, a Streamlit interface is used. It captures video input, processes the emotion, and recommends yoga sessions based on the recognized emotion.
       
3. Results
Model Accuracy: The model was trained for 50 epochs and achieved an accuracy of around 80-90% based on the facial and hand landmarks.
Confusion Matrix: A confusion matrix was generated to evaluate the model's performance across different classes.
Visualization: A real-time webcam stream was used to predict emotions, and yoga sessions were recommended accordingly.

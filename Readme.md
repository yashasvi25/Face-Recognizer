<h1>Face-Recognizer</h1>
A project for recognition of faces among several faces in real-time.

<h2>Requirements</h2>

* Python 3.7
* Numpy 1.16 You can install it using pip :

```
$ pip3 install numpy
```
* OpenCV - contrib - python 4.1.0 You can install it using pip :

```
$ pip3 install opencv-python
```
<h2>Usage</h2>

* Download face_data_collect.py. Prepare a set of images of the known people you want to recognize by running the downloaded file. To run the file in command line use the following command :
```
$ python face_data_collect.py 
```
* Organize these images in a single directory(specify the path where you want to store images). Also while running the program you will be prompted to give the name of the person of whom images are being stored.

* Now download face_recognition.py and run the program file. To run the file in command line use the following command :
```
$ python face_recognition.py
```
You can see the predicted labeling to all the faces in live video stream captured from your webcam. Press 'q' key to exit the program.

<h2>How it works</h2>
<h4>Face_data_collection :</h4>

Here I used built-in Haar-cascade classifier in opencv to detect and extract faces in an image and store these extracted images in a particular location as numpy linear arrays.
Images are extracted from the live stream extracted through the webcam using opencv.

<h4>Face_Recognition :</h4>

Here again I used built-in Haar-cascade classifier in opencv to detect and extract faces in an image(frame of the live stream).
Then I used KNN (K-nearest-neighbours) algorithm to classify or label the face as a face of a particular person.
Algorithm Description:
The KNN classifier is first trained on a set of labeled (known) faces and can then predict the person in an unknown image by finding the K most similar faces (images with closet face-features under eucledian distance) in its training set, and performing a majority vote on their label. For example , if k = 10 , and the ten closest face images to the given image in the training set are three image of A and seven images of B , The result would be 'B'.

# Recognize faces using KNN algorithm
# Steps
# 1. Load the training data (Numpy arrays of all persons)
#      x - values are stored in the numpy arrays
#      y = assign values we need to assign for each person
# 2. Read a video stream using opencv
# 3. Extract faces out of it
# 4. Use KNN to find prediction of face(integer)
# 5. Map the predictor id to name of the user
# 6. Display the predictions(Name) on the screen - Bounding box and name

# Imports
import numpy as np
import cv2
import os


####################
# KNN algorithm that return prediction
# using euclidean distance
def dist(queryX, X):
    return np.sqrt(np.sum((queryX - X) ** 2))


# KNN
# queryX is the flattened np array of the frame we want to predict
# X if the training data which contains flattened images of users
# Y is the training data which maps all the images of users with a certain integer label
def KNN(X, Y, queryX, K=5):
    vals = []
    m = X.shape[0]
    # Calculating distance and storing it in val array
    for i in range(m):
        d = dist(queryX, X[i])
        vals.append((d, Y[i, 0]))
    # Sorting val to fink K - nearest neighbours
    vals = sorted(vals)
    # converting list into numpy array and slicing K elements
    vals = np.array(vals)
    vals = vals[: K]
    # Majority Vote
    new_vals = np.unique(vals[:, 1], return_counts=True)
    index = new_vals[1].argmax()
    # Prediction
    pred = new_vals[0][index]
    return pred


####################

# Reading training data
# face_data has the data for all the faces of every persons
# labels has the corresponding label for each image data
face_data = []
labels = []
# class_id tells the label value for a particular person file
dataset_path = "Data/"
class_id = 0
# names is a dictionary used to create a mapping between id and name
names = {}
# iterating over files in a directory with given path
for fx in os.listdir(dataset_path):
    # checking for a npy file
    if fx.endswith('.npy'):
        # mapping class_id with file name by slicing file name by 4 characters
        names[class_id] = fx[: -4]
        # Loading the file
        print("File loaded : " + dataset_path + fx)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)
        # create labels for current file
        target_label = class_id * np.ones((data_item.shape[0]))
        class_id += 1
        labels.append(target_label)
# concatenating all dataset
face_data_set = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape(-1, 1)

####################

# Testing and reading video stream
# Reading Video stream and extracting faces out of it
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
# init camera(Default web cam)
cap = cv2.VideoCapture(0)
# read video stream
while True:
    # Reading frame
    ret, frame = cap.read()
    # If frame not captured properly try again
    if not ret:
        continue
    # Getting list of coordinates(x , y , width , height) for faces in
    # the frame using detectMultiScale() method of face_cascade object
    # 1.3 here is scaleFactor (30% shrink)
    # 5 is minNeighbours
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    # If face not detected properly (Does not consider that frame)
    if len(faces) == 0:
        continue
    # Iterating on faces to draw rectangles over them and extract image
    for face in faces:
        # parameters of faces (coordinates , height and width)
        x, y, w, h = face
        # Extract Region of Interest
        # Giving some padding to image using offset
        # by convention in frame[y coordinates , x coordinates]
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        # CALLING PREDICTION
        predicted_label = KNN(face_data_set, face_labels, face_section.flatten())
        # To display the name and a rectangle around the face
        pred_name = names[int(predicted_label)]
        cv2.putText(frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Displaying the frame
    cv2.imshow('Frame', frame)

    # To stop the video capture (By pressing key 'q')
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
# Releasing the captured device and destroying all windows
cap.release()
cv2.destroyAllWindows()

####################
#  END OF PROGRAM  #

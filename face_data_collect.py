# Write a python script that captures images from web cam video stream
# Extracts all Faces from image frame(using haarcascades)
# Store the Face information into numpy arrays

# Steps
# 1. Read and show video stream , capture images
# 2. Detect faces and show bounding box (haarcascade)
# 3. Flatten the largest face image and save in a numpy array
#    Image should be grayscale if we want to save memory
# 4. Repeat the above for multiple people to generate training data

# Imports
import numpy as np
import cv2

# inputting the file name
filepath = "Data/"
filename = input("Enter the name of person : ")

# To keep count of which frame to store
skip = 0
# Face data to store
face_data = []

# face detection classifier using pre-built haarcascade model
# creating an object face_cascade
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

    # Now we need the largest face and store it
    # Sorting the faces array (acc. to area f[2])
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    # Iterating on faces to draw rectangles over them and extract image
    for face in faces:
        x, y, w, h = face
        # To put a rectangle around the face of certain color ans size(2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract 10th frame (Crop out required face) : Region of Interest
        # Giving some padding to image using offset
        # by convention in frame[y coordinates , x coordinates]
        if skip % 10 == 0:
            offset = 10
            face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            face_section = cv2.resize(face_section , (100 , 100))
            face_data.append(face_section)
            cv2.imshow('Face Section', face_section)

        skip += 1

    # Displaying the frame
    cv2.imshow('Frame', frame)

    # To stop the video capture (By pressing key 'q')
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
# Releasing the captured device and destroying all windows
cap.release()
cv2.destroyAllWindows()

# Convert face_data into a flattened numpy array and saving it to file location
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))

np.save(filepath+filename+'.npy', face_data)
print('Data successfully saved for person : ' + filename)

"""
    faces = face_cascade.detectMultiScale(frame, 1.1, 5)
scaleFactor - Parameter specifying how much the image size is reduced at each image scale.
We need scaling to shrink our image to the size on which the haarcascades algo was trained
minNeighbours - Parameter specifying how many neighbours each candidate rectangle should
have to retain it(based on surrounding neighbours). It affects the quality of detected faces.
Lower value results in poor detection or false detection.
"""

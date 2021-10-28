import cv2
import time
from detection_model import RetinaFace
import pickle
from retinaface.commons.preprocess import preprocess_image
from sklearn.metrics.pairwise import cosine_distances
from utils import functions
from extended_models import Gender, Age
import numpy as np
import os
import tensorflow as tf
from retinaface.commons import postprocess
from recognition_models import ArcFace

start = time.process_time()

def _load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def _most_similarity(embed_vecs, vec, labels):
#     sim = cosine_similarity(embed_vecs, vec)
    sim = cosine_distances(embed_vecs, vec)
    sim = np.squeeze(sim, axis = 1)
#     argmax = np.argsort(sim)[::-1][:1]
    argmax = np.argsort(sim)[::][:1]
    label = [labels[idx] for idx in argmax][0]
    return label

def predict_gender(model, image_link):
    # Resize image to model's input
    # processed_img = functions.preprocess_face(image_link, target_size = (224,224), detector_backend = "skip")
    prediction = model.predict(processed_img)
    if np.argmax(prediction) == 0:
        gender = "Female"
    elif np.argmax(prediction) == 1:
        gender = "Male"
    return gender

def predict_age(model, image_link):
    # processed_img = functions.preprocess_face(image_link, target_size = (224,224), detector_backend = "skip")
    prediction = model.predict(processed_img)[0,:]
    apparent_age = int(Age.findApparentAge(prediction))
    return apparent_age

# Create embedding model
model = ArcFace.loadModel()
# Create gender model
gender_model = Gender.loadModel()
age_model = Age.loadModel()
cap = cv2.VideoCapture("./sample/Ali_HoangDuong(2).mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./sample/demo2.avi', fourcc, 20.0, size)

# pTime=0
embed_faces = _load_pickle("./embed_vecs.pkl")
embed_faces = np.squeeze(np.stack(embed_faces), axis=1)
labels = _load_pickle("./labels.pkl")
detector_model = RetinaFace.build_model()

while True:
    success, img = cap.read()
    faces= {}
    faces = RetinaFace.detect_faces(img, threshold=0.9, model=detector_model)
    if (type(faces)==dict):
        for face in faces:
            facial_area = faces[face]['facial_area']
            landmarks = faces[face]["landmarks"]
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            nose = landmarks["nose"]
            # Draw bbox
            cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)
            # img = cv2.imread(img)
            facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
            # Perform alignment
            facial_img = postprocess.alignment_procedure(facial_img, right_eye, left_eye, nose)
            # Resize image
            facial_identification = functions.preprocess_face(facial_img, target_size = (112,112), detector_backend = "skip", align = False)
            # Get embedding of face
            vec = model.predict(facial_identification)
            vec = vec.reshape(1,-1)
            # Get label
            id_pred = _most_similarity(embed_faces, vec, labels)
            

            # Gender classification
            processed_img = functions.preprocess_face(facial_img, target_size = (224,224), detector_backend = "skip")
            gender_pred = predict_gender(gender_model, processed_img)
            age_pred = predict_age(age_model, processed_img)
            pred = id_pred + ", " + gender_pred + ", " + str(age_pred)
            # cv2.putText(img, pred, (facial_area[0], facial_area[1]), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 1)
            cv2.putText(img, pred, (0, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

    # cTime = time.time()
    # fps = 1/(cTime-pTime)
    # pTime = cTime
    # cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
    out.write(img)
    cv2.imshow("Image",img)
    if cv2.waitKey(20) & 0xFF ==ord('q'):
        break
print(time.process_time() - start)

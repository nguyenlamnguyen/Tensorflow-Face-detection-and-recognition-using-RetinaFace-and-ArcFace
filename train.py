import cv2
import os
import numpy as np
from imutils import paths
import pickle
from utils import functions
from recognition_models import ArcFace

model = ArcFace.loadModel()
# Generate embedding from face images
DATA_PATH = "./data/processed/"

def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def preprocessing(data_dir):
    image_links = list(paths.list_images(data_dir))
    faces=[]
    labels=[]
    embed_vecs=[]
    total = 0
    for image_link in image_links:
        split_img_links = image_link.split("\\")
        # Lấy nhãn của ảnh
        split_img_link = image_link.split("\\")
        label = split_img_link[0].split("/")[-1]
        print(label)
        # Đọc ảnh
        image = functions.preprocess_face(image_link, target_size = (112,112), detector_backend = "skip", align = False)
        image = functions.normalize_input(image, normalization="ArcFace")
        faces.append(image)
        labels.append(label)
        total+=1
    print("Total face preprocessed: {}".format(total))
    return faces, labels

def embedding_faces(model, faces):
    emb_vecs = []
    for face in faces:
        vec = model.predict(face)
        emb_vecs.append(vec)
    return emb_vecs

faces, labels = preprocessing(DATA_PATH)
embed_vec = embedding_faces(model, faces)

save_pickle(embed_vec,"./embed_vecs.pkl")
save_pickle(labels,"./labels.pkl")
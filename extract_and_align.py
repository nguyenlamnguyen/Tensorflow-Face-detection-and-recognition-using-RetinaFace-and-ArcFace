import cv2
import os
import numpy as np
from utils import functions
from imutils import paths
from detection_model import RetinaFace
detector_model = RetinaFace.build_model()

data_path = "./data/raw/"
dest_path = "./data/processed/"

def align_crop_resize(data_dir,dest_dir):
    image_links = list(paths.list_images(data_path))
    total = 0
    # face = RetinaFace.extract_faces(image_links[0], model=detector_model,align=True)
    # print(type(face))
    # split_img_link = image_links[0].split("\\")
    # label = split_img_link[0].split("/")[-1]
    # dest_path = os.path.join(dest_dir,label)
    # filename = split_img_link[-1]
    # print(os.path.join(dest_path,filename))


    for image_link in image_links:
        # Create dest path for each class in the dataset
        split_img_link = image_link.split("\\")
        label = split_img_link[0].split("/")[-1]
        dest_path = os.path.join(dest_dir,label)
        # Get file name
        filename = split_img_link[-1]
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)
        
        face = RetinaFace.extract_faces(image_link, model=detector_model,align=True)
        if (len(face)!=0):
            cv2.imwrite(os.path.join(dest_path,filename), cv2.cvtColor(face[0], cv2.COLOR_RGB2BGR))
            total += 1
    print("Successfully process ",total," images")
align_crop_resize(data_path, dest_path)
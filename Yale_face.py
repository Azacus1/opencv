from PIL import Image
import cv2
import numpy as np
import zipfile
import os



path = './Newfloder/Datasets/yaleface.zip'
zip_object = zipfile.ZipFile(file=path, mode = 'r')
zip_object.extractall('./')
zip_object.close()


# PRE-PROCESSING IMAGE

def get_image():
    paths = [os.path.join('yaleface/train', f) for f in os.listdir('/yalefaces/train')]
    faces = []
    ids = []
    for path in paths:
        print(path)
        image = Image.open(path).convert('L')
        print(image)
        image_np = np.array(image, 'uint8')
        print(image_np)
        id = int(os.path.split(path)[1].split('.')[0].replace('subject',''))
        print(id)
        ids.append(id)
        faces.append(image_np)
    return np.array(ids), faces


# threshold: 1.7976931348623157e+308
# radius: 1
# neighbors: 8
# grid_x: 8
# grid_y: 8


# TRAINING LBPH classifier
def lbph(faces,ids):
    lbph_classifier = cv2.face.LBPHFaceRecognizer_create(radius = 4, neighbors=14, grid_x = 9, grid_y = 9)
    lbph_classifier.train(faces, ids)
    lbph_classifier.write('lbph_classifier.yml')
    return lbph_classifier




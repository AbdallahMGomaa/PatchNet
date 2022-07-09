import cv2
import random
from pathlib import Path
from os.path import join, isfile
from os import listdir
from mtcnn import MTCNN
import numpy as np
import math
from PIL import Image



def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)
    if type(test_representation) == list:
        test_representation = np.array(test_representation)
    euclidean_distance = np.linalg.norm(source_representation - test_representation)
    return euclidean_distance

def alignment_procedure(img, left_eye, right_eye):
	left_eye_x, left_eye_y = left_eye
	right_eye_x, right_eye_y = right_eye
	if left_eye_y > right_eye_y:
		point_3rd = (right_eye_x, left_eye_y)
		direction = -1
	else:
		point_3rd = (left_eye_x, right_eye_y)
		direction = 1
	a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
	b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
	c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))
	if b != 0 and c != 0:
		cos_a = (b*b + c*c - a*a)/(2*b*c)
		angle = np.arccos(cos_a)
		angle = (angle * 180) / math.pi
		if direction == -1:
			angle = 90 - angle
		img = Image.fromarray(img)
		img = np.array(img.rotate(direction * angle))
	return img

def mtcnn(img, align = True):
    FaceDetector = MTCNN()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resp = []
    detected_face = None
    img_region = [0, 0, img.shape[0], img.shape[1]]
    detections = FaceDetector.detect_faces(img_rgb)
    if len(detections) > 0:
        for detection in detections:
            x, y, w, h = detection["box"]
            detected_face = img[int(y):int(y+h), int(x):int(x+w)]
            img_region = [x, y, w, h]
            if align:
                keypoints = detection["keypoints"]
                left_eye = keypoints["left_eye"]
                right_eye = keypoints["right_eye"]
                detected_face = alignment_procedure(detected_face, left_eye, right_eye)
            resp.append((detected_face, img_region))
    return resp

def detect_face(img):	
	obj = mtcnn(img)
	if len(obj) > 0:
		face, region = obj[0]
	else:
		face = None
		region = [0, 0, img.shape[0], img.shape[1]]
	return face, region

def getFrames(img_path, dst_path):
    fileName = Path(img_path).parts[-2]
    videoCap = cv2.VideoCapture(img_path)
    success, image = videoCap.read()
    frames = []
    count_frames = 0
    while success:
        frames.append(image)
        success, image = videoCap.read()
        count_frames += 1
    i = 0
    indices = []
    while i < 3:
        n = random.randint(0,count_frames-1)
        if n not in indices:
            indices.append(n)
            face, _ = detect_face(frames[indices[i]])
            if face is not None:
                cv2.imwrite(join(dst_path,"frame{}_{}.jpg".format(fileName,indices[i])),face)
                i += 1

def prepareData(data_path, dst_path):
    folders = listdir(data_path)
    skip_files = ['5.avi','6.avi','HR_3.avi']
    for folder in folders:
        files = listdir(join(data_path,folder))
        i = 0
        for file in files:
            if isfile(join(data_path,folder,file)) and file not in skip_files:
                file_path = join(data_path,folder,file)
                dst_folder = join(dst_path,str(i))
                getFrames(file_path, dst_folder)
                i += 1

train_src_path = ''
test_src_path = ''
train_dst_path = ''
test_dst_path = ''




prepareData(train_src_path, train_dst_path)
prepareData(test_src_path, test_dst_path)

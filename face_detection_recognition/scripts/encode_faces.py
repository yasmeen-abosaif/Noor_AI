#!/usr/bin/env python3

import os
import pickle
import face_recognition
import cv2  # Added for image loading control

def encode_faces(input_dir, output_file):
    encodings = []
    names = []
    
    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        if not os.path.isdir(person_dir):
            continue
            
        for fn in os.listdir(person_dir):
            if not fn.lower().endswith(('jpg', 'jpeg', 'png')):
                continue
                
            path = os.path.join(person_dir, fn)
            
            # Load with OpenCV for better control
            image = cv2.imread(path)
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use face_recognition's detection
            boxes = face_recognition.face_locations(rgb, model='hog')
            if not boxes:
                continue
                
            encs = face_recognition.face_encodings(rgb, boxes)
            if encs:
                encodings.append(encs[0])
                names.append(person)

    with open(output_file, 'wb') as f:
        pickle.dump({'encodings': encodings, 'names': names}, f)

if __name__ == '__main__':
    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(pkg_dir, 'data', 'faces')
    out_file = os.path.join(pkg_dir, 'data', 'encodings.pickle')
    encode_faces(data_dir, out_file)

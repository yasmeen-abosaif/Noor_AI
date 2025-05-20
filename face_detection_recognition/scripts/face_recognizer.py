#!/usr/bin/env python3

import os
import pickle
import rospy
import cv2
import mediapipe as mp
import face_recognition  # Ensure this is imported
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

class MediaPipeFaceRecognizer:
    def __init__(self):
        rospy.init_node('face_recognizer')
        self.bridge = CvBridge()

        # MediaPipe Face Detection
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

        # Load face encodings
        pkg_dir = os.path.dirname(os.path.dirname(__file__))
        enc_path = os.path.join(pkg_dir, 'data', 'encodings.pickle')
        with open(enc_path, 'rb') as f:
            data = pickle.load(f)
        self.known_encs = data['encodings']
        self.known_names = data['names']
        self.tolerance = rospy.get_param('~tolerance', 0.6)

        # ROS Setup
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback)
        self.pub = rospy.Publisher('/face_recognizer/output_image', Image, queue_size=1)

    def callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Detect faces with MediaPipe
        results = self.mp_face.process(rgb_frame)
        boxes = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                boxes.append((x, y, width, height))

        # Recognize faces
        for (x, y, w, h) in boxes:
            # Convert to face_recognition format (top, right, bottom, left)
            face_location = [(y, x + w, y + h, x)]  # Crucial fix!
            
            # Get encodings using original image + location
            encs = face_recognition.face_encodings(
                rgb_frame, 
                known_face_locations=face_location,
                num_jitters=1
            )
            
            name = "Unknown"
            if encs:
                matches = face_recognition.compare_faces(
                    self.known_encs,
                    encs[0],
                    tolerance=self.tolerance
                )
                if True in matches:
                    name = self.known_names[matches.index(True)]

            # Draw results
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Publish output
        try:
            self.pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    fr = MediaPipeFaceRecognizer()
    rospy.spin()

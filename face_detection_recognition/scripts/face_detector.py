#!/usr/bin/env python3# 
# -*- coding: utf-8 -*-

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class FaceDetector:
    def __init__(self):
        self.bridge = CvBridge()
        haar = rospy.get_param('~haar_path')
        self.face_cascade = cv2.CascadeClassifier(haar)
        if self.face_cascade.empty():
            rospy.logerr('failed to load cascade at %s' % haar)
            rospy.signal_shutdown('cascade missing')
        in_topic = rospy.get_param('~input_image', '/usb_cam/image_raw')
        self.image_sub = rospy.Subscriber(in_topic, Image, self.image_callback)
        out_topic = rospy.get_param('~output_image', '/face_detector/output_image')
        self.image_pub = rospy.Publisher(out_topic, Image, queue_size=1)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('face_detection', frame)
        cv2.waitKey(1)
        out_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        self.image_pub.publish(out_msg)

if __name__ == '__main__':
    rospy.init_node('face_detector')
    FaceDetector()
    rospy.spin()


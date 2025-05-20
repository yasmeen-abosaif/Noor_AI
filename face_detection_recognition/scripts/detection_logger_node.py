#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from darknet_ros_msgs.msg import BoundingBoxes

class DetectionLogger:
    def __init__(self):
        rospy.init_node('detection_logger_node')

        # Subscribers
        rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.yolo_callback)
        rospy.Subscriber('/face_recognizer/recognized_faces', String, self.face_callback)
        rospy.Subscriber('/emotion_gender_detector/results', String, self.emotion_callback)

        self.yolo_objects = []
        self.faces = ""
        self.emotion_gender = ""

    def yolo_callback(self, msg):
        self.yolo_objects = [box.Class for box in msg.bounding_boxes]
        self.print_summary()

    def face_callback(self, msg):
        self.faces = msg.data
        self.print_summary()

    def emotion_callback(self, msg):
        self.emotion_gender = msg.data
        self.print_summary()

    def print_summary(self):
        rospy.loginfo("\n==================== DETECTION SUMMARY ====================")
        rospy.loginfo("🟡 YOLO Objects Detected:     {}".format(", ".join(self.yolo_objects) if self.yolo_objects else "None"))
        rospy.loginfo("🟢 Recognized Faces:          {}".format(self.faces if self.faces else "None"))
        rospy.loginfo("🔵 Emotion & Gender:          {}".format(self.emotion_gender if self.emotion_gender else "None"))
        rospy.loginfo("===========================================================\n")

if __name__ == '__main__':
    try:
        DetectionLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

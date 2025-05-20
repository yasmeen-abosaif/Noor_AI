#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import String # Simple publisher
# Uncomment the line below if you have set up the custom message
# from emotion_gender_detector.msg import PersonPerception # For structured output
from cv_bridge import CvBridge, CvBridgeError
import cv2
from deepface import DeepFace
import numpy as np
import time # For FPS calculation

class EnhancedPerceptionNode:
    def __init__(self):
        rospy.init_node('enhanced_perception_node', anonymous=True)

        # --- ROS Parameters - For better configurability (Recommendation IV.1) ---
        # Image input
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.output_debug_image = rospy.get_param("~output_debug_image", True)

        # Performance tuning (Recommendation II.4, II.5)
        self.process_every_n_frames = rospy.get_param("~process_every_n_frames", 1) # Process every frame by default
        self.resize_input_width = rospy.get_param("~resize_input_width", 0) # 0 means no resize, otherwise resize to this width

        # DeepFace and Model Parameters (Recommendation I.1, I.6)
        self.face_detector_backend = rospy.get_param("~face_detector_backend", "mtcnn") # Options: "opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe", "yolov8", "centerface"
        self.actions_to_perform = ['emotion', 'gender', 'race'] # Can add 'age'
        self.enforce_detection_deepface = rospy.get_param("~enforce_detection", True) # If False, DeepFace returns N/A for attributes if no face is detected

        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_time = time.time()

        # --- Subscribers ---
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1, # Process latest frame, drop older ones if callback is slow
            buff_size=2**24 # Default is 65536, increase if you have large images
        )

        # --- Publishers ---
        # Simple publishers (can be replaced by custom message publisher)
        self.emotion_pub = rospy.Publisher("perception/dominant_emotion", String, queue_size=10)
        self.gender_pub = rospy.Publisher("perception/dominant_gender", String, queue_size=10)
        self.race_pub = rospy.Publisher("perception/dominant_race", String, queue_size=10)

        # Custom message publisher (Recommendation IV.2)
        # Uncomment if PersonPerception.msg is set up
        # self.person_perception_pub = rospy.Publisher("perception/person_details", PersonPerception, queue_size=10)

        if self.output_debug_image:
            self.debug_image_pub = rospy.Publisher("perception/debug_image", Image, queue_size=1)

        rospy.loginfo("Enhanced Perception Node Initialized.")
        rospy.loginfo(f"  Input topic: {self.image_topic}")
        rospy.loginfo(f"  Process every: {self.process_every_n_frames} frames")
        rospy.loginfo(f"  Resize input width to: {'No resize' if self.resize_input_width == 0 else self.resize_input_width}")
        rospy.loginfo(f"  Face detector backend: {self.face_detector_backend}")
        rospy.loginfo(f"  Actions: {self.actions_to_perform}")
        rospy.loginfo(f"  Enforce detection (DeepFace): {self.enforce_detection_deepface}")
        rospy.loginfo(f"  Output debug image: {self.output_debug_image}")

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return # Skip frame processing (Recommendation II.4)

        try:
            cv_image_original = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # --- Performance: Optional input image resizing (Recommendation II.5) ---
        if self.resize_input_width > 0 and cv_image_original.shape[1] > self.resize_input_width:
            aspect_ratio = cv_image_original.shape[0] / cv_image_original.shape[1]
            new_height = int(self.resize_input_width * aspect_ratio)
            cv_image_processed = cv2.resize(cv_image_original, (self.resize_input_width, new_height))
            # rospy.loginfo_throttle(5, f"Resized image from {cv_image_original.shape} to {cv_image_processed.shape}")
        else:
            cv_image_processed = cv_image_original.copy() # Use a copy to avoid modifying the original if needed elsewhere

        # --- Core Perception Logic using DeepFace ---
        try:
            # Recommendation I.1: Using a configurable (and potentially more robust) face detector.
            # DeepFace.analyze will detect faces and then perform the specified actions.
            # It returns a list of dictionaries, one for each detected face.
            # Note: Some backends like 'mtcnn' might need specific dependencies (e.g., Keras via mtcnn library).
            # Ensure these are installed: pip install mtcnn
            demographies = DeepFace.analyze(
                img_path=cv_image_processed,
                actions=self.actions_to_perform,
                detector_backend=self.face_detector_backend,
                enforce_detection=self.enforce_detection_deepface, # Recommendation I.6 (related to confidence)
                silent=True # Suppress DeepFace's own console logs for cleaner ROS logs
            )
            # DeepFace.analyze returns a list, even for one face. If no face, and enforce_detection=False,
            # it might return a list with a single dict containing N/A values.
            # If enforce_detection=True and no face, it raises an exception (handled below).

            # --- Handling Multiple Faces (Recommendation III.1) ---
            if isinstance(demographies, list) and len(demographies) > 0:
                for i, person_data in enumerate(demographies):
                    # Check if valid data was returned (not just N/A if enforce_detection=False)
                    if 'region' not in person_data:
                        # This can happen if enforce_detection=False and no face was found.
                        # The 'region' key is usually present if a face was processed.
                        rospy.logwarn_throttle(10, "No valid face region data in DeepFace result.")
                        continue

                    face_region = person_data['region'] # {'x': X, 'y': Y, 'w': W, 'h': H}
                    emotion = person_data.get('dominant_emotion', 'N/A')
                    gender = person_data.get('dominant_gender', 'N/A')
                    race = person_data.get('dominant_race', 'N/A')
                    # age = person_data.get('age', 'N/A') # If 'age' is in actions

                    rospy.loginfo_throttle(1, f"Person {i+1}: E:{emotion}, G:{gender}, R:{race} @ Region(x:{face_region['x']}, y:{face_region['y']})")

                    # Publish simple messages
                    self.emotion_pub.publish(emotion)
                    self.gender_pub.publish(gender)
                    self.race_pub.publish(race)

                    # --- Publish Custom Message (Recommendation IV.2) ---
                    # Uncomment and adapt if PersonPerception.msg is set up
                    # perception_msg = PersonPerception()
                    # perception_msg.header = msg.header # Use original image header for timestamp
                    # perception_msg.id = i # Simple ID, for tracking this would be a persistent ID
                    # perception_msg.face_roi.x_offset = face_region['x']
                    # perception_msg.face_roi.y_offset = face_region['y']
                    # perception_msg.face_roi.width = face_region['w']
                    # perception_msg.face_roi.height = face_region['h']
                    # perception_msg.emotion = emotion
                    # perception_msg.gender = gender
                    # perception_msg.dominant_race = race
                    # # perception_msg.emotion_confidence = person_data.get('emotion_confidence', 0.0) # If available
                    # self.person_perception_pub.publish(perception_msg)

                    # --- Enhanced Debug Image (Recommendation IV.3) ---
                    if self.output_debug_image:
                        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                        cv2.rectangle(cv_image_processed, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label_y = y - 10
                        cv2.putText(cv_image_processed, f"E: {emotion}", (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
                        label_y -= 15
                        cv2.putText(cv_image_processed, f"G: {gender}", (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
                        label_y -= 15
                        cv2.putText(cv_image_processed, f"R: {race}", (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

            elif self.enforce_detection_deepface:
                 rospy.logwarn_throttle(10, "No faces detected by DeepFace (enforce_detection=True).")


        except ValueError as ve: # Often raised by DeepFace if no face found and enforce_detection=True
            rospy.logwarn_throttle(5, f"DeepFace processing error (likely no face found or bad input): {ve}")
        except Exception as e:
            rospy.logerr(f"An unexpected error occurred during DeepFace analysis: {e}")


        # --- Publish Debug Image (Recommendation IV.3) ---
        if self.output_debug_image:
            # Add FPS to debug image
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
            self.last_time = current_time
            cv2.putText(cv_image_processed, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            try:
                debug_img_msg = self.bridge.cv2_to_imgmsg(cv_image_processed, "bgr8")
                self.debug_image_pub.publish(debug_img_msg)
            except CvBridgeError as e_bridge_debug:
                rospy.logerr(f"CvBridge Error on debug publish: {e_bridge_debug}")

        # --- Conceptual Note on Temporal Smoothing (Recommendation III.3) ---
        # To further improve robustness for video:
        # 1. Implement Face Tracking: Assign a unique ID to each face across frames.
        #    (e.g., using OpenCV trackers, or IoU matching with Kalman Filter).
        # 2. Store recent predictions (emotion, gender) for each tracked ID.
        # 3. Apply a smoothing filter (e.g., moving average, majority vote) over these
        #    recent predictions before finalizing the output for that person.
        # This is a more advanced step beyond this script's scope.

if __name__ == '__main__':
    try:
        node = EnhancedPerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Enhanced Perception Node shutting down.")
    finally:
        cv2.destroyAllWindows()

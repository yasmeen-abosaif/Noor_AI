#!/bin/bash
# test_face_detection.sh
export DISPLAY=:0
export QT_X11_NO_MITSHM=1

# Kill any existing nodes
rosnode kill -a 2>/dev/null

# Start nodes individually
rosrun turtlebot3_face_recognition detect_face_image.py \
    _haar_path:=$(rospack find turtlebot3_face_recognition)/data/haarcascade_frontalface_default.xml \
    _image_path:=$(rospack find turtlebot3_face_recognition)/data/test.jpg &

sleep 2

rosrun image_view image_view image:=/image_face_detector/output_image _autosize:=true &

wait

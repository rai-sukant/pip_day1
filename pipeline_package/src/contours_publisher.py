#!/usr/bin/env python

import cv2 as cv
import numpy as np
import rospy
from std_msgs.msg import Int32

def main():
    # Initialize the ROS node
    rospy.init_node('contour_publisher', anonymous=True)

    # Create ROS publishers for the x and y coordinates
    pub_x = rospy.Publisher('contour_x', Int32, queue_size=10)
    pub_y = rospy.Publisher('contour_y', Int32, queue_size=10)

    cap = cv.VideoCapture('/home/sukant/pipes.mp4')

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    ret, cv_image = cap.read()
    if not ret:
        print("Error: Failed to read the first frame.")
        exit()

    width = cv_image.shape[1]
    height = cv_image.shape[0]

    crop_width = int(width / 2)
    crop_height = int(height / 2)
    crop_x = width - crop_width
    crop_y = 0

    while not rospy.is_shutdown():
        ret, cv_image = cap.read()

        if not ret:
            print("Error: Failed to read frame.")
            break

        new_frame = cv_image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

        lower_rgb = (120, 120, 0)
        upper_rgb = (200, 200, 60)

        img_rgb = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)

        lower_threshold = np.array(lower_rgb, dtype=np.uint8)
        upper_threshold = np.array(upper_rgb, dtype=np.uint8)

        mask = cv.inRange(img_rgb, lower_threshold, upper_threshold)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv.contourArea(cnt) > 200:
                M = cv.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Publish the coordinates to ROS topics
                    pub_x.publish(cx)
                    pub_y.publish(cy)
                    print(f"Published coordinates: ({cx}, {cy})")
                else:
                    cx, cy = 0, 0

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
